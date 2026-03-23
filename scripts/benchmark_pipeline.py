from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocrreader.config import load_config


FIELD_ALIAS = {
    "vehicle_type": "type",
}

# ── Worker-process globals ───────────────────────────────────────────────────
_WORKER_PIPELINE = None
_WORKER_EVAL_FIELDS: list[str] = []


def _worker_init(config_path: str, pipeline_module: str) -> None:
    """Initializer called once per worker process."""
    global _WORKER_PIPELINE, _WORKER_EVAL_FIELDS

    # Re-insert project root in case subprocess doesn't inherit it properly
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from ocrreader.config import load_config as _load_config

    cfg = _load_config(config_path)
    pipe_mod = importlib.import_module(pipeline_module)
    pipeline_cls = getattr(pipe_mod, "RuhsatOcrPipeline")
    _WORKER_PIPELINE = pipeline_cls(cfg)


class RowResult(NamedTuple):
    image_path: str
    pred_fields: dict  # src_key -> entry dict
    skipped: bool


def _process_row_worker(image_path: str) -> RowResult:
    """Called inside a worker process to run OCR on a single image."""
    global _WORKER_PIPELINE

    image_file = Path(image_path)
    if not image_file.exists():
        return RowResult(image_path=image_path, pred_fields={}, skipped=True)

    result = _WORKER_PIPELINE.process_path(image_path)
    pred_fields = result.get("fields", {}) if isinstance(result, dict) else {}
    return RowResult(image_path=image_path, pred_fields=pred_fields, skipped=False)


# ── Utility helpers ──────────────────────────────────────────────────────────

def _clean(s: str) -> str:
    s = s or ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_date(value: str) -> str:
    value = _clean(value)
    if not value:
        return ""
    value = value.replace(".", "/").replace("-", "/")
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", value)
    if not m:
        return value.upper()
    d, mth, y = m.group(1), m.group(2), m.group(3)
    if len(y) == 2:
        y = f"20{y}"
    return f"{int(d):02d}/{int(mth):02d}/{y}"


def normalize_field(field: str, value: str) -> str:
    value = _clean(value)
    if not value:
        return ""

    if field in {"first_registration_date", "registration_date", "inspection_date"}:
        return _normalize_date(value)

    if field in {"tax_or_id_no"}:
        return re.sub(r"\D", "", value)

    if field in {"model_year"}:
        m = re.search(r"(19\d{2}|20\d{2})", value)
        return m.group(1) if m else ""

    if field in {"engine_no", "chassis_no", "plate"}:
        return re.sub(r"[^A-Z0-9\-]", "", value.upper())

    if field in {"serial_no"}:
        nums = re.findall(r"\d{4,7}", value)
        return nums[-1] if nums else ""

    return re.sub(r"\s+", " ", value.upper()).strip()


def safe_div(n: int, d: int) -> float:
    return round((n / d), 4) if d else 0.0


def token_f1(gt: str, pred: str) -> float:
    gt_tokens = re.findall(r"[A-Z0-9]+", gt)
    pred_tokens = re.findall(r"[A-Z0-9]+", pred)
    if not gt_tokens and not pred_tokens:
        return 1.0
    if not gt_tokens or not pred_tokens:
        return 0.0

    gt_c = Counter(gt_tokens)
    pred_c = Counter(pred_tokens)
    overlap = sum(min(gt_c[t], pred_c[t]) for t in gt_c.keys() & pred_c.keys())
    if overlap == 0:
        return 0.0

    p = overlap / max(1, sum(pred_c.values()))
    r = overlap / max(1, sum(gt_c.values()))
    return (2 * p * r) / max(1e-9, p + r)


def relaxed_normalize_field(field: str, value: str) -> str:
    value = normalize_field(field, value)
    if not value:
        return ""

    if field == "serial_no":
        nums = re.findall(r"\d{3,7}", value)
        if nums:
            return nums[-1]

    if field in {"owner_name", "owner_surname", "brand", "vehicle_type"}:
        value = value.replace("-", " ").replace("/", " ").replace(".", " ")
        value = re.sub(r"\s+", " ", value).strip()

    return value


# ── Per-row metric computation (runs in main process after OCR) ──────────────

def _compute_row_metrics(
    row: dict,
    row_result: RowResult,
    eval_fields: list[str],
    gt_prefix: str,
) -> tuple[
    dict[str, str],        # pred_row
    list[dict],            # detail_rows
    list[dict],            # mismatch_rows
    list[dict],            # scored_records
    dict[str, dict],       # field metric increments
    bool,                  # row_all_equal
]:
    image_path = row_result.image_path
    pred_fields = row_result.pred_fields

    pred_row: dict[str, str] = {"image": image_path}
    detail_rows: list[dict] = []
    mismatch_rows: list[dict] = []
    scored_records: list[dict] = []
    field_increments: dict[str, dict] = {}
    row_all_equal = True

    for field in eval_fields:
        src_key = FIELD_ALIAS.get(field, field)
        pred_val = ""
        pred_conf = 0
        pred_method = ""
        pred_low_conf = False
        entry = pred_fields.get(src_key)
        if isinstance(entry, dict):
            pred_val = str(entry.get("value", "") or "")
            pred_conf = int(entry.get("confidence_score", 0) or 0)
            pred_method = str(entry.get("method", "") or "")
            pred_low_conf = bool(entry.get("low_confidence", False))

        gt_key = f"{gt_prefix}{field}" if gt_prefix else field
        gt_raw = row.get(gt_key, "")

        gt_norm = normalize_field(field, gt_raw)
        pred_norm = normalize_field(field, pred_val)
        gt_relaxed = relaxed_normalize_field(field, gt_raw)
        pred_relaxed = relaxed_normalize_field(field, pred_val)

        pred_row[field] = pred_norm
        detail_rows.append(
            {
                "image": image_path,
                "field": field,
                "gt": gt_norm,
                "pred": pred_norm,
                "confidence_score": pred_conf,
                "method": pred_method,
                "low_confidence": int(pred_low_conf),
            }
        )
        scored_records.append(
            {
                "field": field,
                "gt": gt_norm,
                "pred": pred_norm,
                "confidence_score": pred_conf,
            }
        )

        inc = {
            "total": 1,
            "gt_non_empty": 1 if gt_norm else 0,
            "pred_non_empty": 1 if pred_norm else 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "exact_all": 0,
            "relaxed_tp": 0,
            "token_f1_num": 0,
            "token_f1_den": 0,
        }

        equal = gt_norm == pred_norm
        if equal:
            inc["exact_all"] = 1
        else:
            row_all_equal = False

        if gt_relaxed and pred_relaxed and gt_relaxed == pred_relaxed:
            inc["relaxed_tp"] = 1

        if gt_norm or pred_norm:
            inc["token_f1_num"] = int(round(token_f1(gt_norm, pred_norm) * 10000))
            inc["token_f1_den"] = 1

        if gt_norm and pred_norm and equal:
            inc["tp"] = 1
        else:
            if pred_norm and (not gt_norm or pred_norm != gt_norm):
                inc["fp"] = 1
            if gt_norm and (not pred_norm or pred_norm != gt_norm):
                inc["fn"] = 1

        if (gt_norm or pred_norm) and not equal:
            mismatch_rows.append(
                {
                    "image": image_path,
                    "field": field,
                    "gt": gt_norm,
                    "pred": pred_norm,
                    "confidence_score": str(pred_conf),
                    "method": pred_method,
                }
            )

        field_increments[field] = inc

    return pred_row, detail_rows, mismatch_rows, scored_records, field_increments, row_all_equal


# ── Main ─────────────────────────────────────────────────────────────────────

def run() -> int:
    parser = argparse.ArgumentParser(description="Benchmark OCR pipeline against train annotations")
    parser.add_argument("--annotations", default="dataset/generated/train_annotations.csv", help="Ground truth CSV")
    parser.add_argument("--config", default="config/ruhsat_schema.yaml", help="Pipeline config YAML")
    parser.add_argument(
        "--pipeline-module",
        default="ocrreader.pipeline",
        help="Pipeline module path that exports RuhsatOcrPipeline (e.g. ocrreader.pipeline_v28)",
    )
    parser.add_argument("--output-dir", default="dataset/generated/qa/benchmark", help="Benchmark output directory")
    parser.add_argument("--max-images", type=int, default=0, help="Optional max images to process (0=all)")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Print progress/ETA every N images (0=disable).",
    )
    parser.add_argument(
        "--image-contains",
        default="",
        help="Optional comma-separated substrings. If set, only images containing one of these are evaluated.",
    )
    parser.add_argument(
        "--min-gt-fields",
        type=int,
        default=0,
        help="Optional minimum number of non-empty ground-truth fields per row (0=disabled).",
    )
    parser.add_argument(
        "--require-fields",
        default="",
        help="Optional comma-separated field list; each listed field must be non-empty in GT.",
    )
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        help="If set, only rows with non-empty review_status are evaluated.",
    )
    parser.add_argument(
        "--no-reviewed-only",
        action="store_true",
        help="Disable reviewed-only filtering even if enabled from YAML benchmark config.",
    )
    parser.add_argument(
        "--review-statuses",
        default="",
        help="Optional comma-separated review_status filter (e.g. ok,fixed).",
    )
    parser.add_argument(
        "--gt-prefix",
        default="",
        help="Optional GT column prefix, e.g. gold_ for gold subset template files.",
    )
    parser.add_argument(
        "--confidence-sweep",
        default="0,8,12,16,20,24,28,32",
        help="Comma-separated confidence thresholds for precision@coverage report.",
    )
    parser.add_argument(
        "--exclude-fields",
        default="",
        help=(
            "Comma-separated fields to exclude from evaluation "
            "(e.g. owner_name). Fields with benchmark_exclude:true in the "
            "schema YAML are also auto-excluded when --config is provided."
        ),
    )
    parser.add_argument(
        "--ignore-benchmark-config",
        action="store_true",
        help="Ignore benchmark defaults from YAML benchmark section.",
    )
    parser.add_argument(
        "--include-benchmark-excluded-fields",
        action="store_true",
        help="Evaluate fields even if they have benchmark_exclude:true in the schema YAML.",
    )
    parser.add_argument(
        "--anchor-debug",
        action="store_true",
        help=(
            "Dump OCR words inside each anchor search region to CSV, including "
            "best alias-token similarity score for debugging anchor misses."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes for OCR inference. "
            "Each worker loads its own GPU/CPU pipeline instance. "
            "Recommended: 2 for RTX 3060 (each worker uses ~1-2 GB VRAM). "
            "Use 1 (default) with GLM config since Ollama is serial. "
            "NOTE: --anchor-debug is disabled when workers > 1."
        ),
    )
    args = parser.parse_args()

    # anchor-debug requires single-worker mode
    if args.anchor_debug and args.workers > 1:
        print(
            "[benchmark] --anchor-debug is not supported with --workers > 1; forcing --workers 1.",
            file=sys.stderr,
        )
        args.workers = 1

    ann_path = Path(args.annotations)
    if not ann_path.exists():
        raise RuntimeError(f"Annotations not found: {ann_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with ann_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v or "") for k, v in row.items()})

    if not rows:
        raise RuntimeError("No annotation rows found")

    gt_prefix = args.gt_prefix.strip()
    if not gt_prefix and any(k.startswith("gold_") for k in rows[0].keys()):
        gt_prefix = "gold_"

    # ── Config'i erken yükle: benchmark defaults + excluded fields ──────────
    config = load_config(args.config)

    if args.ignore_benchmark_config:
        bench_reviewed_only = False
        bench_allowed_review_statuses = []
        bench_min_gt_fields = 0
        bench_respect_exclude = True
        bench_confidence_sweep = ""
    else:
        bench_cfg_raw = getattr(config, "benchmark", {})
        if hasattr(config, "get"):
            bench_cfg_raw = config.get("benchmark", bench_cfg_raw) or bench_cfg_raw

        if isinstance(bench_cfg_raw, dict):
            bench_reviewed_only = bool(bench_cfg_raw.get("reviewed_only", False))
            bench_allowed_review_statuses = [s for s in bench_cfg_raw.get("allowed_review_statuses", []) if str(s).strip()]
            bench_min_gt_fields = int(bench_cfg_raw.get("min_gt_fields", 0) or 0)
            bench_respect_exclude = bool(bench_cfg_raw.get("respect_benchmark_exclude", True))
            bench_confidence_sweep = str(bench_cfg_raw.get("confidence_sweep", "")).strip()
        else:
            bench_reviewed_only = bool(getattr(bench_cfg_raw, "reviewed_only", False))
            bench_allowed_review_statuses = [
                str(s).strip() for s in getattr(bench_cfg_raw, "allowed_review_statuses", ()) if str(s).strip()
            ]
            bench_min_gt_fields = int(getattr(bench_cfg_raw, "min_gt_fields", 0) or 0)
            bench_respect_exclude = bool(getattr(bench_cfg_raw, "respect_benchmark_exclude", True))
            bench_confidence_sweep = str(getattr(bench_cfg_raw, "confidence_sweep", "")).strip()

    if not args.reviewed_only and bench_reviewed_only:
        args.reviewed_only = True
    if not args.review_statuses and bench_allowed_review_statuses:
        args.review_statuses = ",".join(bench_allowed_review_statuses)
    if args.min_gt_fields == 0 and bench_min_gt_fields:
        args.min_gt_fields = bench_min_gt_fields
    if args.confidence_sweep == parser.get_default("confidence_sweep") and bench_confidence_sweep:
        args.confidence_sweep = bench_confidence_sweep
    if args.no_reviewed_only:
        args.reviewed_only = False
        args.review_statuses = ""

    yaml_excluded: set[str] = set()
    if bench_respect_exclude and not args.include_benchmark_excluded_fields:
        fields_cfg = getattr(config, "fields", {})
        if hasattr(config, "get"):
            fields_cfg = config.get("fields", fields_cfg) or fields_cfg
        if isinstance(fields_cfg, dict):
            for fname, fcfg in fields_cfg.items():
                if isinstance(fcfg, dict):
                    if fcfg.get("benchmark_exclude"):
                        yaml_excluded.add(fname)
                elif bool(getattr(fcfg, "benchmark_exclude", False)):
                    yaml_excluded.add(fname)

    cli_excluded = {f.strip() for f in args.exclude_fields.split(",") if f.strip()}
    all_excluded = yaml_excluded | cli_excluded
    # ─────────────────────────────────────────────────────────────────────────

    if gt_prefix:
        eval_fields = [
            c[len(gt_prefix) :]
            for c in rows[0].keys()
            if c.startswith(gt_prefix)
        ]
    else:
        eval_fields = [
            c
            for c in rows[0].keys()
            if c not in {"image", "source_markdown", "review_status", "review_notes"}
            and not c.startswith("label_")
            and not c.startswith("gold_")
        ]

    if not eval_fields:
        raise RuntimeError("No evaluable fields found")

    if all_excluded:
        removed = [f for f in eval_fields if f in all_excluded]
        eval_fields = [f for f in eval_fields if f not in all_excluded]
        if removed:
            print(f"[benchmark] Excluded from evaluation: {removed}", file=sys.stderr)

    if not eval_fields:
        raise RuntimeError("All fields were excluded — nothing to evaluate")

    required_fields = [f.strip() for f in args.require_fields.split(",") if f.strip()]
    unknown_required = [f for f in required_fields if f not in eval_fields]
    if unknown_required:
        raise RuntimeError(f"Unknown fields in --require-fields: {unknown_required}")

    has_review_status_col = "review_status" in rows[0]
    if (args.reviewed_only or args.review_statuses.strip()) and not has_review_status_col:
        print(
            "[benchmark] review_status column not found; disabling review-based filters.",
            file=sys.stderr,
        )
        args.reviewed_only = False
        args.review_statuses = ""

    allowed_review_statuses = [s.strip() for s in args.review_statuses.split(",") if s.strip()]
    if args.reviewed_only:
        rows = [r for r in rows if _clean(r.get("review_status", ""))]
    if allowed_review_statuses:
        allowed = {s.lower() for s in allowed_review_statuses}
        rows = [r for r in rows if _clean(r.get("review_status", "")).lower() in allowed]

    if args.image_contains.strip():
        needles = [n.strip().lower() for n in args.image_contains.split(",") if n.strip()]
        rows = [r for r in rows if any(n in r.get("image", "").lower() for n in needles)]

    if required_fields:
        rows = [
            r
            for r in rows
            if all(normalize_field(f, r.get(f, "")) for f in required_fields)
        ]

    if args.min_gt_fields > 0:
        rows = [
            r
            for r in rows
            if sum(1 for f in eval_fields if normalize_field(f, r.get(f, ""))) >= args.min_gt_fields
        ]

    if args.max_images > 0:
        rows = rows[: args.max_images]

    if not rows:
        raise RuntimeError("No rows left after filters")

    # ── Build image_path -> row lookup (for parallel mode) ──────────────────
    image_paths = [r.get("image", "") for r in rows]
    row_by_image: dict[str, dict] = {r.get("image", ""): r for r in rows}

    # ── Metrics accumulators ─────────────────────────────────────────────────
    metrics: dict[str, dict[str, int]] = {
        f: {
            "total": 0,
            "gt_non_empty": 0,
            "pred_non_empty": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "exact_all": 0,
            "relaxed_tp": 0,
            "token_f1_num": 0,
            "token_f1_den": 0,
        }
        for f in eval_fields
    }

    prediction_rows: list[dict[str, str]] = []
    prediction_detail_rows: list[dict[str, object]] = []
    mismatch_rows: list[dict[str, str]] = []
    scored_records: list[dict[str, object]] = []
    anchor_debug_rows: list[dict[str, object]] | None = [] if args.anchor_debug else None
    row_exact_count = 0
    processed = 0
    skipped = 0
    total_rows = len(rows)
    t0 = time.perf_counter()

    workers = max(1, args.workers)

    if args.progress_every > 0:
        print(
            f"[benchmark] Starting evaluation for {total_rows} images "
            f"(workers={workers})...",
            file=sys.stderr,
            flush=True,
        )

    def _report_progress(done: int) -> None:
        if args.progress_every > 0 and (done % args.progress_every == 0 or done == total_rows):
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            remain = max(0, total_rows - done)
            eta = (remain / rate) if rate > 0 else 0.0
            print(
                f"[benchmark] {done}/{total_rows} ({(done / total_rows) * 100:0.1f}%) "
                f"processed={processed} skipped={skipped} elapsed={elapsed:0.1f}s eta={eta:0.1f}s",
                file=sys.stderr,
                flush=True,
            )

    def _integrate_result(row_result: RowResult, idx: int) -> None:
        """Merge a RowResult into the accumulators (runs in main process)."""
        nonlocal processed, skipped, row_exact_count

        if row_result.skipped:
            skipped += 1
            _report_progress(idx)
            return

        row = row_by_image[row_result.image_path]

        # anchor_debug only runs in single-worker mode (args.anchor_debug=False otherwise)
        if anchor_debug_rows is not None:
            # In single-worker mode we re-run through the pipeline; this path
            # only triggers if --anchor-debug was passed (workers forced to 1).
            pass

        pred_row, detail_rows, mis_rows, sc_records, field_incs, row_all_equal = _compute_row_metrics(
            row, row_result, eval_fields, gt_prefix
        )

        prediction_rows.append(pred_row)
        prediction_detail_rows.extend(detail_rows)
        mismatch_rows.extend(mis_rows)
        scored_records.extend(sc_records)

        for field, inc in field_incs.items():
            m = metrics[field]
            for k, v in inc.items():
                m[k] += v

        if row_all_equal:
            row_exact_count += 1

        processed += 1
        _report_progress(idx)

    # ── Single-worker path (preserves anchor_debug support) ──────────────────
    if workers == 1:
        try:
            pipe_mod = importlib.import_module(args.pipeline_module)
            pipeline_cls = getattr(pipe_mod, "RuhsatOcrPipeline")
        except Exception as e:
            raise RuntimeError(f"Unable to load pipeline class from {args.pipeline_module}: {e}") from e

        pipeline = pipeline_cls(config)

        for idx, row in enumerate(rows, start=1):
            image_path = row.get("image", "")
            image_file = Path(image_path)
            if not image_file.exists():
                row_result = RowResult(image_path=image_path, pred_fields={}, skipped=True)
            else:
                ocr_result = pipeline.process_path(image_path, anchor_debug_rows=anchor_debug_rows)
                pred_fields = ocr_result.get("fields", {}) if isinstance(ocr_result, dict) else {}
                row_result = RowResult(image_path=image_path, pred_fields=pred_fields, skipped=False)

            _integrate_result(row_result, idx)

    # ── Multi-worker path ────────────────────────────────────────────────────
    else:
        # Resolve absolute config path so subprocess can find it regardless of cwd
        config_path_abs = str(Path(args.config).resolve())

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(config_path_abs, args.pipeline_module),
        ) as executor:
            # map() preserves submission order → deterministic CSV output.
            # chunksize batches items to reduce IPC overhead for large datasets.
            chunksize = max(1, total_rows // (workers * 4))
            results_iter = executor.map(
                _process_row_worker,
                image_paths,
                chunksize=chunksize,
            )
            for idx, row_result in enumerate(results_iter, start=1):
                _integrate_result(row_result, idx)

    # ── Compute field metrics ─────────────────────────────────────────────────
    field_metric_rows: list[dict[str, object]] = []
    for field in eval_fields:
        m = metrics[field]
        precision = safe_div(m["tp"], m["tp"] + m["fp"])
        recall = safe_div(m["tp"], m["tp"] + m["fn"])
        f1 = safe_div(int(2 * m["tp"]), int((2 * m["tp"]) + m["fp"] + m["fn"]))
        exact_labeled = safe_div(m["tp"], m["gt_non_empty"])
        exact_all = safe_div(m["exact_all"], m["total"])
        relaxed_exact_labeled = safe_div(m["relaxed_tp"], m["gt_non_empty"])
        token_f1_avg = round((m["token_f1_num"] / m["token_f1_den"]) / 10000, 4) if m["token_f1_den"] else 0.0

        field_metric_rows.append(
            {
                "field": field,
                "total": m["total"],
                "gt_non_empty": m["gt_non_empty"],
                "pred_non_empty": m["pred_non_empty"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "exact_match_labeled": exact_labeled,
                "exact_match_all": exact_all,
                "relaxed_exact_labeled": relaxed_exact_labeled,
                "token_f1_avg": token_f1_avg,
            }
        )

    total_tp = sum(metrics[f]["tp"] for f in eval_fields)
    total_fp = sum(metrics[f]["fp"] for f in eval_fields)
    total_fn = sum(metrics[f]["fn"] for f in eval_fields)

    summary = {
        "processed_images": processed,
        "skipped_images": skipped,
        "fields_evaluated": len(eval_fields),
        "fields_excluded": sorted(all_excluded),
        "micro_precision": safe_div(total_tp, total_tp + total_fp),
        "micro_recall": safe_div(total_tp, total_tp + total_fn),
        "row_exact_match": safe_div(row_exact_count, processed),
        "image_filter": args.image_contains,
        "min_gt_fields": args.min_gt_fields,
        "require_fields": required_fields,
        "gt_prefix": gt_prefix,
        "reviewed_only": args.reviewed_only,
        "review_statuses": allowed_review_statuses,
        "pipeline_module": args.pipeline_module,
        "output_dir": out_dir.as_posix(),
        "anchor_debug_enabled": bool(args.anchor_debug),
        "workers": workers,
    }

    field_metrics_csv = out_dir / "field_metrics.csv"
    with field_metrics_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "field",
                "total",
                "gt_non_empty",
                "pred_non_empty",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "exact_match_labeled",
                "exact_match_all",
                "relaxed_exact_labeled",
                "token_f1_avg",
            ],
        )
        w.writeheader()
        w.writerows(field_metric_rows)

    relaxed_csv = out_dir / "field_metrics_relaxed.csv"
    with relaxed_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "field",
                "gt_non_empty",
                "pred_non_empty",
                "exact_match_labeled",
                "relaxed_exact_labeled",
                "token_f1_avg",
            ],
        )
        w.writeheader()
        for r in field_metric_rows:
            w.writerow(
                {
                    "field": r["field"],
                    "gt_non_empty": r["gt_non_empty"],
                    "pred_non_empty": r["pred_non_empty"],
                    "exact_match_labeled": r["exact_match_labeled"],
                    "relaxed_exact_labeled": r["relaxed_exact_labeled"],
                    "token_f1_avg": r["token_f1_avg"],
                }
            )

    mismatches_csv = out_dir / "mismatches.csv"
    with mismatches_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "field", "gt", "pred", "confidence_score", "method"])
        w.writeheader()
        w.writerows(mismatch_rows)

    predictions_csv = out_dir / "predictions.csv"
    pred_fields_list = ["image", *eval_fields]
    with predictions_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pred_fields_list)
        w.writeheader()
        w.writerows(prediction_rows)

    pred_detail_csv = out_dir / "predictions_detail.csv"
    with pred_detail_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["image", "field", "gt", "pred", "confidence_score", "method", "low_confidence"],
        )
        w.writeheader()
        w.writerows(prediction_detail_rows)

    if anchor_debug_rows is not None:
        anchor_debug_csv = out_dir / "anchor_debug.csv"
        with anchor_debug_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "image",
                    "anchor",
                    "word_text",
                    "word_norm",
                    "word_conf",
                    "block_num",
                    "par_num",
                    "line_num",
                    "word_x",
                    "word_y",
                    "word_w",
                    "word_h",
                    "word_center_x",
                    "word_center_y",
                    "region_x",
                    "region_y",
                    "region_w",
                    "region_h",
                    "anchor_min_score",
                    "best_alias",
                    "best_alias_token_score",
                    "best_alias_pass",
                ],
            )
            w.writeheader()
            w.writerows(anchor_debug_rows)
        summary["anchor_debug_csv"] = anchor_debug_csv.as_posix()
        summary["anchor_debug_rows"] = len(anchor_debug_rows)

    threshold_values = sorted({int(v.strip()) for v in args.confidence_sweep.split(",") if v.strip()})
    gt_non_empty_total = sum(1 for rec in scored_records if rec["gt"])
    precision_cov_rows: list[dict[str, object]] = []
    precision_cov_field_rows: list[dict[str, object]] = []

    for thr in threshold_values:
        included = [rec for rec in scored_records if rec["pred"] and int(rec["confidence_score"]) >= thr]
        tp = sum(1 for rec in included if rec["gt"] and rec["pred"] == rec["gt"])
        returned = len(included)
        fp = returned - tp
        returned_labeled = sum(1 for rec in included if rec["gt"])
        precision_cov_rows.append(
            {
                "threshold": thr,
                "returned": returned,
                "tp": tp,
                "fp": fp,
                "precision": round(tp / returned, 4) if returned else 0.0,
                "recall": round(tp / gt_non_empty_total, 4) if gt_non_empty_total else 0.0,
                "coverage_all": round(returned / len(scored_records), 4) if scored_records else 0.0,
                "coverage_labeled": round(returned_labeled / gt_non_empty_total, 4) if gt_non_empty_total else 0.0,
            }
        )

        for field in eval_fields:
            f_records = [rec for rec in scored_records if rec["field"] == field]
            f_included = [rec for rec in f_records if rec["pred"] and int(rec["confidence_score"]) >= thr]
            f_gt_total = sum(1 for rec in f_records if rec["gt"])
            f_tp = sum(1 for rec in f_included if rec["gt"] and rec["pred"] == rec["gt"])
            f_ret = len(f_included)
            f_fp = f_ret - f_tp
            precision_cov_field_rows.append(
                {
                    "threshold": thr,
                    "field": field,
                    "returned": f_ret,
                    "tp": f_tp,
                    "fp": f_fp,
                    "precision": round(f_tp / f_ret, 4) if f_ret else 0.0,
                    "recall": round(f_tp / f_gt_total, 4) if f_gt_total else 0.0,
                }
            )

    precision_cov_csv = out_dir / "precision_coverage.csv"
    with precision_cov_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["threshold", "returned", "tp", "fp", "precision", "recall", "coverage_all", "coverage_labeled"],
        )
        w.writeheader()
        w.writerows(precision_cov_rows)

    precision_cov_field_csv = out_dir / "precision_coverage_by_field.csv"
    with precision_cov_field_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["threshold", "field", "returned", "tp", "fp", "precision", "recall"],
        )
        w.writeheader()
        w.writerows(precision_cov_field_rows)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
