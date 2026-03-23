from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import pytesseract
from pytesseract import Output

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocrreader.anchors import detect_anchors
from ocrreader.config import RuhsatConfig, load_config
from ocrreader.fields import extract_fields, resolve_field_rois
from ocrreader.io_utils import imread_color
from ocrreader.ocr_engine import OCREngine, create_ocr_engine
from ocrreader.preprocess import preprocess_document


FIELD_ALIAS = {
    "vehicle_type": "type",
}


def _clean(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s).strip()


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

    if field == "plate":
        return re.sub(r"[^A-Z0-9]", "", value.upper())

    if field in {"brand", "owner_name", "owner_surname", "vehicle_type"}:
        return re.sub(r"\s+", " ", value.upper()).strip()

    return re.sub(r"\s+", " ", value.upper()).strip()


def _get_field_value(fields: dict[str, object], field: str) -> str:
    src_key = FIELD_ALIAS.get(field, field)
    item = fields.get(src_key)
    if isinstance(item, dict):
        return str(item.get("value", "") or "")
    return ""


def _extract_on_document(
    document: object,
    cfg: RuhsatConfig,
    engine: OCREngine,
) -> tuple[list[object], dict[str, object], dict[str, object]]:
    words = engine.iter_words(document, psm=cfg.ocr.psm, min_conf=0.0)
    anchors = detect_anchors(words, cfg.anchors)
    rois = resolve_field_rois(document.shape, cfg.fields, anchors)
    fields = extract_fields(
        document,
        rois,
        cfg.fields,
        engine,
        page_words=words,
        anchor_matches=anchors,
        page_regex_fallback_enabled=cfg.pipeline.page_regex_fallback_enabled,
    )
    return words, anchors, fields


def _osd_info(document: object) -> tuple[int, float]:
    try:
        rgb = cv2.cvtColor(document, cv2.COLOR_BGR2RGB)
        info = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
        rotate = int(float(info.get("rotate", 0) or 0)) % 360
        conf = float(info.get("orientation_conf", 0.0) or 0.0)
        return rotate, conf
    except Exception:
        return 0, 0.0


def _parse_csv_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v or "") for k, v in row.items()})
    return rows


def _apply_row_filters(
    rows: list[dict[str, str]],
    cfg: RuhsatConfig,
    image_contains: str,
    reviewed_only: bool,
    no_reviewed_only: bool,
    review_statuses: str,
    min_gt_fields: int,
    ignore_benchmark_config: bool,
    gt_prefix: str,
    key_fields: list[str],
) -> tuple[list[dict[str, str]], bool, list[str], int]:
    if ignore_benchmark_config:
        bench_reviewed_only = False
        bench_statuses: list[str] = []
        bench_min_gt = 0
    else:
        bench_reviewed_only = bool(cfg.benchmark.reviewed_only)
        bench_statuses = [s for s in cfg.benchmark.allowed_review_statuses if s]
        bench_min_gt = int(cfg.benchmark.min_gt_fields or 0)

    final_reviewed_only = reviewed_only or bench_reviewed_only
    final_statuses = [s.strip() for s in review_statuses.split(",") if s.strip()]
    if not final_statuses and bench_statuses:
        final_statuses = [str(s).strip() for s in bench_statuses if str(s).strip()]

    final_min_gt = min_gt_fields if min_gt_fields > 0 else bench_min_gt

    if no_reviewed_only:
        final_reviewed_only = False
        final_statuses = []

    out = rows
    if image_contains.strip():
        needles = [n.strip().lower() for n in image_contains.split(",") if n.strip()]
        out = [r for r in out if any(n in r.get("image", "").lower() for n in needles)]

    has_review_status = bool(out and "review_status" in out[0])
    if (final_reviewed_only or final_statuses) and not has_review_status:
        final_reviewed_only = False
        final_statuses = []

    if final_reviewed_only:
        out = [r for r in out if _clean(r.get("review_status", ""))]
    if final_statuses:
        allowed = {s.lower() for s in final_statuses}
        out = [r for r in out if _clean(r.get("review_status", "")).lower() in allowed]

    if final_min_gt > 0:
        out = [
            r
            for r in out
            if sum(
                1
                for f in key_fields
                if normalize_field(f, r.get(f"{gt_prefix}{f}" if gt_prefix else f, ""))
            )
            >= final_min_gt
        ]

    return out, final_reviewed_only, final_statuses, final_min_gt


def run() -> int:
    parser = argparse.ArgumentParser(description="Audit orientation impact by comparing normal vs forced 180° flow")
    parser.add_argument("--annotations", default="dataset/generated/train_annotations.csv", help="Ground truth CSV")
    parser.add_argument("--config", default="config/ruhsat_schema.yaml", help="Pipeline config YAML")
    parser.add_argument(
        "--output-dir",
        default="dataset/generated/qa/orientation_audit",
        help="Directory for orientation audit outputs",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Optional max images to process (0=all)")
    parser.add_argument(
        "--image-contains",
        default="",
        help="Optional comma-separated substrings. If set, only matching images are processed.",
    )
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        help="If set, only rows with non-empty review_status are processed.",
    )
    parser.add_argument(
        "--no-reviewed-only",
        action="store_true",
        help="Disable reviewed-only filtering even if enabled by YAML benchmark config.",
    )
    parser.add_argument(
        "--review-statuses",
        default="",
        help="Optional comma-separated review_status filter (e.g. ok,fixed).",
    )
    parser.add_argument(
        "--min-gt-fields",
        type=int,
        default=0,
        help="Optional min number of non-empty GT key fields per row (0=disabled).",
    )
    parser.add_argument(
        "--ignore-benchmark-config",
        action="store_true",
        help="Ignore benchmark defaults from YAML benchmark section.",
    )
    parser.add_argument(
        "--gt-prefix",
        default="",
        help="Optional GT column prefix (e.g. gold_).",
    )
    parser.add_argument(
        "--key-fields",
        default="plate,brand,first_registration_date,registration_date",
        help="Comma-separated key fields for TP-like orientation effect estimate.",
    )
    args = parser.parse_args()

    ann_path = Path(args.annotations)
    if not ann_path.exists():
        raise RuntimeError(f"Annotations not found: {ann_path}")

    cfg = load_config(args.config)
    rows = _parse_csv_rows(ann_path)
    if not rows:
        raise RuntimeError("No annotation rows found")

    gt_prefix = args.gt_prefix.strip()
    if not gt_prefix and any(k.startswith("gold_") for k in rows[0].keys()):
        gt_prefix = "gold_"

    key_fields = [f.strip() for f in args.key_fields.split(",") if f.strip()]
    if not key_fields:
        raise RuntimeError("No key fields provided")

    rows, final_reviewed_only, final_statuses, final_min_gt = _apply_row_filters(
        rows=rows,
        cfg=cfg,
        image_contains=args.image_contains,
        reviewed_only=args.reviewed_only,
        no_reviewed_only=args.no_reviewed_only,
        review_statuses=args.review_statuses,
        min_gt_fields=args.min_gt_fields,
        ignore_benchmark_config=args.ignore_benchmark_config,
        gt_prefix=gt_prefix,
        key_fields=key_fields,
    )

    if args.max_images > 0:
        rows = rows[: args.max_images]
    if not rows:
        raise RuntimeError("No rows left after filters")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_cfg = replace(cfg.pipeline, orientation_osd_enabled=False)
    engine = create_ocr_engine(cfg.ocr)

    per_field = {
        f: {
            "gt_non_empty": 0,
            "pred_non_empty_normal": 0,
            "pred_non_empty_rot180": 0,
            "tp_like_normal": 0,
            "tp_like_rot180": 0,
        }
        for f in key_fields
    }

    processed = 0
    skipped = 0
    zero_anchor_normal = 0
    zero_anchor_rot180 = 0
    zero_anchor_recovered = 0
    zero_anchor_lost = 0
    anchor_improved = 0
    anchor_worsened = 0
    key_improved = 0
    key_worsened = 0
    key_tie = 0
    total_key_gt = 0
    total_key_tp_like_normal = 0
    total_key_tp_like_rot180 = 0

    out_rows: list[dict[str, object]] = []

    date_fields = [f for f in key_fields if "date" in f]

    for row in rows:
        image_path = row.get("image", "")
        if not image_path:
            skipped += 1
            continue
        image_file = Path(image_path)
        if not image_file.exists():
            skipped += 1
            continue

        image = imread_color(image_path)
        prep = preprocess_document(image, pipeline_cfg)
        normal_doc = prep.normalized_image
        rot180_doc = cv2.rotate(normal_doc, cv2.ROTATE_180)

        _, anchors_n, fields_n = _extract_on_document(normal_doc, cfg, engine)
        _, anchors_r, fields_r = _extract_on_document(rot180_doc, cfg, engine)

        anchor_n = len(anchors_n)
        anchor_r = len(anchors_r)

        if anchor_n == 0:
            zero_anchor_normal += 1
        if anchor_r == 0:
            zero_anchor_rot180 += 1
        if anchor_n == 0 and anchor_r > 0:
            zero_anchor_recovered += 1
        if anchor_n > 0 and anchor_r == 0:
            zero_anchor_lost += 1
        if anchor_r > anchor_n:
            anchor_improved += 1
        elif anchor_r < anchor_n:
            anchor_worsened += 1

        osd_rotate, osd_conf = _osd_info(normal_doc)

        key_score_n = 0
        key_score_r = 0
        key_pred_non_empty_n = 0
        key_pred_non_empty_r = 0

        row_out: dict[str, object] = {
            "image": image_path,
            "review_status": row.get("review_status", ""),
            "osd_rotate": osd_rotate,
            "osd_orientation_conf": round(osd_conf, 3),
            "anchors_normal": anchor_n,
            "anchors_rot180": anchor_r,
            "anchor_delta": anchor_r - anchor_n,
            "anchor_names_normal": "|".join(sorted(anchors_n.keys())),
            "anchor_names_rot180": "|".join(sorted(anchors_r.keys())),
            "zero_anchor_normal": int(anchor_n == 0),
            "zero_anchor_rot180": int(anchor_r == 0),
        }

        for field in key_fields:
            gt_key = f"{gt_prefix}{field}" if gt_prefix else field
            gt_norm = normalize_field(field, row.get(gt_key, ""))
            pred_n = normalize_field(field, _get_field_value(fields_n, field))
            pred_r = normalize_field(field, _get_field_value(fields_r, field))

            match_n = int(bool(gt_norm and pred_n == gt_norm))
            match_r = int(bool(gt_norm and pred_r == gt_norm))

            row_out[f"{field}_gt"] = gt_norm
            row_out[f"{field}_normal"] = pred_n
            row_out[f"{field}_rot180"] = pred_r
            row_out[f"{field}_match_normal"] = match_n
            row_out[f"{field}_match_rot180"] = match_r

            per_field[field]["gt_non_empty"] += int(bool(gt_norm))
            per_field[field]["pred_non_empty_normal"] += int(bool(pred_n))
            per_field[field]["pred_non_empty_rot180"] += int(bool(pred_r))
            per_field[field]["tp_like_normal"] += match_n
            per_field[field]["tp_like_rot180"] += match_r

            key_score_n += match_n
            key_score_r += match_r
            key_pred_non_empty_n += int(bool(pred_n))
            key_pred_non_empty_r += int(bool(pred_r))
            total_key_gt += int(bool(gt_norm))
            total_key_tp_like_normal += match_n
            total_key_tp_like_rot180 += match_r

        date_any_n = int(any(bool(row_out.get(f"{f}_normal", "")) for f in date_fields)) if date_fields else 0
        date_any_r = int(any(bool(row_out.get(f"{f}_rot180", "")) for f in date_fields)) if date_fields else 0

        row_out["date_any_non_empty_normal"] = date_any_n
        row_out["date_any_non_empty_rot180"] = date_any_r
        row_out["key_gt_matches_normal"] = key_score_n
        row_out["key_gt_matches_rot180"] = key_score_r
        row_out["key_gt_match_delta"] = key_score_r - key_score_n
        row_out["key_pred_non_empty_normal"] = key_pred_non_empty_n
        row_out["key_pred_non_empty_rot180"] = key_pred_non_empty_r
        row_out["key_pred_non_empty_delta"] = key_pred_non_empty_r - key_pred_non_empty_n

        if key_score_r > key_score_n:
            row_out["winner"] = "rot180"
            key_improved += 1
        elif key_score_r < key_score_n:
            row_out["winner"] = "normal"
            key_worsened += 1
        else:
            row_out["winner"] = "tie"
            key_tie += 1

        out_rows.append(row_out)
        processed += 1

    out_csv = out_dir / "orientation_audit.csv"
    headers: list[str]
    if out_rows:
        headers = list(out_rows[0].keys())
    else:
        headers = [
            "image",
            "review_status",
            "osd_rotate",
            "osd_orientation_conf",
            "anchors_normal",
            "anchors_rot180",
            "anchor_delta",
            "zero_anchor_normal",
            "zero_anchor_rot180",
        ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(out_rows)

    zero_anchor_csv = out_dir / "orientation_audit_zero_anchor_normal.csv"
    zero_anchor_rows = [r for r in out_rows if int(r.get("zero_anchor_normal", 0)) == 1]
    with zero_anchor_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(zero_anchor_rows)

    recovered_csv = out_dir / "orientation_audit_zero_anchor_recovered_by_rot180.csv"
    recovered_rows = [
        r
        for r in out_rows
        if int(r.get("zero_anchor_normal", 0)) == 1 and int(r.get("zero_anchor_rot180", 0)) == 0
    ]
    with recovered_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(recovered_rows)

    per_field_summary: dict[str, dict[str, object]] = {}
    for f in key_fields:
        s = per_field[f]
        gt_non_empty = int(s["gt_non_empty"])
        pred_n = int(s["pred_non_empty_normal"])
        pred_r = int(s["pred_non_empty_rot180"])
        tp_n = int(s["tp_like_normal"])
        tp_r = int(s["tp_like_rot180"])
        per_field_summary[f] = {
            "gt_non_empty": gt_non_empty,
            "pred_non_empty_normal": pred_n,
            "pred_non_empty_rot180": pred_r,
            "tp_like_normal": tp_n,
            "tp_like_rot180": tp_r,
            "tp_like_delta": tp_r - tp_n,
            "tp_like_recall_normal": round(tp_n / gt_non_empty, 4) if gt_non_empty else 0.0,
            "tp_like_recall_rot180": round(tp_r / gt_non_empty, 4) if gt_non_empty else 0.0,
        }

    summary = {
        "processed_images": processed,
        "skipped_images": skipped,
        "filters": {
            "image_contains": args.image_contains,
            "reviewed_only": final_reviewed_only,
            "review_statuses": final_statuses,
            "min_gt_fields": final_min_gt,
            "gt_prefix": gt_prefix,
            "key_fields": key_fields,
            "ignore_benchmark_config": bool(args.ignore_benchmark_config),
            "max_images": args.max_images,
        },
        "zero_anchor": {
            "normal": zero_anchor_normal,
            "rot180": zero_anchor_rot180,
            "recovered_by_rot180": zero_anchor_recovered,
            "lost_by_rot180": zero_anchor_lost,
        },
        "anchor_hit_comparison": {
            "improved_images": anchor_improved,
            "worsened_images": anchor_worsened,
            "tie_images": max(0, processed - anchor_improved - anchor_worsened),
        },
        "key_match_comparison": {
            "improved_images": key_improved,
            "worsened_images": key_worsened,
            "tie_images": key_tie,
            "gt_total": total_key_gt,
            "tp_like_normal": total_key_tp_like_normal,
            "tp_like_rot180": total_key_tp_like_rot180,
            "tp_like_delta": total_key_tp_like_rot180 - total_key_tp_like_normal,
            "tp_like_recall_normal": round(total_key_tp_like_normal / total_key_gt, 4) if total_key_gt else 0.0,
            "tp_like_recall_rot180": round(total_key_tp_like_rot180 / total_key_gt, 4) if total_key_gt else 0.0,
        },
        "per_field": per_field_summary,
        "artifacts": {
            "orientation_audit_csv": out_csv.as_posix(),
            "zero_anchor_normal_csv": zero_anchor_csv.as_posix(),
            "zero_anchor_recovered_csv": recovered_csv.as_posix(),
        },
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

