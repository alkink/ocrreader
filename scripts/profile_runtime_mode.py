from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

from ocrreader.config import load_config
from ocrreader.pipeline import RuhsatOcrPipeline


def _env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import paddle  # type: ignore

        info["paddle_version"] = getattr(paddle, "__version__", "unknown")
        info["paddle_cuda_compiled"] = bool(paddle.device.is_compiled_with_cuda())
        try:
            info["paddle_device"] = paddle.device.get_device()
        except Exception as exc:  # pragma: no cover - diagnostic only
            info["paddle_device_error"] = repr(exc)
    except Exception as exc:  # pragma: no cover - optional dependency
        info["paddle_import_error"] = repr(exc)
    return info


def _flat_fields(result: dict[str, object]) -> dict[str, Any]:
    fields = result.get("fields", {})
    if not isinstance(fields, dict):
        return {}
    return {
        name: {
            "value": entry.get("value"),
            "method": entry.get("method"),
            "confidence_score": entry.get("confidence_score"),
        }
        for name, entry in fields.items()
        if isinstance(entry, dict)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile one OCR runtime mode on one image.")
    parser.add_argument("--label", required=True, help="Short label for the run")
    parser.add_argument("--image", required=True, help="Image path")
    parser.add_argument("--config", required=True, help="Config path")
    parser.add_argument("--runs", type=int, default=2, help="How many process_path runs to execute")
    parser.add_argument("--summary-out", default=None, help="Optional JSON summary output")
    parser.add_argument("--debug-dir", default=None, help="Optional debug dir for the final run only")
    args = parser.parse_args()

    cfg = load_config(args.config)

    t0 = time.perf_counter()
    pipeline = RuhsatOcrPipeline(cfg)
    init_seconds = time.perf_counter() - t0

    run_summaries: list[dict[str, Any]] = []
    final_result: dict[str, object] | None = None
    for idx in range(max(1, args.runs)):
        t1 = time.perf_counter()
        final_result = pipeline.process_path(
            args.image,
            debug_dir=args.debug_dir if idx == args.runs - 1 else None,
        )
        elapsed = time.perf_counter() - t1
        fields = _flat_fields(final_result)
        run_summaries.append(
            {
                "run_index": idx + 1,
                "seconds": round(elapsed, 3),
                "nonempty_fields": sum(1 for item in fields.values() if str(item.get("value") or "").strip()),
                "fields": fields,
            }
        )

    summary = {
        "label": args.label,
        "config": args.config,
        "image": args.image,
        "init_seconds": round(init_seconds, 3),
        "env": _env_info(),
        "runtime": (final_result or {}).get("runtime", {}) if isinstance(final_result, dict) else {},
        "runs": run_summaries,
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
