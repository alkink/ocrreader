from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from ocrreader.config import OCRConfig
from ocrreader.ocr_engine import create_glm_fallback_engine


def _crop(img, roi: dict[str, int]):
    x = int(roi.get("x", 0))
    y = int(roi.get("y", 0))
    w = int(roi.get("w", 1))
    h = int(roi.get("h", 1))
    return img[y : y + h, x : x + w]


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile PaddleOCR-VL on selected field ROIs")
    parser.add_argument("--image", default="testdata/WhatsApp_Image_2026-03-03_at_18.31.01.jpeg")
    parser.add_argument("--result-json", default="output/result_glm_ready.json")
    parser.add_argument("--fields", default="type,owner_surname,owner_name,serial_no")
    parser.add_argument("--max-side", type=int, default=0)
    parser.add_argument("--summary", default="tests/profile_paddle_vl_roi_summary.json")
    parser.add_argument("--output-dir", default="tests/profile_paddle_vl_roi_outputs")
    args = parser.parse_args()

    config = OCRConfig(
        engine="paddle",
        glm_fallback_enabled=True,
        glm_model="paddle_vl",
        paddle_vl_use_layout_detection=False,
        paddle_vl_use_ocr_for_image_block=True,
        paddle_vl_max_side=(args.max_side if args.max_side > 0 else None),
    )

    engine = create_glm_fallback_engine(config)
    if engine is None:
        raise RuntimeError("Failed to create PaddleOCR-VL engine")

    image = cv2.imread(str(Path(args.image)))
    if image is None:
        raise RuntimeError(f"Failed to load image: {args.image}")

    data = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    field_names = [f.strip() for f in args.fields.split(",") if f.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for field_name in field_names:
        field = data.get("fields", {}).get(field_name)
        if not isinstance(field, dict):
            rows.append({"field": field_name, "ok": False, "error": "missing_field"})
            continue
        roi = field.get("roi") or field.get("value_bbox")
        if not isinstance(roi, dict):
            rows.append({"field": field_name, "ok": False, "error": "missing_roi"})
            continue

        crop = _crop(image, roi)
        t0 = time.perf_counter()
        text = engine.read_text(crop)
        dt = round(time.perf_counter() - t0, 3)
        out_path = out_dir / f"{field_name}.txt"
        out_path.write_text(text, encoding="utf-8")
        rows.append(
            {
                "field": field_name,
                "ok": True,
                "seconds": dt,
                "pred": text,
                "baseline_value": field.get("value"),
                "output": str(out_path),
            }
        )
        print(f"[roi-done] {field_name}: {dt:.3f}s")

    ok_rows = [r for r in rows if r.get("ok")]
    summary = {
        "image": args.image,
        "result_json": args.result_json,
        "fields": field_names,
        "total_seconds": round(sum(float(r["seconds"]) for r in ok_rows), 3),
        "avg_seconds": round(sum(float(r["seconds"]) for r in ok_rows) / max(1, len(ok_rows)), 3),
        "rows": rows,
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
