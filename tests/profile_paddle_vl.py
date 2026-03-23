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


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile PaddleOCR-VL on one image")
    parser.add_argument("--image", default="testdata/WhatsApp_Image_2026-03-03_at_18.31.01.jpeg")
    parser.add_argument("--layout", action="store_true")
    parser.add_argument("--no-layout", action="store_true")
    parser.add_argument("--image-block-ocr", action="store_true")
    parser.add_argument("--no-image-block-ocr", action="store_true")
    parser.add_argument("--max-side", type=int, default=0)
    parser.add_argument("--output", default="tests/profile_paddle_vl_output.txt")
    parser.add_argument("--summary", default="tests/profile_paddle_vl_summary.json")
    args = parser.parse_args()

    use_layout = True
    if args.no_layout:
        use_layout = False
    elif args.layout:
        use_layout = True

    use_image_block_ocr = True
    if args.no_image_block_ocr:
        use_image_block_ocr = False
    elif args.image_block_ocr:
        use_image_block_ocr = True

    config = OCRConfig(
        engine="paddle",
        glm_fallback_enabled=True,
        glm_model="paddle_vl",
        paddle_vl_use_layout_detection=use_layout,
        paddle_vl_use_ocr_for_image_block=use_image_block_ocr,
        paddle_vl_max_side=(args.max_side if args.max_side > 0 else None),
    )

    engine = create_glm_fallback_engine(config)
    if engine is None:
        raise RuntimeError("Failed to create PaddleOCR-VL engine")

    image_path = Path(args.image)
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    t0 = time.perf_counter()
    text = engine.read_text(img)
    dt = round(time.perf_counter() - t0, 3)

    Path(args.output).write_text(text, encoding="utf-8")
    summary = {
        "image": str(image_path),
        "seconds": dt,
        "layout": use_layout,
        "image_block_ocr": use_image_block_ocr,
        "max_side": args.max_side,
        "output": args.output,
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
