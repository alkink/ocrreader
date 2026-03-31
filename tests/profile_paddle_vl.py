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
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-pixels", type=int, default=0)
    parser.add_argument("--max-pixels", type=int, default=0)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--no-use-cache", action="store_true")
    parser.add_argument("--use-queues", action="store_true")
    parser.add_argument("--no-use-queues", action="store_true")
    parser.add_argument("--prompt-label", default=None)
    parser.add_argument("--runtime-profile", default="auto")
    parser.add_argument("--service-url", default=None)
    parser.add_argument("--service-model-name", default=None)
    parser.add_argument("--service-api-key", default=None)
    parser.add_argument("--service-max-concurrency", type=int, default=0)
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

    use_cache = True
    if args.no_use_cache:
        use_cache = False
    elif args.use_cache:
        use_cache = True

    use_queues = True
    if args.no_use_queues:
        use_queues = False
    elif args.use_queues:
        use_queues = True

    config = OCRConfig(
        engine="paddle",
        glm_fallback_enabled=True,
        glm_model="paddle_vl",
        paddle_vl_use_layout_detection=use_layout,
        paddle_vl_use_ocr_for_image_block=use_image_block_ocr,
        paddle_vl_runtime_profile=(str(args.runtime_profile).strip().lower() or "auto"),
        paddle_vl_service_url=(str(args.service_url).strip() if args.service_url else None),
        paddle_vl_service_model_name=(str(args.service_model_name).strip() if args.service_model_name else None),
        paddle_vl_service_api_key=(str(args.service_api_key).strip() if args.service_api_key else None),
        paddle_vl_service_max_concurrency=(args.service_max_concurrency if args.service_max_concurrency > 0 else None),
        paddle_vl_max_side=(args.max_side if args.max_side > 0 else None),
        paddle_vl_max_new_tokens=(args.max_new_tokens if args.max_new_tokens > 0 else None),
        paddle_vl_min_pixels=(args.min_pixels if args.min_pixels > 0 else None),
        paddle_vl_max_pixels=(args.max_pixels if args.max_pixels > 0 else None),
        paddle_vl_use_cache=use_cache,
        paddle_vl_use_queues=use_queues,
        paddle_vl_prompt_label=(str(args.prompt_label).strip() if args.prompt_label else None),
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
        "runtime_profile": args.runtime_profile,
        "effective_runtime_profile": getattr(engine, "runtime_profile", None),
        "effective_runtime_backend": getattr(engine, "runtime_backend", None),
        "service_url": args.service_url,
        "service_model_name": args.service_model_name,
        "service_max_concurrency": args.service_max_concurrency,
        "max_side": args.max_side,
        "max_new_tokens": args.max_new_tokens,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "use_cache": use_cache,
        "use_queues": use_queues,
        "prompt_label": args.prompt_label,
        "output": args.output,
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
