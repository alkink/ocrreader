from __future__ import annotations

from pathlib import Path
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
    config = OCRConfig(
        engine="paddle",
        glm_fallback_enabled=True,
        glm_model="paddle_vl",
    )

    print("Initializing integrated PaddleOCRVLEngine...")
    engine = create_glm_fallback_engine(config)
    if engine is None:
        print("Failed to create engine")
        return 1

    test_dir = PROJECT_ROOT / "testdata"
    out_dir = PROJECT_ROOT / "tests" / "batch_user_image_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        print("No images found in testdata")
        return 1

    timings: list[dict[str, object]] = []
    summary_path = out_dir / "timing_summary.json"

    def write_summary() -> None:
        ok_rows = [row for row in timings if row.get("ok")]
        avg_s = round(sum(float(row["seconds"]) for row in ok_rows) / max(1, len(ok_rows)), 3) if ok_rows else 0.0
        total_s = round(sum(float(row["seconds"]) for row in ok_rows), 3) if ok_rows else 0.0
        summary = {
            "processed": len(ok_rows),
            "total_images": len(timings),
            "avg_seconds": avg_s,
            "total_seconds": total_s,
            "rows": timings,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            timings.append({"image": image_path.name, "ok": False, "error": "imread_failed"})
            write_summary()
            continue

        started = time.perf_counter()
        result = engine.read_text(img)
        elapsed = round(time.perf_counter() - started, 3)

        out_path = out_dir / f"{image_path.stem}.txt"
        out_path.write_text(result, encoding="utf-8")

        timings.append(
            {
                "image": image_path.name,
                "ok": True,
                "seconds": elapsed,
                "output": str(out_path.relative_to(PROJECT_ROOT)),
            }
        )
        write_summary()
        print(f"[done] {image_path.name}: {elapsed:.3f}s", flush=True)

    print(f"Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
