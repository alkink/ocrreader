from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .pipeline import RuhsatOcrPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ruhsat OCR extractor")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--config", default="config/ruhsat_schema.yaml", help="YAML config path")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--debug-dir", default=None, help="Optional debug image folder")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = load_config(args.config)
    pipeline = RuhsatOcrPipeline(config)
    result = pipeline.process_path(args.image, debug_dir=args.debug_dir)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

