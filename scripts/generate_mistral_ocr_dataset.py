from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

from mistralai import Mistral


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def image_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".bmp":
        return "image/bmp"
    if ext in {".tif", ".tiff"}:
        return "image/tiff"
    return "application/octet-stream"


def collect_images(photo_dir: Path) -> list[Path]:
    files = [p for p in photo_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: p.name.lower())


def response_to_dict(resp: Any) -> dict[str, Any]:
    if hasattr(resp, "model_dump"):
        return resp.model_dump()  # pydantic v2 style
    if hasattr(resp, "dict"):
        return resp.dict()  # pydantic v1 style
    if isinstance(resp, dict):
        return resp
    return json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))


def extract_markdown(raw: dict[str, Any]) -> str:
    pages = raw.get("pages", [])
    lines: list[str] = []
    for page in pages:
        md = page.get("markdown")
        if md:
            lines.append(md)
    if lines:
        return "\n\n".join(lines).strip()

    if isinstance(raw.get("markdown"), str):
        return str(raw["markdown"]).strip()

    text = raw.get("text")
    return str(text).strip() if text else ""


def run() -> int:
    parser = argparse.ArgumentParser(description="Generate OCR dataset using Mistral OCR API")
    parser.add_argument("--photo-dir", default="dataset/photo", help="Input image directory")
    parser.add_argument("--out-dir", default="dataset/generated", help="Output dataset directory")
    parser.add_argument("--model", default="mistral-ocr-latest", help="Mistral OCR model")
    parser.add_argument("--resume", action="store_true", help="Skip images already processed")
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in environment")

    photo_dir = Path(args.photo_dir)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    md_dir = out_dir / "markdown"
    manifest_path = out_dir / "manifest.jsonl"

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(photo_dir)
    if not images:
        raise RuntimeError(f"No images found in {photo_dir}")

    client = Mistral(api_key=api_key)

    processed = 0
    skipped = 0
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for image_path in images:
            stem = image_path.stem
            raw_out = raw_dir / f"{stem}.json"
            md_out = md_dir / f"{stem}.md"

            if args.resume and raw_out.exists() and md_out.exists():
                skipped += 1
                continue

            b64 = encode_file(image_path)
            mime = image_mime(image_path)

            resp = client.ocr.process(
                document={
                    "type": "image_url",
                    "image_url": f"data:{mime};base64,{b64}",
                },
                model=args.model,
                include_image_base64=True,
            )

            raw = response_to_dict(resp)
            markdown = extract_markdown(raw)

            raw_out.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            md_out.write_text(markdown, encoding="utf-8")

            record = {
                "image": str(image_path.as_posix()),
                "raw_json": str(raw_out.as_posix()),
                "markdown": str(md_out.as_posix()),
                "has_markdown": bool(markdown.strip()),
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1

            print(f"[OK] {image_path.name}")

    print(f"Done. processed={processed}, skipped={skipped}, total={len(images)}")
    print(f"Manifest: {manifest_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

