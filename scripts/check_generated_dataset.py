from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELD_KEYWORDS = [
    "plaka",
    "şase",
    "sase",
    "motor no",
    "vergi",
    "adi",
    "adı",
    "soyadi",
    "soyadı",
    "tescil",
    "seri",
]


def run() -> int:
    parser = argparse.ArgumentParser(description="Check generated OCR dataset consistency")
    parser.add_argument("--generated-dir", default="dataset/generated", help="Generated dataset root")
    args = parser.parse_args()

    root = Path(args.generated_dir)
    manifest = root / "manifest.jsonl"
    qa_dir = root / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        raise RuntimeError(f"Manifest missing: {manifest}")

    records: list[dict] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

    issues: list[tuple[str, str]] = []
    coverage: list[tuple[str, int, int]] = []

    for r in records:
        image = Path(r["image"])
        raw_json = Path(r["raw_json"])
        markdown = Path(r["markdown"])

        if not image.exists():
            issues.append((image.name, "missing_image"))
        if not raw_json.exists():
            issues.append((image.name, "missing_raw_json"))
        if not markdown.exists():
            issues.append((image.name, "missing_markdown"))
            continue

        text = markdown.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            issues.append((image.name, "empty_markdown"))

        lower = text.lower()
        hits = sum(1 for k in FIELD_KEYWORDS if k in lower)
        coverage.append((image.name, hits, len(text)))

    with (qa_dir / "issues.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "issue"])
        w.writerows(issues)

    with (qa_dir / "coverage.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "field_keyword_hits", "text_length"])
        w.writerows(coverage)

    avg_hits = round(sum(x[1] for x in coverage) / max(1, len(coverage)), 2)
    low_hits = [x[0] for x in coverage if x[1] <= 1]

    summary = {
        "records": len(records),
        "issues_count": len(issues),
        "avg_field_keyword_hits": avg_hits,
        "low_hit_images_count": len(low_hits),
        "low_hit_images_preview": low_hits[:20],
        "qa_dir": str(qa_dir.as_posix()),
    }

    (qa_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

