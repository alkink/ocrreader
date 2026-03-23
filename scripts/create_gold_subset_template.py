from __future__ import annotations

import argparse
import csv
from pathlib import Path


def run() -> int:
    parser = argparse.ArgumentParser(description="Create a manually-reviewable gold subset CSV template")
    parser.add_argument("--annotations", default="dataset/generated/train_annotations.csv", help="Source annotations CSV")
    parser.add_argument("--output", default="dataset/generated/qa/gold_subset_template.csv", help="Output template CSV")
    parser.add_argument("--max-rows", type=int, default=30, help="How many rows to include (0=all)")
    args = parser.parse_args()

    ann_path = Path(args.annotations)
    if not ann_path.exists():
        raise RuntimeError(f"Annotations file not found: {ann_path}")

    with ann_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [{k: (v or "") for k, v in row.items()} for row in reader]
        header = reader.fieldnames or []

    if not rows:
        raise RuntimeError("No rows found in annotations file")

    selected_rows = rows if args.max_rows <= 0 else rows[: args.max_rows]

    meta_cols = {"image", "source_markdown"}
    field_cols = [c for c in header if c not in meta_cols]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_headers = ["row_id", "image", "source_markdown"]
    for field in field_cols:
        out_headers.extend([f"label_{field}", f"gold_{field}"])
    out_headers.extend(["review_status", "review_notes"])

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_headers)
        writer.writeheader()

        for idx, row in enumerate(selected_rows, start=1):
            out: dict[str, str | int] = {
                "row_id": idx,
                "image": row.get("image", ""),
                "source_markdown": row.get("source_markdown", ""),
                "review_status": "",
                "review_notes": "",
            }
            for field in field_cols:
                value = row.get(field, "").strip()
                out[f"label_{field}"] = value
                out[f"gold_{field}"] = value
            writer.writerow(out)

    print(
        f"Gold template written: {output_path.as_posix()} | "
        f"rows={len(selected_rows)} | fields={len(field_cols)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

