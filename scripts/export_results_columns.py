from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELD_ORDER: list[str] = [
    "image",
    "plate",
    "brand",
    "type",
    "model_year",
    "engine_no",
    "chassis_no",
    "tax_or_id_no",
    "owner_surname",
    "owner_name",
    "first_registration_date",
    "registration_date",
    "inspection_date",
    "serial_no",
]


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def run() -> int:
    parser = argparse.ArgumentParser(
        description="Export OCR JSON outputs to a column-style CSV (one row per image)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing per-image JSON outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path (default: <input-dir>/results_columns.csv).",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    out_path = args.output or (input_dir / "results_columns.csv")

    json_paths = sorted(
        p
        for p in input_dir.glob("*.json")
        if p.name.lower() != "timing_summary.json"
    )

    rows: list[dict[str, str]] = []
    for path in json_paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        fields = data.get("fields") or {}

        image_path = _as_text(data.get("image")).replace("\\", "/")
        image_name = image_path.split("/")[-1] if image_path else f"{path.stem}.jpeg"

        row: dict[str, str] = {"image": image_name}
        for field_name in FIELD_ORDER[1:]:
            entry = fields.get(field_name) or {}
            row[field_name] = _as_text(entry.get("value"))
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_ORDER)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote={out_path.as_posix()} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
