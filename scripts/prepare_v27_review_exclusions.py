from __future__ import annotations

import argparse
import csv
from pathlib import Path


_MOTO_BRANDS = {
    "ARORA",
    "CITYCOCO",
    "KUBA",
    "MONDIAL",
    "SYM",
}


def _norm(s: str) -> str:
    return (s or "").strip().upper()


def _is_gemini(row: dict[str, str]) -> bool:
    image = (row.get("image", "") or "").lower()
    return "gemini_generated_image" in image


def _is_motorcycle(row: dict[str, str]) -> bool:
    image = (row.get("image", "") or "").lower()
    brand = _norm(row.get("brand", ""))
    vehicle_type = _norm(row.get("vehicle_type", ""))

    if brand in _MOTO_BRANDS:
        return True

    if any(token in image for token in ("motosiklet", "sym-motor", "arora-motor", "motor-")):
        return True

    # Ruhsat class hints: motorcycles are usually L-category.
    if vehicle_type.startswith("L") and len(vehicle_type) >= 2 and vehicle_type[1].isdigit():
        return True

    return False


def run() -> int:
    parser = argparse.ArgumentParser(
        description="Create v27 annotations with review_status-based exclusions for motorcycle/Gemini rows"
    )
    parser.add_argument(
        "--input",
        default="dataset/generated/train_annotations_v2_corrected.csv",
        help="Input annotations CSV",
    )
    parser.add_argument(
        "--output",
        default="dataset/generated/train_annotations_v2_corrected_v27_excluded.csv",
        help="Output CSV with review_status column",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise RuntimeError(f"Input annotations not found: {in_path}")

    rows: list[dict[str, str]] = []
    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v or "") for k, v in row.items()})

    if not rows:
        raise RuntimeError("Input annotations are empty")

    fieldnames = list(rows[0].keys())
    if "review_status" not in fieldnames:
        fieldnames.append("review_status")

    excluded = 0
    excluded_gemini = 0
    excluded_moto = 0
    for row in rows:
        is_gemini = _is_gemini(row)
        is_moto = _is_motorcycle(row)
        if is_gemini or is_moto:
            row["review_status"] = "excluded"
            excluded += 1
            if is_gemini:
                excluded_gemini += 1
            if is_moto:
                excluded_moto += 1
        else:
            row["review_status"] = "ok"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    kept = len(rows) - excluded
    print(
        f"wrote={out_path.as_posix()} rows={len(rows)} kept_ok={kept} "
        f"excluded_total={excluded} excluded_moto={excluded_moto} excluded_gemini={excluded_gemini}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

