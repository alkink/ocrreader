from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


PLATE_RE = re.compile(r"^\d{2}[A-Z]{1,3}\d{2,4}$")
DATE_RE = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")

LABEL_WORDS = {
    "MAKE",
    "MARKA",
    "MARKASI",
    "TYPE",
    "TIP",
    "TIPI",
    "ENGINE",
    "IDENTIFICATION",
    "PLAKA",
    "SOYADI",
    "ADI",
}

ADDRESS_TOKENS = (
    " CAD",
    " CAD.",
    " MAH",
    " MAH.",
    " ADRES",
    " SOK",
    " SOK.",
    " SK",
    " NO ",
    " KULTUR",
    " ALTINDAG",
)


def _norm(v: str) -> str:
    return (v or "").strip().upper()


def run() -> int:
    parser = argparse.ArgumentParser(description="Report suspicious annotation rows (line-by-line)")
    parser.add_argument("--annotations", default="dataset/generated/train_annotations.csv")
    parser.add_argument("--out-dir", default="dataset/generated/qa")
    args = parser.parse_args()

    ann_path = Path(args.annotations)
    if not ann_path.exists():
        raise RuntimeError(f"Annotations file not found: {ann_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detailed_rows: list[dict[str, str]] = []
    grouped: defaultdict[tuple[int, str], list[str]] = defaultdict(list)

    with ann_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for i, row in enumerate(rows, start=2):
        image = row.get("image", "")

        def add(field: str, reason: str, value: str) -> None:
            detailed_rows.append(
                {
                    "line_no": str(i),
                    "image": image,
                    "field": field,
                    "value": value,
                    "reason": reason,
                }
            )
            grouped[(i, image)].append(f"{field}:{reason}:{value}")

        plate = _norm(row.get("plate", "")).replace(" ", "")
        if plate and not PLATE_RE.fullmatch(plate):
            add("plate", "invalid_plate_pattern", row.get("plate", ""))

        for dfield in ("first_registration_date", "registration_date", "inspection_date"):
            dval = (row.get(dfield) or "").strip()
            if not dval:
                continue
            m = DATE_RE.fullmatch(dval)
            if not m:
                add(dfield, "bad_date_format", dval)
                continue
            dd, mm, yy = map(int, m.groups())
            if not (1 <= dd <= 31 and 1 <= mm <= 12 and 1950 <= yy <= 2035):
                add(dfield, "date_out_of_range", dval)

        brand = _norm(row.get("brand", ""))
        if brand in LABEL_WORDS:
            add("brand", "label_text_as_value", row.get("brand", ""))

        vtype = _norm(row.get("vehicle_type", ""))
        if vtype in LABEL_WORDS:
            add("vehicle_type", "label_text_as_value", row.get("vehicle_type", ""))

        for ofield in ("owner_name", "owner_surname"):
            oval = _norm(row.get(ofield, ""))
            if not oval:
                continue
            if "<BR/>" in oval:
                add(ofield, "contains_html", row.get(ofield, ""))
            if any(tok in oval for tok in ADDRESS_TOKENS):
                add(ofield, "address_like_text", row.get(ofield, ""))

        # serial format is noisy in OCR; flag only clearly non-serial alphabetic blobs without digits
        serial = _norm(row.get("serial_no", ""))
        if serial and not re.search(r"\d{3,7}", serial):
            add("serial_no", "no_serial_digits", row.get("serial_no", ""))

    detailed_csv = out_dir / "dataset_issues_line_by_line.csv"
    with detailed_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["line_no", "image", "field", "value", "reason"])
        w.writeheader()
        w.writerows(detailed_rows)

    grouped_csv = out_dir / "dataset_issue_rows.csv"
    with grouped_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line_no", "image", "issues"])
        for (line_no, image), issues in sorted(grouped.items(), key=lambda x: x[0][0]):
            w.writerow([line_no, image, " | ".join(dict.fromkeys(issues))])

    print(
        f"rows={len(rows)} issue_rows={len(grouped)} issue_items={len(detailed_rows)} "
        f"out={grouped_csv.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

