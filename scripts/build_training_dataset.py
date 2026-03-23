from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


FIELD_ORDER = [
    "image",
    "plate",
    "first_registration_date",
    "registration_date",
    "brand",
    "vehicle_type",
    "model_year",
    "engine_no",
    "chassis_no",
    "tax_or_id_no",
    "owner_name",
    "owner_surname",
    "inspection_date",
    "serial_no",
    "source_markdown",
]


TR_MAP = str.maketrans(
    {
        "İ": "I",
        "I": "I",
        "ı": "i",
        "Ş": "S",
        "ş": "s",
        "Ğ": "G",
        "ğ": "g",
        "Ü": "U",
        "ü": "u",
        "Ö": "O",
        "ö": "o",
        "Ç": "C",
        "ç": "c",
        "Â": "A",
        "â": "a",
    }
)


def _clean(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("|", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" :-|\t")


def _normalize_for_search(text: str) -> str:
    text = text.translate(TR_MAP)
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for ln in text.split("\n"):
        ln = ln.replace("|", " | ")
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        lines.append(ln.upper())
    return "\n".join(lines)


def _find(patterns: list[str], text: str) -> str:
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return _clean(m.group(1))
    return ""


def _looks_like_label(line: str) -> bool:
    return bool(re.search(r"\(\s*[A-Z0-9.]+\s*\)", line))


def _extract_label_value_from_lines(
    lines: list[str],
    label_regexes: list[str],
    max_follow_lines: int,
    stop_contains: list[str],
) -> str:
    compiled = [re.compile(rx, re.IGNORECASE) for rx in label_regexes]

    for i, line in enumerate(lines):
        for creg in compiled:
            if not creg.search(line):
                continue

            # Table-aware extraction: label often in one column, value in next column.
            cells = [_clean(c) for c in line.split("|") if _clean(c)]
            for idx, cell in enumerate(cells):
                if not creg.search(cell):
                    continue

                inline = _clean(creg.sub("", cell))
                if inline and inline not in {"ADI", "SOYADI", "TICARI UNVANI"} and not _looks_like_label(inline):
                    return inline

                if idx + 1 < len(cells):
                    nxt = _clean(cells[idx + 1])
                    if nxt and not _looks_like_label(nxt):
                        return nxt

            # Inline fallback from full line tail.
            m = creg.search(line)
            if m:
                tail = _clean(line[m.end() :])
                if tail and tail not in {"ADI", "SOYADI", "TICARI UNVANI"} and not _looks_like_label(tail):
                    return tail

            # Multi-line fallback.
            parts: list[str] = []
            for j in range(i + 1, min(len(lines), i + 1 + max_follow_lines)):
                nxt_line = _clean(lines[j])
                if not nxt_line:
                    continue
                if _looks_like_label(nxt_line):
                    break
                if any(tok in nxt_line for tok in stop_contains):
                    break
                parts.append(nxt_line)
                if len(" ".join(parts)) > 120:
                    break

            if parts:
                return _clean(" ".join(parts))

    return ""


def _normalize_date(value: str) -> str:
    value = _clean(value)
    if not value:
        return ""
    value = value.replace(".", "/").replace("-", "/")
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", value)
    if not m:
        return value
    day, month, year = m.group(1), m.group(2), m.group(3)
    if len(year) == 2:
        year = f"20{year}"
    return f"{int(day):02d}/{int(month):02d}/{year}"


def _extract_owner_fields(text: str) -> tuple[str, str]:
    lines = text.splitlines()

    surname = _find(
        [
            r"\(\s*C\s*[.]?\s*[13]\s*[.]?\s*1\s*\)\s*SOY[A-Z/ ]{0,60}[:\-]?\s*([^\n|]{2,140})",
            r"\(\s*C\s*[.]?\s*[13]\s*[.]?\s*1\s*\)\s*SOY[A-Z/ ]{0,60}\s*\|\s*([^\n|]{2,140})",
        ],
        text,
    )
    if not surname:
        surname = _extract_label_value_from_lines(
            lines,
            label_regexes=[
                r"\(\s*C\s*[.]?\s*[13]\s*[.]?\s*1\s*\)\s*SOY",
                r"SOYADI",
                r"SOVADI",
                r"SOYADE",
            ],
            max_follow_lines=3,
            stop_contains=["(C.1.2)", "(C1.2)", " ADRES", " NOTER", " BELGE", " SERI", " (Z.", " MUA", " MUAYENE"],
        )

    name = _find(
        [
            r"\(\s*C\s*[.]?\s*[13]\s*[.]?\s*2\s*\)\s*ADI?\s*[:\-]?\s*([^\n|]{2,120})",
            r"\(\s*C\s*[.]?\s*[13]\s*[.]?\s*2\s*\)\s*ADI?\s*\|\s*([^\n|]{2,120})",
        ],
        text,
    )
    if not name:
        name = _extract_label_value_from_lines(
            lines,
            label_regexes=[r"\(\s*C\s*[.]?\s*[13]\s*[.]?\s*2\s*\)\s*ADI?", r"\bADI\b"],
            max_follow_lines=2,
            stop_contains=[" ADRES", " NOTER", " BELGE", " SERI", " (Z.", " MUA", " MUAYENE"],
        )

    surname = _clean(surname)
    name = _clean(name)

    # De-noise obvious header carry-over.
    surname = re.sub(r"^TICARI\s+UNVANI\s*[:\-]?\s*", "", surname).strip()
    name = re.sub(r"^ADI\s*[:\-]?\s*", "", name).strip()

    bad_tokens = ["ADRES", "NOTER", "DIGER BILGI", "MUA", "MUAYENE", "BELGE SERI"]
    if any(tok in surname for tok in bad_tokens):
        surname = ""
    if any(tok in name for tok in bad_tokens):
        name = ""

    if len(name) <= 1:
        name = ""
    if len(surname) <= 1:
        surname = ""

    return surname, name


def _extract_inspection_date(text: str) -> str:
    value = _find(
        [
            r"(?:MUA|MUAYENE|META|MOTA|MUS)\.?\s*G[EA]C[A-Z.]*\s*(?:TRH|TARIH(?:I|LIK)?)?\.?\s*[:\-]?\s*([0-9]{1,2}[./\-][0-9]{1,2}[./\-][0-9]{2,4})",
            r"MUAYENE\s*GECERLILIK\s*(?:TARIHI|TRH)?\s*[:\-]?\s*([0-9]{1,2}[./\-][0-9]{1,2}[./\-][0-9]{2,4})",
            r"MUA\.?\s*GEC\.?\s*TRH\.?\s*[:\-]?\s*([0-9]{1,2}[./\-][0-9]{1,2}[./\-][0-9]{2,4})",
        ],
        text,
    )
    return _normalize_date(value)


def parse_markdown(md_text: str) -> dict[str, str]:
    text = _normalize_for_search(md_text)
    owner_surname, owner_name = _extract_owner_fields(text)

    out: dict[str, str] = {
        "plate": _find(
            [
                r"\(A\)\s*PLAKA\s*[:\-]?\s*([A-Z0-9 ]{5,12})",
                r"\bPLAKA\b\s*[:\-]?\s*([A-Z0-9 ]{5,12})",
            ],
            text,
        ),
        "first_registration_date": _find(
            [
                r"\(B\)\s*ILK\s*TESCIL\s*TARIHI\s*[:\-]?\s*([0-9./\-]{8,12})",
                r"ILK\s*TESCIL\s*TARIHI\s*[:\-]?\s*([0-9./\-]{8,12})",
            ],
            text,
        ),
        "registration_date": _find(
            [
                r"\(I\)\s*TESCIL\s*TARIHI\s*[:\-]?\s*([0-9./\-]{8,12})",
                r"\bTESCIL\s*TARIHI\b\s*[:\-]?\s*([0-9./\-]{8,12})",
            ],
            text,
        ),
        "brand": _find(
            [
                r"\(D\.1\)\s*MARKA[SI]*\s*[:\-]?\s*([A-Z0-9 .\-]{2,40})",
                r"\bMARKA[SI]*\b\s*[:\-]?\s*([A-Z0-9 .\-]{2,40})",
            ],
            text,
        ),
        "vehicle_type": _find(
            [
                r"\(D\.[23]\)\s*TIPI\s*[:\-]?\s*([A-Z0-9 .\-/]{2,50})",
                r"\bTIPI\b\s*[:\-]?\s*([A-Z0-9 .\-/]{2,50})",
            ],
            text,
        ),
        "model_year": _find(
            [
                r"\(D\.4\)\s*MODEL\s*YILI\s*[:\-]?\s*([0-9]{4})",
                r"MODEL\s*YILI\s*[:\-]?\s*([0-9]{4})",
            ],
            text,
        ),
        "engine_no": _find(
            [
                r"\((?:P\.5|F\.?3)\)\s*MOTOR\s*NO\s*[:\-]?\s*([A-Z0-9\-]{6,30})",
                r"MOTOR\s*NO\s*[:\-]?\s*([A-Z0-9\-]{6,30})",
            ],
            text,
        ),
        "chassis_no": _find(
            [
                r"\((?:E|K)\)\s*SASE\s*NO\s*[:\-]?\s*([A-Z0-9\-]{8,30})",
                r"SASE\s*NO\s*[:\-]?\s*([A-Z0-9\-]{8,30})",
            ],
            text,
        ),
        "tax_or_id_no": _find(
            [
                r"VERGI\s*NO\s*[:\-]?\s*([0-9]{8,14})",
                r"T\.?C\.?\s*KIMLIK\s*NO\s*[:\-]?\s*([0-9]{8,14})",
            ],
            text,
        ),
        "owner_surname": owner_surname,
        "owner_name": owner_name,
        "inspection_date": _extract_inspection_date(text),
        "serial_no": _find(
            [
                r"BELGE\s*SERI\s*[:\-]?\s*([A-Z0-9 №N.:\-]{4,40})",
                r"\bSERI\b\s*[:\-]?\s*([A-Z0-9 №N.:\-]{4,40})",
            ],
            text,
        ),
    }

    out = {k: _clean(v) for k, v in out.items()}

    out["plate"] = re.sub(r"[^A-Z0-9]", "", out["plate"])
    out["engine_no"] = re.sub(r"[^A-Z0-9\-]", "", out["engine_no"])
    out["chassis_no"] = re.sub(r"[^A-Z0-9\-]", "", out["chassis_no"])
    out["tax_or_id_no"] = re.sub(r"[^0-9]", "", out["tax_or_id_no"])
    out["model_year"] = _find([r"([0-9]{4})"], out["model_year"])

    out["first_registration_date"] = _normalize_date(out["first_registration_date"])
    out["registration_date"] = _normalize_date(out["registration_date"])
    out["inspection_date"] = _normalize_date(out["inspection_date"])

    return out


def run() -> int:
    parser = argparse.ArgumentParser(description="Build trainable CSV from generated OCR markdown files")
    parser.add_argument("--generated-dir", default="dataset/generated", help="Generated dataset root")
    parser.add_argument("--output-csv", default="dataset/generated/train_annotations.csv", help="Output CSV")
    parser.add_argument("--output-jsonl", default="dataset/generated/train_annotations.jsonl", help="Output JSONL")
    args = parser.parse_args()

    root = Path(args.generated_dir)
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        raise RuntimeError(f"Manifest not found: {manifest}")

    rows: list[dict[str, str]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        rec = json.loads(line)
        md_path = Path(rec["markdown"])
        if not md_path.exists():
            continue

        md_text = md_path.read_text(encoding="utf-8", errors="ignore")
        fields = parse_markdown(md_text)

        row = {k: "" for k in FIELD_ORDER}
        row.update(fields)
        row["image"] = rec["image"]
        row["source_markdown"] = rec["markdown"]
        rows.append(row)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_ORDER)
        writer.writeheader()
        writer.writerows(rows)

    out_jsonl = Path(args.output_jsonl)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    coverage = {
        field: sum(1 for row in rows if row.get(field, "").strip())
        for field in FIELD_ORDER
        if field not in {"image", "source_markdown"}
    }

    summary = {
        "rows": len(rows),
        "coverage": coverage,
        "output_csv": out_csv.as_posix(),
        "output_jsonl": out_jsonl.as_posix(),
    }

    qa_dir = root / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "train_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

