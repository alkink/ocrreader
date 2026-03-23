from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


TR_MAP = str.maketrans(
    {
        "Ç": "C",
        "Ğ": "G",
        "İ": "I",
        "I": "I",
        "Ö": "O",
        "Ş": "S",
        "Ü": "U",
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
    }
)


def norm_text(value: str) -> str:
    v = (value or "").replace("\u00a0", " ").translate(TR_MAP)
    v = re.sub(r"\s+", " ", v).strip()
    return v.upper()


def norm_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", norm_text(value))


def parse_date(value: str) -> str:
    text = norm_text(value).replace(".", "/").replace("-", "/")
    m = re.search(r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{2,4})", text)
    if not m:
        return ""
    d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
    if len(y) == 2:
        y = f"20{y}"
    yy = int(y)
    if not (1 <= d <= 31 and 1 <= mo <= 12 and 1950 <= yy <= 2035):
        return ""
    return f"{d:02d}/{mo:02d}/{yy:04d}"


def clean_plate(value: str) -> str:
    p = re.sub(r"[^A-Z0-9]", "", norm_text(value))
    if not p:
        return ""
    if re.fullmatch(r"\d{2}[A-Z]{1,3}\d{2,4}", p):
        return p
    return ""


def clean_engine_or_chassis(value: str, field: str) -> str:
    t = re.sub(r"[^A-Z0-9]", "", norm_text(value))
    if not t:
        return ""
    if field == "engine_no" and len(t) < 6:
        return ""
    if field == "chassis_no" and len(t) < 12:
        return ""
    return t


def clean_tax(value: str) -> str:
    d = re.sub(r"\D", "", value or "")
    return d if len(d) >= 8 else ""


def clean_model_year(value: str) -> str:
    m = re.search(r"(19\d{2}|20\d{2})", norm_text(value))
    return m.group(1) if m else ""


def clean_serial(value: str) -> str:
    t = norm_text(value).replace("№", "NO")
    t = re.sub(r"\(\?\)", "", t)
    m = re.search(r"([A-Z]{1,3}\s*(?:NO|N)?\s*[:.]?\s*\d{3,7})", t)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    d = re.search(r"\d{3,7}", t)
    if d:
        return d.group(0)
    return ""


def clean_text(value: str) -> str:
    t = norm_text(value)
    t = re.sub(r"\(\?\)", "", t)
    t = re.sub(r"\s+", " ", t).strip(" -")
    if t in {"", "-", "OKUNAMADI", "OKUNMADI"}:
        return ""
    return t


def plate_near_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if len(a) != len(b):
        return False
    diff = sum(1 for x, y in zip(a, b) if x != y)
    return diff == 1


def normalize_existing_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "plate": clean_plate(row.get("plate", "")),
        "first_registration_date": parse_date(row.get("first_registration_date", "")),
        "registration_date": parse_date(row.get("registration_date", "")),
        "brand": clean_text(row.get("brand", "")),
        "vehicle_type": clean_text(row.get("vehicle_type", "")),
        "model_year": clean_model_year(row.get("model_year", "")),
        "engine_no": clean_engine_or_chassis(row.get("engine_no", ""), "engine_no"),
        "chassis_no": clean_engine_or_chassis(row.get("chassis_no", ""), "chassis_no"),
        "tax_or_id_no": clean_tax(row.get("tax_or_id_no", "")),
        "owner_name": clean_text(row.get("owner_name", "")),
        "owner_surname": clean_text(row.get("owner_surname", "")),
        "inspection_date": parse_date(row.get("inspection_date", "")),
        "serial_no": clean_serial(row.get("serial_no", "")),
    }


def pick_col(headers: list[str], patterns: list[str]) -> str | None:
    nmap = {norm_key(h): h for h in headers}
    for pat in patterns:
        for nk, orig in nmap.items():
            if pat in nk:
                return orig
    return None


def normalize_latest_row(row: dict[str, str], cols: dict[str, str | None]) -> dict[str, str]:
    def get(name: str) -> str:
        c = cols.get(name)
        return row.get(c, "") if c else ""

    return {
        "plate": clean_plate(get("plate")),
        "first_registration_date": parse_date(get("first_registration_date")),
        "registration_date": parse_date(get("registration_date")),
        "brand": clean_text(get("brand")),
        "vehicle_type": clean_text(get("vehicle_type")),
        "model_year": clean_model_year(get("model_year")),
        "engine_no": clean_engine_or_chassis(get("engine_no"), "engine_no"),
        "chassis_no": clean_engine_or_chassis(get("chassis_no"), "chassis_no"),
        "tax_or_id_no": clean_tax(get("tax_or_id_no")),
        "owner_name": clean_text(get("owner_name")),
        "owner_surname": clean_text(get("owner_surname")),
        "inspection_date": parse_date(get("inspection_date")),
        "serial_no": clean_serial(get("serial_no")),
    }


def run() -> int:
    parser = argparse.ArgumentParser(description="Match and merge latest_dataset.csv into train_annotations.csv")
    parser.add_argument("--base", default="dataset/generated/train_annotations.csv", help="Base train annotations CSV")
    parser.add_argument("--latest", default="dataset/generated/qa/latest_dataset.csv", help="Latest LLM OCR dataset CSV")
    parser.add_argument(
        "--output",
        default="dataset/generated/train_annotations_merged_latest.csv",
        help="Merged output annotations CSV",
    )
    parser.add_argument("--qa-dir", default="dataset/generated/qa", help="QA report output dir")
    parser.add_argument("--min-apply-score", type=int, default=80, help="Minimum match score to apply updates")
    args = parser.parse_args()

    base_path = Path(args.base)
    latest_path = Path(args.latest)
    if not base_path.exists():
        raise RuntimeError(f"Base CSV not found: {base_path}")
    if not latest_path.exists():
        raise RuntimeError(f"Latest CSV not found: {latest_path}")

    qa_dir = Path(args.qa_dir)
    qa_dir.mkdir(parents=True, exist_ok=True)

    with base_path.open("r", encoding="utf-8", newline="") as f:
        base_reader = csv.DictReader(f)
        base_rows = [{k: (v or "") for k, v in r.items()} for r in base_reader]
        base_headers = base_reader.fieldnames or []

    with latest_path.open("r", encoding="utf-8", newline="") as f:
        latest_reader = csv.DictReader(f)
        latest_rows = [{k: (v or "") for k, v in r.items()} for r in latest_reader]
        latest_headers = latest_reader.fieldnames or []

    if not base_rows or not latest_rows:
        raise RuntimeError("Base or latest dataset is empty")

    cols: dict[str, str | None] = {
        "plate": pick_col(latest_headers, ["PLAKA", "PLAKAA"]),
        "first_registration_date": pick_col(latest_headers, ["ILKTESCIL", "ILKTESCILB"]),
        "registration_date": pick_col(latest_headers, ["TESCILTAR", "TESCILTARI"]),
        "brand": pick_col(latest_headers, ["MARKAD1", "MARKA"]),
        "vehicle_type": pick_col(latest_headers, ["TIPD2", "TIP"]),
        "model_year": pick_col(latest_headers, ["MODELD4", "MODEL"]),
        "engine_no": pick_col(latest_headers, ["MOTORNOP5", "MOTORNO"]),
        "chassis_no": pick_col(latest_headers, ["SASENOE", "SASENO", "SASENO"]),
        "tax_or_id_no": pick_col(latest_headers, ["TCVKNY4", "TCVKN"]),
        "owner_surname": pick_col(latest_headers, ["UNVANSOYAD", "UNVANSOYADC11"]),
        "owner_name": pick_col(latest_headers, ["ADC12", "AD"]),
        "inspection_date": pick_col(latest_headers, ["DIGERMUAYENE", "MUAYENE"]),
        "serial_no": pick_col(latest_headers, ["BELGESERI", "SERI"]),
    }

    normalized_base = [normalize_existing_row(r) for r in base_rows]
    normalized_latest = [normalize_latest_row(r, cols) for r in latest_rows]

    plate_idx: defaultdict[str, list[int]] = defaultdict(list)
    engine_idx: defaultdict[str, list[int]] = defaultdict(list)
    chassis_idx: defaultdict[str, list[int]] = defaultdict(list)

    for i, nr in enumerate(normalized_base):
        if nr["plate"]:
            plate_idx[nr["plate"]].append(i)
        if nr["engine_no"]:
            engine_idx[nr["engine_no"]].append(i)
        if nr["chassis_no"]:
            chassis_idx[nr["chassis_no"]].append(i)

    assigned_counts: defaultdict[int, int] = defaultdict(int)
    match_report: list[dict[str, str]] = []
    change_rows: list[dict[str, str]] = []
    unmatched_rows: list[dict[str, str]] = []

    apply_fields = [
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
    ]

    for li, lr in enumerate(normalized_latest, start=2):
        scores: defaultdict[int, int] = defaultdict(int)
        reasons: defaultdict[int, list[str]] = defaultdict(list)

        if lr["plate"] and lr["plate"] in plate_idx:
            for bi in plate_idx[lr["plate"]]:
                scores[bi] += 120
                reasons[bi].append("plate")

        if lr["chassis_no"] and lr["chassis_no"] in chassis_idx:
            for bi in chassis_idx[lr["chassis_no"]]:
                scores[bi] += 100
                reasons[bi].append("chassis")

        if lr["engine_no"] and lr["engine_no"] in engine_idx:
            for bi in engine_idx[lr["engine_no"]]:
                scores[bi] += 80
                reasons[bi].append("engine")

        # fuzzy plate if strict key did not match (e.g., one OCR char off)
        if lr["plate"] and not scores:
            for bi, br in enumerate(normalized_base):
                bp = br["plate"]
                if plate_near_match(lr["plate"], bp):
                    scores[bi] += 65
                    reasons[bi].append("plate_near")

        for bi in list(scores.keys()):
            br = normalized_base[bi]
            if lr["brand"] and br["brand"] and lr["brand"] == br["brand"]:
                scores[bi] += 12
                reasons[bi].append("brand")
            if lr["registration_date"] and br["registration_date"] and lr["registration_date"] == br["registration_date"]:
                scores[bi] += 10
                reasons[bi].append("reg_date")
            if lr["first_registration_date"] and br["first_registration_date"] and lr["first_registration_date"] == br["first_registration_date"]:
                scores[bi] += 8
                reasons[bi].append("first_reg_date")
            if lr["model_year"] and br["model_year"] and lr["model_year"] == br["model_year"]:
                scores[bi] += 5
                reasons[bi].append("model_year")

        if not scores:
            unmatched_rows.append(
                {
                    "latest_line": str(li),
                    "latest_plate": lr["plate"],
                    "latest_engine_no": lr["engine_no"],
                    "latest_chassis_no": lr["chassis_no"],
                }
            )
            match_report.append(
                {
                    "latest_line": str(li),
                    "matched_line": "",
                    "matched_image": "",
                    "score": "0",
                    "reasons": "",
                    "status": "unmatched",
                }
            )
            continue

        ranked = sorted(scores.keys(), key=lambda bi: (scores[bi], -assigned_counts[bi]), reverse=True)
        best_idx = ranked[0]
        best_score = scores[best_idx]
        second_score = scores[ranked[1]] if len(ranked) > 1 else -1

        ambiguous = len(ranked) > 1 and (best_score - second_score) <= 8
        if ambiguous:
            match_report.append(
                {
                    "latest_line": str(li),
                    "matched_line": str(best_idx + 2),
                    "matched_image": base_rows[best_idx].get("image", ""),
                    "score": str(best_score),
                    "reasons": "+".join(reasons[best_idx]),
                    "status": "ambiguous_skip",
                }
            )
            continue

        status = "matched_no_change"
        if best_score >= args.min_apply_score:
            changed = 0
            for field in apply_fields:
                new_v = lr.get(field, "")
                if not new_v:
                    continue
                old_v = base_rows[best_idx].get(field, "")
                old_n = normalize_existing_row({field: old_v}).get(field, "")
                if old_n == new_v:
                    continue
                base_rows[best_idx][field] = new_v
                changed += 1
                change_rows.append(
                    {
                        "latest_line": str(li),
                        "matched_line": str(best_idx + 2),
                        "image": base_rows[best_idx].get("image", ""),
                        "field": field,
                        "old": old_v,
                        "new": new_v,
                        "score": str(best_score),
                    }
                )

            status = "applied" if changed else "matched_no_change"
        else:
            status = "low_score_skip"

        assigned_counts[best_idx] += 1
        match_report.append(
            {
                "latest_line": str(li),
                "matched_line": str(best_idx + 2),
                "matched_image": base_rows[best_idx].get("image", ""),
                "score": str(best_score),
                "reasons": "+".join(reasons[best_idx]),
                "status": status,
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_headers)
        w.writeheader()
        w.writerows(base_rows)

    match_report_path = qa_dir / "latest_dataset_match_report.csv"
    with match_report_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["latest_line", "matched_line", "matched_image", "score", "reasons", "status"],
        )
        w.writeheader()
        w.writerows(match_report)

    changes_path = qa_dir / "latest_dataset_applied_changes.csv"
    with changes_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["latest_line", "matched_line", "image", "field", "old", "new", "score"],
        )
        w.writeheader()
        w.writerows(change_rows)

    unmatched_path = qa_dir / "latest_dataset_unmatched.csv"
    with unmatched_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["latest_line", "latest_plate", "latest_engine_no", "latest_chassis_no"])
        w.writeheader()
        w.writerows(unmatched_rows)

    applied_count = sum(1 for r in match_report if r["status"] == "applied")
    print(
        f"latest_rows={len(latest_rows)} matched={len(match_report) - len(unmatched_rows)} "
        f"applied_rows={applied_count} changes={len(change_rows)} unmatched={len(unmatched_rows)} "
        f"output={out_path.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

