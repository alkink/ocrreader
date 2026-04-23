from __future__ import annotations

from html import unescape
import re

from .text_utils import collapse_spaces, normalize_turkish_ascii


_ADDRESS_HINTS = re.compile(r"\b(?:MAH\.?|MAHALLESI|CAD\.?|CADDESI|SOK\.?|SOKAK|NO\b|KAPI|MERKEZ|DENIZLI|ANKARA|ISTANBUL|IZMIR)\b")
_COMPANY_HINTS = (
    "TICARI",
    "UNVAN",
    "LIMITED",
    "LTD",
    "SIRKET",
    "SANAYI",
    "OTOMOTIV",
    "EMLAK",
    "PROMOSYON",
    "KIRTASIYE",
    "OFIS",
    "INSAAT",
    "TASIMACILIK",
    "TURIZM",
)
_GENERIC_VEHICLE_TYPE_HINTS = (
    "OTOMOBIL",
    "MOTOSIKLET",
    "MINIBUS",
    "KAMYONET",
    "KAMYON",
    "PANELVAN",
    "BISIKLET",
    "MOTORLU",
)
_OWNER_NOISE_HINTS = (
    "NOTER",
    "NOTERLIGI",
    "PLAKA",
    "TESCIL",
    "VERILDIGI",
    "IL/ILCE",
    "MENFAATI",
    "MERKEZEFENDI",
)
_OWNER_LOCATION_TOKENS = {
    "DENIZLI",
    "ANKARA",
    "ISTANBUL",
    "IZMIR",
    "BURSA",
    "ADANA",
    "ANTALYA",
    "KONYA",
    "MERKEZEFENDI",
}


def _norm(text: str) -> str:
    return collapse_spaces(normalize_turkish_ascii(unescape(text or "")))


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return _norm(text)


def _extract_tables(text: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    for table_html in re.findall(r"<table[^>]*>(.*?)</table>", text or "", flags=re.I | re.S):
        rows: list[list[str]] = []
        for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.I | re.S):
            cells = [
                _strip_html(cell_html)
                for cell_html in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.I | re.S)
            ]
            cleaned = [cell for cell in cells if cell]
            if cleaned:
                rows.append(cleaned)
        if rows:
            tables.append(rows)
    return tables


def _extract_lines(text: str) -> list[str]:
    text = re.sub(r"</(?:tr|td|th|table)>", "\n", text or "", flags=re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "\n", text)
    return [line for line in (_norm(part) for part in re.split(r"[\r\n]+", text)) if line]


def _clean_vehicle_type(value: str) -> str:
    up = _norm(value)
    up = re.sub(r"\bP/?5\s+MOTOR\s+NO\b.*$", " ", up)
    up = re.sub(r"\bR\s+RENGI\b.*$", " ", up)
    up = collapse_spaces(up)
    up = up.replace("BISIKI LET", "BISIKLET")
    return up


def _clean_owner_title(value: str) -> str:
    up = _norm(value)
    match = re.search(r"TICARI\s+UNVAN[A-Z]*\s+(.+)$", up)
    if match:
        suffix = collapse_spaces(match.group(1))
        if suffix:
            return f"TICARI UNVASI {suffix}"
    up = re.sub(r"\bSOYADI/?\b", " ", up)
    up = collapse_spaces(up)
    return up


def _clean_owner_name(value: str) -> str:
    up = _norm(value)
    toks = [tok for tok in re.findall(r"[A-Z]+", up) if len(tok) >= 2]
    return collapse_spaces(" ".join(toks[:3]))


def _looks_like_address(value: str) -> bool:
    return bool(_ADDRESS_HINTS.search(_norm(value)))


def _has_company_hint(value: str) -> bool:
    up = _norm(value)
    return any(hint in up for hint in _COMPANY_HINTS)


def _looks_like_owner_noise(value: str) -> bool:
    up = _norm(value)
    if not up:
        return False
    if _looks_like_address(up):
        return True
    if any(hint in up for hint in _OWNER_NOISE_HINTS):
        return True
    tokens = [tok for tok in re.findall(r"[A-Z0-9]+", up) if tok]
    if len(tokens) >= 2 and len(set(tokens)) == 1:
        return True
    if tokens and not _has_company_hint(up) and all(tok in _OWNER_LOCATION_TOKENS for tok in tokens):
        return True
    return False


def _looks_like_person_name(value: str) -> bool:
    up = _norm(value)
    if not up or _has_company_hint(up) or _looks_like_owner_noise(up):
        return False
    tokens = [tok for tok in re.findall(r"[A-Z]+", up) if tok]
    if not tokens or len(tokens) > 3:
        return False
    return all(len(tok) >= 2 for tok in tokens)


def _looks_like_owner_title(value: str) -> bool:
    up = _norm(value)
    if not up or _looks_like_owner_noise(up):
        return False
    return _has_company_hint(up) or len(up) >= 5


def _looks_like_province(value: str) -> bool:
    up = _norm(value)
    return bool(up and "/" in up and "PLAKA" not in up and "TESCIL" not in up)


def _looks_like_brand(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9-]{2,40}", _norm(value)))


def _looks_like_date(value: str) -> bool:
    return bool(re.fullmatch(r"\d{2}[/-]\d{2}[/-]\d{4}", _norm(value)))


def _looks_like_vehicle_type(value: str) -> bool:
    up = _clean_vehicle_type(value)
    if not up or _looks_like_owner_noise(up):
        return False
    return len(re.sub(r"[^A-Z0-9]", "", up)) >= 3


def _looks_like_registration_serial(value: str) -> bool:
    up = re.sub(r"[^A-Z0-9]", "", _norm(value))
    return len(up) >= 10


def _looks_like_doc_serial(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{1,4}", _norm(value)))


def _looks_like_doc_number(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4,10}", _norm(value)))


def parse_right_page_vl_text(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    clean = _strip_html(text)
    lines = _extract_lines(text)

    for idx, line in enumerate(lines):
        up = _norm(line)
        if not up:
            continue

        if "KIMLIK" in up or "VERGI" in up:
            for candidate in [up] + lines[idx + 1 : idx + 3]:
                match = re.search(r"\b(\d{10,11})\b", _norm(candidate))
                if match:
                    out["tax_or_id_no"] = match.group(1)
                    break

        if "SOYADI" in up or "TICARI UNVAN" in up:
            candidate = re.split(r"SOYADI/?\s*TICARI\s+UNVAN[A-Z]*", up, maxsplit=1)[-1].strip()
            if not candidate and idx + 1 < len(lines):
                candidate = _norm(lines[idx + 1])
            owner_title = _clean_owner_title(candidate or up)
            if owner_title:
                out["owner_title"] = owner_title
                out["owner_surname"] = owner_title

        if (
            re.search(r"\bC[.,]?\s*1[.,]?\s*2\b", up)
            or re.search(r"^\(?C[.,]?\s*1[.,]?\s*2\)?\s*ADI", up)
            or re.search(r"\bADI\b", up)
        ):
            candidate = re.split(r"\bADI\b", up, maxsplit=1)[-1].strip()
            if not candidate and idx + 1 < len(lines):
                candidate = _norm(lines[idx + 1])
            owner_name = _clean_owner_name(candidate)
            if owner_name and _looks_like_person_name(owner_name):
                out["owner_name"] = owner_name

            address_lines: list[str] = []
            for follow in lines[idx + 1 : idx + 6]:
                follow_up = _norm(follow)
                if not follow_up:
                    continue
                if any(tag in follow_up for tag in ("NOTER", "BELGE", "MUA", "Z.1", "Z.2", "Z.3", "Z.4", "Z.5")):
                    break
                if _looks_like_address(follow_up):
                    address_lines.append(follow_up)
            if address_lines:
                out["owner_address"] = collapse_spaces(" ".join(address_lines))

        if "MUA" in up or "DIGER BILGILER" in up:
            for candidate in [up] + lines[idx + 1 : idx + 3]:
                match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", _norm(candidate))
                if match:
                    out["inspection_date"] = match.group(0)
                    break

        if ("SERI" in up or "BELGE" in up) and ("NO" in up or "Nº" in up or "N°" in up):
            serial_match = re.search(r"SERI[: ]*([A-Z]{1,4})\s+N[O0Âº°]*\s*(\d{4,8})", up)
            if serial_match:
                out["document_serial"] = serial_match.group(1)
                out["document_number"] = serial_match.group(2)
                out["serial_no"] = f"{serial_match.group(1)} NO {serial_match.group(2)}"

    tax_match = re.search(r"KIMLIK\s+NO/?VERGI\s+NO\s+(\d{10,11})", clean)
    if tax_match:
        out["tax_or_id_no"] = tax_match.group(1)

    serial_match = re.search(r"SERI[: ]*([A-Z]{1,4})\s+N[O0Âº°]*\s*(\d{4,8})", clean)
    if serial_match:
        out["document_serial"] = serial_match.group(1)
        out["document_number"] = serial_match.group(2)
        out["serial_no"] = f"{serial_match.group(1)} NO {serial_match.group(2)}"

    return out


def parse_d_block_vl_text(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    clean = _strip_html(text)
    lines = _extract_lines(text)

    for idx, line in enumerate(lines):
        up = _norm(line)
        if not up:
            continue

        if "MARKASI" in up and "brand" not in out:
            candidate = line.split("MARKASI", 1)[-1].strip()
            if not candidate and idx + 1 < len(lines):
                candidate = lines[idx + 1]
            candidate = collapse_spaces(_norm(candidate))
            if _looks_like_brand(candidate):
                out["brand"] = candidate

        if re.search(r"TIP[II]+\b", up):
            candidate = re.split(r"TIP[II]+\b", up, maxsplit=1)[-1].strip()
            if not candidate and idx + 1 < len(lines):
                candidate = _norm(lines[idx + 1])
            candidate = _clean_vehicle_type(candidate)
            compact = re.sub(r"[^A-Z0-9]", "", candidate)
            if candidate and compact and compact not in {"D2", "D4", "TIPI", "TIP"}:
                out["type"] = candidate

        if "CINSI" in up:
            candidate = re.split(r"CINSI", up, maxsplit=1)[-1].strip()
            if not candidate and idx + 1 < len(lines):
                candidate = _norm(lines[idx + 1])
            candidate = _clean_vehicle_type(candidate)
            compact = re.sub(r"[^A-Z0-9]", "", candidate)
            if candidate and compact and compact not in {"D5", "CINSI"}:
                out["vehicle_type"] = candidate

        if "MODEL YILI" in up and "model_year" not in out:
            match = re.search(r"\b(19\d{2}|20\d{2})\b", up)
            if not match and idx + 1 < len(lines):
                match = re.search(r"\b(19\d{2}|20\d{2})\b", _norm(lines[idx + 1]))
            if match:
                out["model_year"] = match.group(1)

        if "ARAC SINIFI" in up and "vehicle_class" not in out:
            match = re.search(r"\b([LMNO]\d)\b", up)
            if not match and idx + 1 < len(lines):
                match = re.search(r"\b([LMNO]\d)\b", _norm(lines[idx + 1]))
            if match:
                out["vehicle_class"] = match.group(1)

        if "RENGI" in up and "color" not in out:
            match = re.search(r"\b(BEYAZ|SIYAH|GRI|GUMUS|MAVI|KIRMIZI|YESIL|SARI)\b", up)
            if not match and idx + 1 < len(lines):
                match = re.search(r"\b(BEYAZ|SIYAH|GRI|GUMUS|MAVI|KIRMIZI|YESIL|SARI)\b", _norm(lines[idx + 1]))
            if match:
                out["color"] = match.group(1)

    brand_match = re.search(r"MARKASI\s+([A-Z0-9-]{2,40})", clean)
    if brand_match:
        out["brand"] = brand_match.group(1)

    model_year_match = re.search(r"MODEL\s+YILI\s+(19\d{2}|20\d{2})", clean)
    if model_year_match:
        out["model_year"] = model_year_match.group(1)

    vehicle_class_match = re.search(r"ARAC\s+SINIFI\s+([LMNO]\d)", clean)
    if vehicle_class_match:
        out["vehicle_class"] = vehicle_class_match.group(1)

    vehicle_type_match = re.search(r"(?:D/?5\s+)?CINSI\s+(.+?)\s+R\s+RENGI", clean)
    if vehicle_type_match:
        parsed_vehicle_type = _clean_vehicle_type(vehicle_type_match.group(1))
        if parsed_vehicle_type:
            out["vehicle_type"] = parsed_vehicle_type

    return out


def parse_full_page_vl_text(text: str) -> dict[str, str]:
    raw = _norm(text)
    clean = _strip_html(text)
    out: dict[str, str] = {}
    tables = _extract_tables(text)
    lines = _extract_lines(text)

    if tables:
        first_table = tables[0]
        if len(first_table) >= 2:
            first_row_values = first_table[1]
            if len(first_row_values) >= 2:
                if "plate" not in out:
                    plate_match = re.search(r"\b\d{2}[A-Z]{1,3}\d{2,4}\b", first_row_values[0])
                    if plate_match:
                        out["plate"] = plate_match.group(0)
                if "first_registration_date" not in out:
                    first_date_match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", " ".join(first_row_values[1:]))
                    if first_date_match:
                        out["first_registration_date"] = first_date_match.group(0).replace("-", "/")

        if len(first_table) >= 4:
            reg_row_values = first_table[3]
            if reg_row_values:
                if "registration_serial_no" not in out:
                    serial_match = re.search(r"\b\d{10,}\b", reg_row_values[0])
                    if serial_match:
                        out["registration_serial_no"] = serial_match.group(0)
                if len(reg_row_values) >= 2 and "registration_date" not in out:
                    reg_date_match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", " ".join(reg_row_values[1:]))
                    if reg_date_match:
                        out["registration_date"] = reg_date_match.group(0).replace("-", "/")

        if len(first_table) >= 6:
            brand_type_row = first_table[5]
            if brand_type_row:
                if "brand" not in out and len(brand_type_row) >= 1:
                    brand = collapse_spaces(brand_type_row[0])
                    if re.fullmatch(r"[A-Z0-9-]{2,40}", brand):
                        out["brand"] = brand
                if len(brand_type_row) >= 2:
                    type_candidate = collapse_spaces(brand_type_row[1])
                    if type_candidate:
                        out["type"] = type_candidate
                        out.setdefault("vehicle_type", type_candidate)

        if len(first_table) >= 8:
            misc_row = first_table[7]
            if len(misc_row) >= 2 and "model_year" not in out:
                model_year_match = re.search(r"\b(19\d{2}|20\d{2})\b", misc_row[1])
                if model_year_match:
                    out["model_year"] = model_year_match.group(1)
            if len(misc_row) >= 3 and "vehicle_class" not in out:
                vehicle_class_match = re.search(r"\b([LMNO]\d)\b", misc_row[2])
                if vehicle_class_match:
                    out["vehicle_class"] = vehicle_class_match.group(1)

        if len(first_table) >= 12:
            engine_row_values = first_table[10]
            if engine_row_values and "engine_no" not in out:
                engine_match = re.search(r"\b[A-Z0-9]{8,25}\b", engine_row_values[0])
                if engine_match:
                    out["engine_no"] = engine_match.group(0)
        if len(first_table) >= 14:
            chassis_row_values = first_table[13]
            if chassis_row_values and "chassis_no" not in out:
                chassis_match = re.search(r"\b[A-Z0-9]{15,18}\b", chassis_row_values[0])
                if chassis_match:
                    out["chassis_no"] = chassis_match.group(0)

    if len(tables) >= 2:
        owner_table = tables[1]
        for row in owner_table:
            joined = " ".join(row)
            if "tax_or_id_no" not in out:
                tax_match = re.search(r"\b(\d{10,11})\b", joined)
                if tax_match and ("KIMLIK" in joined or "VERGI" in joined):
                    out["tax_or_id_no"] = tax_match.group(1)

            if "SOYADI" in joined or "TICARI UNVAN" in joined:
                owner_title = _clean_owner_title(joined)
                if owner_title:
                    out["owner_title"] = owner_title
                    out["owner_surname"] = owner_title

            if any("ADI" in cell for cell in row):
                if len(row) >= 2:
                    owner_name = _clean_owner_name(row[-1])
                    if owner_name:
                        out["owner_name"] = owner_name

    province_match = re.search(r"VERILDIGI\s+IL/?ILCE\s+(.+?)\s+\(?A\)?\s+PLA[KR]A", clean)
    if province_match:
        out["province_district"] = collapse_spaces(province_match.group(1))

    plate_match = re.search(r"\b\d{2}[A-Z]{1,3}\d{2,4}\b", clean)
    if plate_match:
        out["plate"] = plate_match.group(0)

    first_reg_match = re.search(r"ILK\s+TESCIL\s+TAR[A-Z]*\s+(\d{2}[/-]\d{2}[/-]\d{4})", clean)
    if first_reg_match:
        out["first_registration_date"] = first_reg_match.group(1).replace("-", "/")

    reg_date_match = re.search(r"TESCIL\s+TAR[A-Z]*\s+(\d{2}[/-]\d{2}[/-]\d{4})", clean)
    if reg_date_match:
        out["registration_date"] = reg_date_match.group(1).replace("-", "/")

    reg_serial_match = re.search(r"TESCIL\s+SIRA\s+N[O0]\s+(\d{10,})", clean)
    if reg_serial_match:
        out["registration_serial_no"] = reg_serial_match.group(1)

    brand_match = re.search(r"MARKASI\s+([A-Z0-9-]{2,40})", clean)
    if brand_match:
        out["brand"] = brand_match.group(1)

    type_match = re.search(r"TIP[II]+\s+([A-Z0-9.-]{2,20}(?:\s+[A-Z0-9.-]{1,20}){0,4})", clean)
    if type_match:
        out["type"] = collapse_spaces(type_match.group(1))

    model_year_match = re.search(r"MODEL\s+YILI\s+(19\d{2}|20\d{2})", clean)
    if model_year_match:
        out["model_year"] = model_year_match.group(1)

    vehicle_class_match = re.search(r"ARAC\s+SINIFI\s+([LMNO]\d)", clean)
    if vehicle_class_match:
        out["vehicle_class"] = vehicle_class_match.group(1)

    vehicle_type_match = re.search(r"(?:D/?5\s+)?CINSI\s+(.+?)\s+R\s+RENGI", clean)
    if vehicle_type_match:
        parsed_vehicle_type = _clean_vehicle_type(vehicle_type_match.group(1))
        if parsed_vehicle_type:
            out["vehicle_type"] = parsed_vehicle_type

    color_match = re.search(r"RENGI\s+(BEYAZ|SIYAH|GRI|GUMUS|MAVI|KIRMIZI|YESIL|SARI)", clean)
    if color_match:
        out["color"] = color_match.group(1)

    engine_match = re.search(r"MOTOR\s+NO\s+([A-Z0-9]{8,25})", clean)
    if engine_match:
        out["engine_no"] = engine_match.group(1)

    chassis_match = re.search(r"SASE\s+NO\s+([A-Z0-9]{15,18})", clean)
    if chassis_match:
        out["chassis_no"] = chassis_match.group(1)

    tax_match = re.search(r"KIMLIK\s+NO/?VERGI\s+NO\s+(\d{10,11})", clean)
    if tax_match:
        out["tax_or_id_no"] = tax_match.group(1)

    owner_title_match = re.search(r"SOYADI/?TICARI\s+UNVAN[A-Z]*\s+(.+?)\s+C[.,]?1[.,]?2\s+ADI", clean)
    if owner_title_match:
        owner_title = _clean_owner_title(owner_title_match.group(1))
        if owner_title:
            out["owner_title"] = owner_title
            out["owner_surname"] = owner_title

    owner_name_match = re.search(
        r"C[.,]?1[.,]?2\s+ADI\s+([A-Z]+(?:\s+[A-Z]+){0,2})\s+(?=[A-Z0-9./-]+\s+MAH|[A-Z]+\s+CAD|Z[.,]?\d|BELGE)",
        clean,
    )
    if owner_name_match:
        owner_name = _clean_owner_name(owner_name_match.group(1))
        if owner_name:
            out["owner_name"] = owner_name

    address_match = re.search(
        r"C[.,]?1[.,]?2\s+ADI\s+[A-Z]+(?:\s+[A-Z]+){0,2}\s+(.+?)\s+(?:Z[.,]?\d|BELGE)",
        clean,
    )
    if address_match:
        address = collapse_spaces(address_match.group(1))
        if _looks_like_address(address):
            out["owner_address"] = address

    inspection_match = re.search(r"MUA[^0-9]{0,30}(\d{2}[/-]\d{2}[/-]\d{4})", clean)
    if inspection_match:
        out["inspection_date"] = inspection_match.group(1)

    fuel_match = re.search(r"YAKIT\s+CINSI\s+(DIZEL|BENZIN|ELEKTRIK|HIBRIT|LPG|CNG)", clean)
    if fuel_match:
        out["fuel_type"] = fuel_match.group(1)

    usage_match = re.search(r"KULLANIM\s+AMACI\s+(.+?)\s+K/?\s+TIP\s+ONAY", clean)
    if usage_match:
        out["usage_purpose"] = collapse_spaces(usage_match.group(1))

    approval_match = re.search(r"TIP\s+ONAY\s+NO\s+([A-Z0-9*./-]{8,})", clean)
    if approval_match:
        out["approval_type_no"] = approval_match.group(1)

    serial_match = re.search(r"SERI[: ]*([A-Z]{1,4})\s+N[O0º]*\s*(\d{4,8})", clean)
    if serial_match:
        out["document_serial"] = serial_match.group(1)
        out["document_number"] = serial_match.group(2)
        out["serial_no"] = f"{serial_match.group(1)} NO {serial_match.group(2)}"

    for idx, line in enumerate(lines):
        if "VERILDIGI IL/ILCE" in line and "province_district" not in out:
            for candidate in lines[idx + 1: idx + 3]:
                if "/" in candidate and "PLAKA" not in candidate:
                    out["province_district"] = candidate
                    break

        if ("ILK TESCIL" in line or "ILK TESCIL TAR" in line) and "first_registration_date" not in out:
            date_match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", line)
            if not date_match and idx + 1 < len(lines):
                date_match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", lines[idx + 1])
            if date_match:
                out["first_registration_date"] = date_match.group(0).replace("-", "/")

        if "TESCIL TARIH" in line and "ILK" not in line and "registration_date" not in out:
            date_match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", line)
            if not date_match and idx + 1 < len(lines):
                date_match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", lines[idx + 1])
            if date_match:
                out["registration_date"] = date_match.group(0).replace("-", "/")

        if "MARKASI" in line and "brand" not in out:
            candidate = line.split("MARKASI", 1)[-1].strip()
            if not candidate and idx + 1 < len(lines):
                candidate = lines[idx + 1]
            candidate = collapse_spaces(candidate)
            if re.fullmatch(r"[A-Z0-9-]{2,40}", candidate):
                out["brand"] = candidate

        if (
            re.search(r"C[.,]?\s*1[.,]?\s*1", line)
            or "SOYADI/TICARI" in line
            or "SOYADI TICARI" in line
        ) and "owner_surname" not in out:
            candidate = re.split(r"SOYADI/?\s*TICARI\s+UNVAN[A-Z]*", line, maxsplit=1)[-1].strip()
            if not candidate and idx + 1 < len(lines):
                candidate = lines[idx + 1]
            if candidate.startswith("(") or " ADI" in candidate or candidate == out.get("brand", ""):
                candidate = ""
            owner_title = _clean_owner_title(candidate or line)
            if owner_title:
                out["owner_title"] = owner_title
                out["owner_surname"] = owner_title

        if (
            re.search(r"C[.,]?\s*1[.,]?\s*2", line)
            or re.search(r"^\(?C[.,]?\s*1[.,]?\s*2\)?\s*ADI", line)
        ) and "owner_name" not in out:
            candidate = re.split(r"\bADI\b", line, maxsplit=1)[-1].strip()
            if not candidate and idx + 1 < len(lines):
                candidate = lines[idx + 1]
            if (
                candidate.startswith("(")
                or candidate == out.get("brand", "")
                or any(tag in candidate for tag in ("MARKASI", "PLAKA", "TESCIL", "MODEL", "TIPI", "TIPII"))
            ):
                candidate = ""
            owner_name = _clean_owner_name(candidate)
            if owner_name:
                out["owner_name"] = owner_name
                if "owner_address" not in out:
                    address_lines: list[str] = []
                    for follow in lines[idx + 1: idx + 4]:
                        if any(tag in follow for tag in ("BELGE", "NOTER", "MUA", "TESCIL", "PLAKA")):
                            break
                        if _looks_like_address(follow):
                            address_lines.append(follow)
                    if address_lines:
                        out["owner_address"] = collapse_spaces(" ".join(address_lines))

    return out


def apply_right_page_vl_route(
    fields: dict[str, dict[str, object]],
    parsed: dict[str, str],
    force: bool = False,
) -> list[str]:
    changed: list[str] = []
    if not parsed:
        return changed

    def current_value(name: str) -> str:
        entry = fields.get(name)
        if not isinstance(entry, dict):
            return ""
        return _norm(str(entry.get("value") or ""))

    def patch(name: str, value: str) -> None:
        current = fields.get(name)
        base = dict(current) if isinstance(current, dict) else {}
        base["value"] = value
        base["raw"] = value
        base["method"] = "right_page_vl_route"
        base["confidence_score"] = max(int(base.get("confidence_score", 0) or 0), 80)
        base["low_confidence"] = False
        base["right_page_vl_applied"] = True
        fields[name] = base
        changed.append(name)

    parsed_owner_title = parsed.get("owner_title")
    if parsed_owner_title and _looks_like_owner_title(parsed_owner_title):
        current_owner_title = current_value("owner_title")
        current_owner_surname = current_value("owner_surname")
        if force or not current_owner_title or _looks_like_owner_noise(current_owner_title) or not _has_company_hint(current_owner_title):
            patch("owner_title", parsed_owner_title)
        if force or not current_owner_surname or _looks_like_owner_noise(current_owner_surname) or _looks_like_person_name(current_owner_surname):
            patch("owner_surname", parsed_owner_title)

    parsed_owner_name = parsed.get("owner_name")
    if parsed_owner_name and _looks_like_person_name(parsed_owner_name):
        current_owner_name = current_value("owner_name")
        current_owner_surname = current_value("owner_surname")
        if force or not current_owner_name or current_owner_name == current_owner_surname or _has_company_hint(current_owner_name) or _looks_like_owner_noise(current_owner_name):
            patch("owner_name", parsed_owner_name)

    parsed_owner_address = parsed.get("owner_address")
    if parsed_owner_address and _looks_like_address(parsed_owner_address):
        current_owner_address = current_value("owner_address")
        if force or not current_owner_address or not _looks_like_address(current_owner_address) or "MENFAATI" in current_owner_address:
            patch("owner_address", parsed_owner_address)

    parsed_tax = parsed.get("tax_or_id_no")
    if parsed_tax and re.fullmatch(r"\d{10,11}", _norm(parsed_tax)):
        current_tax = current_value("tax_or_id_no")
        if force or not current_tax:
            patch("tax_or_id_no", parsed_tax)

    for date_field in ("inspection_date",):
        parsed_date = parsed.get(date_field)
        if parsed_date and _looks_like_date(parsed_date):
            current_date = current_value(date_field)
            if force or not current_date or not _looks_like_date(current_date):
                patch(date_field, parsed_date)

    parsed_doc_serial = parsed.get("document_serial")
    if parsed_doc_serial and _looks_like_doc_serial(parsed_doc_serial) and (force or not current_value("document_serial")):
        patch("document_serial", parsed_doc_serial)

    parsed_doc_number = parsed.get("document_number")
    if parsed_doc_number and _looks_like_doc_number(parsed_doc_number) and (force or not current_value("document_number")):
        patch("document_number", parsed_doc_number)

    parsed_serial = parsed.get("serial_no")
    if parsed_serial and (force or not current_value("serial_no")):
        patch("serial_no", parsed_serial)

    return changed


def apply_d_block_vl_route(
    fields: dict[str, dict[str, object]],
    parsed: dict[str, str],
    force: bool = False,
) -> list[str]:
    changed: list[str] = []
    if not parsed:
        return changed

    def current_value(name: str) -> str:
        entry = fields.get(name)
        if not isinstance(entry, dict):
            return ""
        return _norm(str(entry.get("value") or ""))

    def patch(name: str, value: str) -> None:
        current = fields.get(name)
        base = dict(current) if isinstance(current, dict) else {}
        base["value"] = value
        base["raw"] = value
        base["method"] = "d_block_vl_route"
        base["confidence_score"] = max(int(base.get("confidence_score", 0) or 0), 80)
        base["low_confidence"] = False
        base["d_block_vl_applied"] = True
        fields[name] = base
        changed.append(name)

    parsed_brand = parsed.get("brand")
    if parsed_brand and _looks_like_brand(parsed_brand):
        current_brand = current_value("brand")
        if force or not current_brand or _looks_like_owner_noise(current_brand):
            patch("brand", parsed_brand)

    parsed_type = parsed.get("type")
    if parsed_type and _looks_like_vehicle_type(parsed_type):
        current_type = current_value("type")
        if force or not current_type or len(re.sub(r"[^A-Z0-9]", "", current_type)) < 3:
            patch("type", parsed_type)

    parsed_vehicle_type = parsed.get("vehicle_type")
    if parsed_vehicle_type and _looks_like_vehicle_type(parsed_vehicle_type):
        current_vehicle_type = current_value("vehicle_type")
        if force or not current_vehicle_type or len(re.sub(r"[^A-Z0-9]", "", current_vehicle_type)) < 3:
            patch("vehicle_type", parsed_vehicle_type)

    parsed_model_year = parsed.get("model_year")
    if parsed_model_year and re.fullmatch(r"(19\d{2}|20\d{2})", _norm(parsed_model_year)):
        current_model_year = current_value("model_year")
        if force or not current_model_year:
            patch("model_year", parsed_model_year)

    parsed_vehicle_class = parsed.get("vehicle_class")
    if parsed_vehicle_class and re.fullmatch(r"[LMNO]\d", _norm(parsed_vehicle_class)):
        current_vehicle_class = current_value("vehicle_class")
        if force or not current_vehicle_class:
            patch("vehicle_class", parsed_vehicle_class)

    parsed_color = parsed.get("color")
    if parsed_color and re.fullmatch(r"(BEYAZ|SIYAH|GRI|GUMUS|MAVI|KIRMIZI|YESIL|SARI)", _norm(parsed_color)):
        current_color = current_value("color")
        if force or not current_color:
            patch("color", parsed_color)

    return changed


def should_run_full_page_vl_second_pass(fields: dict[str, dict[str, object]], selected_profile: str) -> bool:
    if selected_profile not in {"unknown_or_degraded_photo", "v29_photo_variant_b_candidate"}:
        return False

    def text_of(name: str) -> str:
        entry = fields.get(name)
        if not isinstance(entry, dict):
            return ""
        return _norm(str(entry.get("value") or ""))

    owner_name = text_of("owner_name")
    owner_surname = text_of("owner_surname")
    owner_address = text_of("owner_address")
    province = text_of("province_district")
    vehicle_type = text_of("vehicle_type")
    type_text = text_of("type")
    serial_no = text_of("serial_no")

    owner_issue = (
        not owner_name
        or owner_name == owner_surname
        or _has_company_hint(owner_name)
        or _looks_like_owner_noise(owner_name)
        or not owner_surname
        or _looks_like_owner_noise(owner_surname)
    )
    location_issue = (not owner_address or not _looks_like_address(owner_address)) and (
        not province or not _looks_like_province(province)
    )
    type_issue = not type_text or not _looks_like_vehicle_type(type_text)
    vehicle_type_issue = (
        not vehicle_type
        or vehicle_type in _GENERIC_VEHICLE_TYPE_HINTS
        or not _looks_like_vehicle_type(vehicle_type)
    )
    serial_issue = not serial_no

    issue_score = 0.0
    if owner_issue:
        issue_score += 2.0
    if type_issue:
        issue_score += 1.5
    if vehicle_type_issue:
        issue_score += 1.0
    if serial_issue:
        issue_score += 1.5
    if location_issue:
        issue_score += 0.75

    # Degraded photos still benefit from VL more often, but location-only misses
    # should not be enough to light up an expensive full-page pass by themselves.
    if selected_profile == "unknown_or_degraded_photo" and location_issue:
        issue_score += 0.25

    return issue_score >= 2.5


def score_full_page_vl_candidate(parsed: dict[str, str]) -> int:
    if not parsed:
        return 0

    score = 0
    if _looks_like_person_name(parsed.get("owner_name", "")):
        score += 4
    if _looks_like_owner_title(parsed.get("owner_title", "")):
        score += 4
    if _looks_like_address(parsed.get("owner_address", "")):
        score += 3
    if _looks_like_province(parsed.get("province_district", "")):
        score += 3
    if _looks_like_brand(parsed.get("brand", "")):
        score += 2
    if _looks_like_vehicle_type(parsed.get("type", "")):
        score += 3
    if _looks_like_vehicle_type(parsed.get("vehicle_type", "")):
        score += 3
    if _looks_like_registration_serial(parsed.get("registration_serial_no", "")):
        score += 2
    if _looks_like_doc_serial(parsed.get("document_serial", "")):
        score += 1
    if _looks_like_doc_number(parsed.get("document_number", "")):
        score += 1
    if parsed.get("serial_no"):
        score += 1
    for date_field in ("first_registration_date", "registration_date", "inspection_date"):
        if _looks_like_date(parsed.get(date_field, "")):
            score += 1
    return score


def merge_full_page_vl_fields(fields: dict[str, dict[str, object]], parsed: dict[str, str]) -> list[str]:
    changed: list[str] = []
    if not parsed:
        return changed

    def current_value(name: str) -> str:
        entry = fields.get(name)
        if not isinstance(entry, dict):
            return ""
        return _norm(str(entry.get("value") or ""))

    def patch(name: str, value: str, method: str = "full_page_vl_parser") -> None:
        current = fields.get(name)
        base = dict(current) if isinstance(current, dict) else {}
        base["value"] = value
        base["raw"] = value
        base["method"] = method
        base["confidence_score"] = max(int(base.get("confidence_score", 0) or 0), 70)
        base["low_confidence"] = False
        base["full_page_vl_applied"] = True
        fields[name] = base
        changed.append(name)

    owner_name = current_value("owner_name")
    owner_surname = current_value("owner_surname")
    owner_address = current_value("owner_address")
    province = current_value("province_district")
    type_text = current_value("type")
    vehicle_type = current_value("vehicle_type")
    serial_no = current_value("serial_no")

    parsed_owner_name = _norm(parsed.get("owner_name", ""))
    parsed_owner_title = _norm(parsed.get("owner_title", ""))
    parsed_owner_title_last = parsed_owner_title.split()[-1] if parsed_owner_title else ""
    if parsed_owner_name and (
        not owner_name
        or owner_name == owner_surname
        or _has_company_hint(owner_name)
        or _looks_like_owner_noise(owner_name)
        or (parsed_owner_title_last and owner_name == parsed_owner_title_last)
    ):
        patch("owner_name", parsed["owner_name"])

    if parsed_owner_title:
        if (
            not owner_surname
            or _looks_like_owner_noise(owner_surname)
            or (not _has_company_hint(owner_surname) and len(owner_surname) < len(parsed_owner_title))
        ):
            patch("owner_surname", parsed["owner_title"])
        current_owner_title = current_value("owner_title")
        if not current_owner_title or _looks_like_owner_noise(current_owner_title) or not _has_company_hint(current_owner_title):
            patch("owner_title", parsed["owner_title"])

    parsed_address = _norm(parsed.get("owner_address", ""))
    if parsed_address and (not owner_address or not _looks_like_address(owner_address) or "MENFAATI" in owner_address):
        patch("owner_address", parsed["owner_address"])

    parsed_province = _norm(parsed.get("province_district", ""))
    if parsed_province and (not province or any(token in province for token in ("PLAKA", "TESCIL", "TARIH"))):
        patch("province_district", parsed["province_district"])

    parsed_type = _norm(parsed.get("type", ""))
    if parsed_type and (not type_text or len(re.sub(r"[^A-Z0-9]", "", type_text)) < 3):
        patch("type", parsed["type"])

    parsed_vehicle_type = _norm(parsed.get("vehicle_type", ""))
    if parsed_vehicle_type and (
        not vehicle_type
        or len(re.sub(r"[^A-Z0-9]", "", vehicle_type)) < 2
        or vehicle_type in _GENERIC_VEHICLE_TYPE_HINTS
    ):
        patch("vehicle_type", parsed["vehicle_type"])

    parsed_serial = _norm(parsed.get("serial_no", ""))
    if parsed_serial and not serial_no:
        patch("serial_no", parsed["serial_no"])
    if parsed.get("document_serial") and not current_value("document_serial"):
        patch("document_serial", parsed["document_serial"])
    if parsed.get("document_number") and not current_value("document_number"):
        patch("document_number", parsed["document_number"])

    if parsed.get("inspection_date") and not current_value("inspection_date"):
        patch("inspection_date", parsed["inspection_date"])
    if parsed.get("registration_serial_no") and not current_value("registration_serial_no"):
        patch("registration_serial_no", parsed["registration_serial_no"])

    return changed


def apply_full_page_vl_route(
    fields: dict[str, dict[str, object]],
    parsed: dict[str, str],
    selected_profile: str,
    force: bool = False,
) -> list[str]:
    changed: list[str] = []
    if not parsed:
        return changed
    if not force and selected_profile not in {"v29_photo_variant_b_candidate", "unknown_or_degraded_photo"}:
        return changed

    def current_value(name: str) -> str:
        entry = fields.get(name)
        if not isinstance(entry, dict):
            return ""
        return _norm(str(entry.get("value") or ""))

    def patch(name: str, value: str) -> None:
        current = fields.get(name)
        base = dict(current) if isinstance(current, dict) else {}
        base["value"] = value
        base["raw"] = value
        base["method"] = "full_page_vl_route"
        base["confidence_score"] = max(int(base.get("confidence_score", 0) or 0), 85)
        base["low_confidence"] = False
        base["full_page_vl_applied"] = True
        fields[name] = base
        changed.append(name)

    current_owner_name = current_value("owner_name")
    current_owner_surname = current_value("owner_surname")
    current_owner_title = current_value("owner_title")
    current_owner_address = current_value("owner_address")

    parsed_owner_name = parsed.get("owner_name")
    current_owner_title_last = current_owner_title.split()[-1] if current_owner_title else ""
    if parsed_owner_name and _looks_like_person_name(parsed_owner_name):
        if (
            force
            or selected_profile in {"v29_photo_variant_b_candidate", "unknown_or_degraded_photo"}
            or not current_owner_name
            or current_owner_name == current_owner_surname
            or _has_company_hint(current_owner_name)
            or _looks_like_owner_noise(current_owner_name)
            or (current_owner_title_last and current_owner_name == current_owner_title_last)
        ):
            patch("owner_name", parsed_owner_name)

    parsed_owner_title = parsed.get("owner_title")
    if parsed_owner_title and _looks_like_owner_title(parsed_owner_title):
        if (
            force
            or selected_profile in {"v29_photo_variant_b_candidate", "unknown_or_degraded_photo"}
            or not current_owner_title
            or _looks_like_owner_noise(current_owner_title)
            or not _has_company_hint(current_owner_title)
        ):
            patch("owner_title", parsed_owner_title)
        if (
            force
            or selected_profile in {"v29_photo_variant_b_candidate", "unknown_or_degraded_photo"}
            or not current_owner_surname
            or _looks_like_owner_noise(current_owner_surname)
            or _looks_like_person_name(current_owner_surname)
            or not _has_company_hint(current_owner_surname)
        ):
            patch("owner_surname", parsed_owner_title)

    parsed_owner_address = parsed.get("owner_address")
    if parsed_owner_address and _looks_like_address(parsed_owner_address):
        if (
            force
            or selected_profile in {"v29_photo_variant_b_candidate", "unknown_or_degraded_photo"}
            or not current_owner_address
            or not _looks_like_address(current_owner_address)
            or "MENFAATI" in current_owner_address
        ):
            patch("owner_address", parsed_owner_address)

    parsed_province = parsed.get("province_district")
    if parsed_province and _looks_like_province(parsed_province):
        current_province = current_value("province_district")
        if (
            force
            or selected_profile in {"v29_photo_variant_b_candidate", "unknown_or_degraded_photo"}
            or not current_province
            or any(token in current_province for token in ("PLAKA", "TESCIL", "TARIH"))
        ):
            patch("province_district", parsed_province)

    parsed_brand = parsed.get("brand")
    if parsed_brand and _looks_like_brand(parsed_brand):
        current_brand = current_value("brand")
        if force or not current_brand or _looks_like_owner_noise(current_brand):
            patch("brand", parsed_brand)

    for date_field in ("first_registration_date", "registration_date", "inspection_date"):
        parsed_date = parsed.get(date_field)
        if parsed_date and _looks_like_date(parsed_date):
            current_date = current_value(date_field)
            if force or not current_date or not _looks_like_date(current_date):
                patch(date_field, parsed_date)

    parsed_type = parsed.get("type")
    if parsed_type and _looks_like_vehicle_type(parsed_type):
        current_type = current_value("type")
        if force or not current_type or len(re.sub(r"[^A-Z0-9]", "", current_type)) < 3:
            patch("type", parsed_type)

    parsed_vehicle_type = parsed.get("vehicle_type")
    if parsed_vehicle_type and _looks_like_vehicle_type(parsed_vehicle_type):
        current_vehicle_type = current_value("vehicle_type")
        if force or not current_vehicle_type or len(re.sub(r"[^A-Z0-9]", "", current_vehicle_type)) < 3:
            patch("vehicle_type", parsed_vehicle_type)

    parsed_reg_serial = parsed.get("registration_serial_no")
    if parsed_reg_serial and _looks_like_registration_serial(parsed_reg_serial):
        current_reg_serial = current_value("registration_serial_no")
        if force or not current_reg_serial or len(re.sub(r"[^A-Z0-9]", "", current_reg_serial)) < 10:
            patch("registration_serial_no", parsed_reg_serial)

    parsed_doc_serial = parsed.get("document_serial")
    if parsed_doc_serial and _looks_like_doc_serial(parsed_doc_serial) and (force or not current_value("document_serial")):
        patch("document_serial", parsed_doc_serial)

    parsed_doc_number = parsed.get("document_number")
    if parsed_doc_number and _looks_like_doc_number(parsed_doc_number) and (force or not current_value("document_number")):
        patch("document_number", parsed_doc_number)

    parsed_serial = parsed.get("serial_no")
    if parsed_serial and (force or not current_value("serial_no")):
        patch("serial_no", parsed_serial)

    return changed
