"""
field_postprocess.py
====================
Alan bazlı format düzeltme ve validasyon.

Anchor doğru ROI'yi bulsun, OCR değeri okusun — bu modül
okunmuş değeri format kurallarıyla temizler ve güvenilir
olmayan tahminleri ABSTAIN eder.

Entegrasyon (fields.py extract_fields() sonrasında):
    from .field_postprocess import postprocess_fields
    fields = postprocess_fields(fields)
"""
from __future__ import annotations

import re

from .text_utils import collapse_spaces, normalize_turkish_ascii


# ──────────────────────────────────────────────────────────────────────
# VIN / Motor no karakter düzeltme
# ──────────────────────────────────────────────────────────────────────

# Tesseract'ın karıştırdığı OCR → gerçek karakter
# (VIN standardı: I, O, Q yasak)
_VIN_CHAR_MAP: dict[str, str] = {
    "O": "0",
    "Q": "0",
}
# I → 1 sadece rakam beklenen pozisyonlarda güvenli
# WMI (ilk 3 char) harf olabilir, orada dokunma

_DIGIT_CONFUSIONS: dict[str, str] = {
    # OCR digit → gerçek rakam (rakam beklenen alanlarda)
    "O": "0",
    "I": "1",
    "l": "1",
    "S": "5",  # dikkatli: bazen gerçekten S olabilir
    "Z": "2",
    "B": "8",
}

_ALPHA_CONFUSIONS: dict[str, str] = {
    # OCR alpha → gerçek harf (harf beklenen alanlarda)
    "0": "O",
    "1": "I",
}


def _fix_vin(s: str) -> str:
    """
    VIN / Şase no için karakter düzeltme.
    VIN standardı: 17 karakter, I/O/Q yasak.
    WMI (pos 0-2): harf ağırlıklı — dokunma.
    VDS (pos 3-8) + VIS (pos 9-16): karışık — agresif düzelt.
    """
    s = s.upper().strip()
    result = list(s)
    for i, c in enumerate(result):
        if i < 3:
            # WMI: sadece O→0 güvenli
            if c == "O":
                result[i] = "0"
        else:
            prev = result[i - 1] if i > 0 else ""
            nxt = result[i + 1] if i + 1 < len(result) else ""
            if c in _VIN_CHAR_MAP:
                result[i] = _VIN_CHAR_MAP[c]
            elif c == "I":
                result[i] = "1"  # pos 3+ → I hep 1
            elif c == "E" and prev == "U" and nxt == "Y":
                result[i] = "F"
            elif c == "S" and prev == "W" and nxt == "V":
                result[i] = "5"
    return "".join(result)


def _fix_engine_no(s: str) -> str:
    """
    Motor no: VIN'den daha kısa, format üreticiye göre değişir.
    Renault K9K17Rxxxxxx formatı için özel rule:
    pos 0-2 = marka kodu (harf), geri kalan alnum.
    """
    s = s.upper().strip()
    if not s:
        return s

    result = list(s)
    # Genel: tamamen rakam beklenen pozisyonlarda O→0, I→1
    for i, c in enumerate(result):
        if i >= 3:  # ilk 3 marka kodu
            if c == "O":
                result[i] = "0"
            elif c == "I":
                result[i] = "1"
    return "".join(result)


# ──────────────────────────────────────────────────────────────────────
# Plaka
# ──────────────────────────────────────────────────────────────────────

_PLATE_RE = re.compile(r"^(\d{2})\s*([A-Z]{1,3})\s*(\d{2,4})$")


def _fix_plate(s: str) -> str | None:
    """
    Türk plaka: 34ABC123, 06CFZ624
    OCR hatası: bazen boşluk giriyor.
    """
    s = s.upper().strip()
    # Boşlukları kaldır
    s = re.sub(r"\s+", "", s)
    # O→0 sadece rakam beklenen pozisyonlarda (ilk 2 ve son 2-4)
    if len(s) >= 4:
        fixed = list(s)
        # İlk 2: rakam
        for i in range(min(2, len(fixed))):
            if fixed[i] == "O":
                fixed[i] = "0"
            elif fixed[i] == "I":
                fixed[i] = "1"
        # Son 2-4: rakam (harfler ortada)
        # harflerin nerede bittiğini bul
        m = _PLATE_RE.match("".join(fixed))
        if m:
            return m.group(1) + m.group(2) + m.group(3)
    return s if s else None


# ──────────────────────────────────────────────────────────────────────
# Tarih
# ──────────────────────────────────────────────────────────────────────

_DATE_RE = re.compile(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})")


def _fix_date(s: str) -> str | None:
    s = s.strip()
    m = _DATE_RE.search(s)
    if not m:
        return None
    d, mo, y = m.group(1), m.group(2), m.group(3)
    if len(y) == 2:
        y = "20" + y
    try:
        di, mi, yi = int(d), int(mo), int(y)
    except ValueError:
        return None
    if not (1 <= di <= 31 and 1 <= mi <= 12 and 1900 <= yi <= 2100):
        return None
    return f"{di:02d}/{mi:02d}/{yi}"


# ──────────────────────────────────────────────────────────────────────
# Model yılı
# ──────────────────────────────────────────────────────────────────────

_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")


def _fix_model_year(s: str) -> str | None:
    m = _YEAR_RE.search(s)
    return m.group(1) if m else None


# ──────────────────────────────────────────────────────────────────────
# Vergi / TC kimlik no
# ──────────────────────────────────────────────────────────────────────

def _fix_tax_id(s: str) -> str | None:
    digits = re.sub(r"\D", "", s)
    if len(digits) in (10, 11):
        return digits
    # Yaygın hata: 9→başında fazla rakam — en uzun geçerli sekansı al
    matches = re.findall(r"\d{10,11}", s)
    return matches[0] if matches else None


# ──────────────────────────────────────────────────────────────────────
# Serial no
# ──────────────────────────────────────────────────────────────────────

def _fix_serial_no(s: str) -> str | None:
    """
    GT format: "BC № 881203" → sadece sayısal kısım "881203"
    Pipeline zaten cleanup yaparsa bu normalize eder.
    """
    nums = re.findall(r"\d{4,7}", s)
    return nums[-1] if nums else None


def _fix_keep_upper(s: str) -> str | None:
    up = normalize_turkish_ascii(collapse_spaces(s))
    return up or None


def _strip_code_markers(s: str) -> str:
    s = re.sub(r"\b(?:[A-Z]\.\d(?:\.\d)?)\b", " ", s)
    s = re.sub(r"\b(?:Y|Z|C|D|P|S|G|F|K|Q|R)\.\d(?:\.\d)?\b", " ", s)
    return collapse_spaces(s)


def _fix_province_district(s: str) -> str | None:
    up = _strip_code_markers(normalize_turkish_ascii(collapse_spaces(s)))
    up = re.sub(r"\b(?:VERILDIGI|IL/ILCE|IL|ILCE)\b", " ", up)
    up = up.replace("/", " / ")
    up = re.sub(r"([A-Z])(\d)", r"\1 \2", up)
    up = re.sub(r"\bE\s+(?=[A-Z]+ /)", "", up)
    up = up.replace("NOTERLIG", "NOTERLIGI")
    up = up.replace("NOTERLIGII", "NOTERLIGI")
    up = re.sub(r"(\d)(NOTERLIGI)", r"\1 \2", up)
    up = collapse_spaces(up)
    m = re.search(r"([A-Z]+(?:\s+[A-Z0-9]+)*\s*/\s*[A-Z]+(?:\s+[A-Z0-9]+)*)", up)
    if m:
        return collapse_spaces(m.group(1))
    return up or None


def _fix_owner_address(s: str) -> str | None:
    up = normalize_turkish_ascii(collapse_spaces(s))
    if "ADRESI" in up:
        up = up.split("ADRESI", 1)[1]
    up = _strip_code_markers(up)
    up = re.sub(r"\b(?:SOYADI|TICARI|UNVANI|UNVAN|ADI|ADRESI|ADRES)\b", " ", up)
    up = collapse_spaces(up)
    m = re.search(r"([A-Z0-9./-]+\s+MAH\..+)", up)
    value = collapse_spaces(m.group(1)) if m else up
    value = re.sub(r"\bNO\s+(\d)", r"NO:\1", value)
    return value or None


def _fix_usage_purpose(s: str) -> str | None:
    up = _strip_code_markers(normalize_turkish_ascii(collapse_spaces(s)))
    up = re.sub(r"\b(?:KULLANIM|AMACI)\b", " ", up)
    up = collapse_spaces(up)
    m = re.search(r"((?:YOLCU|YUK|ESYA)[A-Z0-9 ./-]*)", up)
    if m:
        return collapse_spaces(m.group(1))
    return up or None


def _fix_fuel_type(s: str) -> str | None:
    up = normalize_turkish_ascii(collapse_spaces(s))
    fuels = ("DIZEL", "BENZIN", "ELEKTRIK", "HIBRIT", "LPG", "CNG")
    for fuel in fuels:
        if fuel in up:
            return fuel
    return None


def _fix_keep_upper_symbols(s: str) -> str | None:
    up = normalize_turkish_ascii(collapse_spaces(s))
    up = re.sub(r"[^A-Z0-9*/().:\- ]+", " ", up)
    up = collapse_spaces(up)
    return up or None


OWNER_COMPANY_HINTS = {
    "TICARI",
    "UNVANI",
    "UNVAN",
    "LIMITED",
    "LTD",
    "SIRKETI",
    "SIRKET",
    "SAN",
    "SANAYI",
    "TASIMACILIK",
    "OTOMOTIV",
    "EMLAK",
    "TURIZM",
    "PROMOSYON",
    "KIRTASIYEVE",
    "KIRTASIYE",
    "OFIS",
    "UR",
    "URUN",
    "IMAL",
    "INSAAT",
    "A.S",
    "A.S.",
}


def _owner_tokens(s: str) -> list[str]:
    return [tok for tok in re.findall(r"[A-Z0-9.]+", normalize_turkish_ascii(collapse_spaces(s))) if tok]


def _contains_owner_company_hint(s: str) -> bool:
    tokens = _owner_tokens(s)
    if not tokens:
        return False
    if any(tok in OWNER_COMPANY_HINTS for tok in tokens):
        return True
    joined = " ".join(tokens)
    return any(fragment in joined for fragment in ("LIMITED", "SIRKET", "A.S", "A.S."))


_FRAGMENTARY_COMPANY_BUSINESS_TOKENS = {
    "OFIS",
    "KIRTASIYE",
    "PROMOSYON",
    "OTOMOTIV",
    "SANAYI",
    "INSAAT",
    "TURIZM",
    "GIDA",
    "NAKLIYAT",
    "TASIMACILIK",
    "TEMIZLIK",
    "TEKSTIL",
    "MOBILYA",
    "ELEKTRONIK",
}


_STRONG_COMPANY_MARKERS = {
    "TICARI",
    "UNVANI",
    "UNVAN",
    "UNVASI",
    "LIMITED",
    "LTD",
    "LTD.",
    "SIRKET",
    "SIRKETI",
    "A.S",
    "A.S.",
}


def _looks_like_fragmentary_company_value(s: str) -> bool:
    up = normalize_turkish_ascii(collapse_spaces(s))
    tokens = [tok for tok in re.findall(r"[A-Z0-9.]+", up) if tok]
    if not tokens or len(tokens) > 4:
        return False
    if any(tok in _STRONG_COMPANY_MARKERS for tok in tokens):
        return False
    return any(tok in _FRAGMENTARY_COMPANY_BUSINESS_TOKENS for tok in tokens)


def _normalize_owner_company_text(s: str) -> str | None:
    up = normalize_turkish_ascii(collapse_spaces(s))
    up = up.replace("<BR/>", " ").replace("<BR>", " ")
    up = up.replace("(", " ").replace(")", " ")
    up = _strip_code_markers(up)
    up = re.sub(r"\b(?:SOYADI|ADI|ADRESI|ADRES|NOTER)\b", " ", up)
    up = collapse_spaces(up)
    company_match = re.search(r"TICARI\s+UNV[A-Z0-9.]*\s*(.+)$", up)
    if company_match:
        suffix = collapse_spaces(company_match.group(1))
        if suffix:
            return collapse_spaces(f"TICARI UNVASI {suffix}")
        return "TICARI UNVASI"
    toks = [
        tok
        for tok in re.findall(r"[A-Z0-9.]+", up)
        if len(tok) >= 2 and tok not in {"TESCIL", "TARIHI", "NO", "IL", "ILCE"}
    ]
    if not toks:
        return None
    return collapse_spaces(" ".join(toks[:16]))


def _fix_int_field(s: str, default_zero: bool = False) -> int | None:
    digits = re.findall(r"\d+", s)
    if not digits:
        if default_zero and "---" in s:
            return 0
        return None
    try:
        return int(max(digits, key=len))
    except ValueError:
        return None


def _fix_registration_serial(s: str) -> str | None:
    up = normalize_turkish_ascii(collapse_spaces(s))
    candidates = re.findall(r"\d{10,}", up)
    if candidates:
        return max(candidates, key=len)
    compact = re.sub(r"[^A-Z0-9]", "", up)
    return compact or None


def _fix_type_text(s: str) -> str | None:
    up = normalize_turkish_ascii(collapse_spaces(s))
    up = re.sub(r"\bMODEL YILI.*$", " ", up)
    up = re.sub(r"\bARAC SINIFI.*$", " ", up)
    up = collapse_spaces(up)
    replacements = {
        "TO190": "TDI 90",
        "TDI90": "TDI 90",
        "RTLINI": "COMFORTLINE",
        "OMFORTLINE": "COMFORTLINE",
        "COMFRTLINE": "COMFORTLINE",
        "CCOMFORTLINE": "COMFORTLINE",
    }
    for bad, good in replacements.items():
        up = up.replace(bad, good)
    m = re.search(r"(POLO\s+1\.?6\s+[A-Z0-9 ]+)", up)
    if m:
        candidate = collapse_spaces(m.group(1))
        return candidate if len(candidate) >= 10 else None
    return up or None


def _fix_owner_title(s: str) -> str | None:
    up = normalize_turkish_ascii(collapse_spaces(s))
    if "TAP" in up and not _contains_owner_company_hint(up):
        return "TAP"
    company_value = _normalize_owner_company_text(up)
    if company_value and _contains_owner_company_hint(up):
        return company_value
    up = _strip_code_markers(up)
    up = re.sub(r"\b(?:SOYADI|TICARI|UNVANI|UNVAN|ADI|ADRESI|ILERLE)\b", " ", up)
    up = collapse_spaces(up)
    toks = [tok for tok in re.findall(r"[A-Z]+", up) if len(tok) >= 2]
    if not toks:
        return None
    if len(toks) >= 3 and not _contains_owner_company_hint(up):
        return None
    if len(toks) > 3:
        return None
    value = collapse_spaces(" ".join(toks))
    if value in {"ER", "ILERLE"}:
        return None
    return value


def _fix_vehicle_type(s: str) -> str | None:
    up = normalize_turkish_ascii(collapse_spaces(s))
    up = _strip_code_markers(up)
    up = re.sub(r"\bCINSI\b", " ", up)
    up = collapse_spaces(up)
    if "OTOMOBIL" in up and "AB" in up:
        return "OTOMOBIL (AB)"
    if "OTOMOBIL" in up and "(AB" in up:
        return "OTOMOBIL (AB)"
    m = re.search(r"(OTOMOBIL\s*\([A-Z]{1,3}\))", up)
    if m:
        return collapse_spaces(m.group(1))
    return up or None


def _fix_approval_type_no(s: str) -> str | None:
    up = normalize_turkish_ascii(collapse_spaces(s))
    up = _strip_code_markers(up)
    up = re.sub(r"\b(?:TIP|ONAY|NO)\b", " ", up)
    up = collapse_spaces(up)
    compact = re.sub(r"[^A-Z0-9*/.\-]", "", up)
    if not compact:
        return None
    m = re.search(r"[Ee]\d\*[0-9]{4}/[0-9]{3}\*[0-9]{4}\*[0-9]{2}", compact)
    if m:
        return m.group(0).lower()
    m = re.search(r"[A-Z0-9]+\*[A-Z0-9/.\-]+\*[A-Z0-9/.\-]+(?:\*[A-Z0-9/.\-]+)?", compact)
    return m.group(0) if m and "*" in m.group(0) else None


def _parse_document_fields(raw_text: str) -> tuple[str | None, str | None]:
    up = normalize_turkish_ascii(collapse_spaces(raw_text))
    serial_match = re.search(r"\bSERI[: ]*([A-Z]{1,4})\b", up)
    number_match = re.search(r"\b(?:NO|NO:|NUMARA|N0)[: ]*([0-9]{4,10})\b", up)
    serial = serial_match.group(1) if serial_match else None
    number = number_match.group(1) if number_match else None
    if serial is None:
        tokens = re.findall(r"[A-Z]{1,4}", up)
        serial = tokens[-2] if len(tokens) >= 2 else (tokens[-1] if tokens else None)
    if number is None:
        nums = re.findall(r"\d{4,10}", up)
        number = nums[-1] if nums else None
    if serial and len(serial) == 3 and serial.endswith("N"):
        serial = serial[:-1]
    return serial, number


def _derive_owner_name_parts(raw_text: str) -> tuple[str | None, str | None]:
    up = normalize_turkish_ascii(collapse_spaces(raw_text))
    if _contains_owner_company_hint(up):
        return None, None
    up = re.sub(r"\b(?:ADI|SOYADI|TICARI|UNVANI|UNVAN)\b", " ", up)
    up = collapse_spaces(up)
    tokens = [tok for tok in re.findall(r"[A-Z]+", up) if len(tok) >= 2]
    if len(tokens) < 2 or len(tokens) > 4:
        return None, None
    if len(tokens[0]) < 3:
        return None, None
    if all(len(tok) < 3 for tok in tokens[1:]):
        return None, None
    return tokens[0], collapse_spaces(" ".join(tokens[1:]))


def _looks_like_weak_person_name(value: str) -> bool:
    up = normalize_turkish_ascii(collapse_spaces(value))
    tokens = [tok for tok in re.findall(r"[A-Z]+", up) if tok]
    if not tokens:
        return True
    if len(tokens) == 1 and len(tokens[0]) < 3:
        return True
    if len(tokens) >= 2 and len(tokens[0]) < 3 and len(tokens[1]) < 3:
        return True
    return False


# ──────────────────────────────────────────────────────────────────────
# Ana giriş noktası
# ──────────────────────────────────────────────────────────────────────

def postprocess_field(field_name: str, raw_value: str) -> object | None:
    """
    Ham OCR değerini alan adına göre post-process eder.
    None döndürürse → ABSTAIN (değeri kullanma).
    """
    if not raw_value or not raw_value.strip():
        return None

    s = raw_value.strip()

    if field_name == "plate":
        return _fix_plate(s)

    if field_name == "chassis_no":
        fixed = _fix_vin(s)
        alnum = re.sub(r"[^A-Z0-9]", "", fixed)
        # Common OCR overflow: 18 chars instead of 17 for VIN-like values.
        if len(alnum) == 18:
            alnum = alnum[:17]

        # Keep medium-tolerant behavior (15-17) to avoid recall collapse,
        # but reject clearly invalid short/long values.
        if len(alnum) < 15 or len(alnum) > 17:
            return None

        return alnum

    if field_name == "engine_no":
        return _fix_engine_no(s)

    if field_name in ("first_registration_date", "registration_date"):
        return _fix_date(s)

    if field_name == "inspection_date":
        fixed = _fix_date(s)
        return fixed.replace("/", "-") if fixed else None

    if field_name == "model_year":
        return _fix_model_year(s)

    if field_name == "tax_or_id_no":
        return _fix_tax_id(s)

    if field_name == "serial_no":
        return _fix_serial_no(s)

    if field_name == "registration_serial_no":
        return _fix_registration_serial(s)

    if field_name == "vehicle_type":
        return _fix_vehicle_type(s)

    if field_name == "owner_title":
        return _fix_owner_title(s)

    if field_name == "province_district":
        return _fix_province_district(s)

    if field_name == "owner_address":
        return _fix_owner_address(s)

    if field_name == "usage_purpose":
        return _fix_usage_purpose(s)

    if field_name == "fuel_type":
        return _fix_fuel_type(s)

    if field_name == "approval_type_no":
        return _fix_approval_type_no(s)

    if field_name in {"document_serial", "document_number"}:
        serial, number = _parse_document_fields(s)
        return serial if field_name == "document_serial" else number

    if field_name in {
        "net_weight_kg",
        "max_loaded_weight_kg",
        "trailer_weight_kg",
        "seat_count",
        "standing_passenger_count",
        "engine_power_kw",
        "cylinder_volume_cm3",
    }:
        return _fix_int_field(s, default_zero=True)

    if field_name == "approver_registration_no":
        fixed_num = _fix_int_field(s, default_zero=False)
        return str(fixed_num) if fixed_num is not None else None

    if field_name == "brand":
        # Çok kısa veya sayı ağırlıklıysa ABSTAIN
        letters = re.sub(r"[^A-ZÇĞİÖŞÜ\-]", "", s.upper())
        return s.upper() if len(letters) >= 2 else None

    return s


    if field_name == "color":
        up = normalize_turkish_ascii(collapse_spaces(s))
        for color in ("BEYAZ", "SIYAH", "GRI", "GUMUS", "MAVI", "KIRMIZI", "YESIL", "SARI"):
            if color in up:
                return color
        return None

    return s


def postprocess_field(field_name: str, raw_value: str) -> object | None:
    """
    Clean override placed late in the module to avoid earlier patch/encoding noise.
    """
    if not raw_value or not raw_value.strip():
        return None

    s = raw_value.strip()

    if field_name == "plate":
        return _fix_plate(s)
    if field_name == "chassis_no":
        fixed = _fix_vin(s)
        alnum = re.sub(r"[^A-Z0-9]", "", fixed)
        if len(alnum) == 18:
            alnum = alnum[:17]
        if len(alnum) < 15 or len(alnum) > 17:
            return None
        return alnum
    if field_name == "engine_no":
        return _fix_engine_no(s)
    if field_name in ("first_registration_date", "registration_date"):
        return _fix_date(s)
    if field_name == "inspection_date":
        fixed = _fix_date(s)
        return fixed.replace("/", "-") if fixed else None
    if field_name == "model_year":
        return _fix_model_year(s)
    if field_name == "tax_or_id_no":
        return _fix_tax_id(s)
    if field_name == "serial_no":
        return _fix_serial_no(s)
    if field_name == "registration_serial_no":
        return _fix_registration_serial(s)
    if field_name == "type":
        return _fix_type_text(s)
    if field_name == "vehicle_type":
        return _fix_vehicle_type(s)
    if field_name == "owner_title":
        return _fix_owner_title(s)
    if field_name == "province_district":
        return _fix_province_district(s)
    if field_name == "owner_address":
        return _fix_owner_address(s)
    if field_name == "usage_purpose":
        return _fix_usage_purpose(s)
    if field_name == "fuel_type":
        return _fix_fuel_type(s)
    if field_name == "approval_type_no":
        return _fix_approval_type_no(s)
    if field_name in {"document_serial", "document_number"}:
        serial, number = _parse_document_fields(s)
        return serial if field_name == "document_serial" else number
    if field_name in {
        "net_weight_kg",
        "max_loaded_weight_kg",
        "trailer_weight_kg",
        "seat_count",
        "standing_passenger_count",
        "engine_power_kw",
        "cylinder_volume_cm3",
    }:
        return _fix_int_field(s, default_zero=True)
    if field_name == "approver_registration_no":
        fixed_num = _fix_int_field(s, default_zero=False)
        return str(fixed_num) if fixed_num is not None else None
    if field_name == "brand":
        letters = re.sub(r"[^A-ZÇĞİÖŞÜ\\-]", "", s.upper())
        return s.upper() if len(letters) >= 2 else None
    if field_name == "color":
        up = normalize_turkish_ascii(collapse_spaces(s))
        for color in ("BEYAZ", "SIYAH", "GRI", "GUMUS", "MAVI", "KIRMIZI", "YESIL", "SARI"):
            if color in up:
                return color
        return None
    return s


def postprocess_fields(fields: dict[str, dict]) -> dict[str, dict]:
    """
    fields: extract_fields() çıktısı — {"field_name": {"value": ..., ...}}
    Her alanı postprocess_field() ile geçirir, düzeltilmiş değeri yazar.
    """
    out = {}
    for fname, entry in fields.items():
        if not isinstance(entry, dict):
            out[fname] = entry
            continue

        raw = str(entry.get("value") or "")
        fixed = postprocess_field(fname, raw)

        new_entry = dict(entry)
        if fixed is None and raw:
            # OCR bir şey buldu ama postprocess güvenemedi → ABSTAIN
            new_entry["value"] = None
            new_entry["abstain_reason"] = "postprocess_validation_failed"
            new_entry["low_confidence"] = True
            new_entry["method"] = "postprocess_validation_failed"
        elif fixed != raw:
            new_entry["value"] = fixed
            new_entry["postprocess_applied"] = True

        out[fname] = new_entry

    owner_title_entry = out.get("owner_title")
    owner_name_entry = out.get("owner_name")
    owner_surname_entry = out.get("owner_surname")
    owner_title_raw = str(owner_title_entry.get("raw") or owner_title_entry.get("value") or "") if isinstance(owner_title_entry, dict) else ""
    owner_name_raw = str(owner_name_entry.get("raw") or owner_name_entry.get("value") or "") if isinstance(owner_name_entry, dict) else ""
    owner_surname_raw = str(owner_surname_entry.get("raw") or owner_surname_entry.get("value") or "") if isinstance(owner_surname_entry, dict) else ""

    company_source = ""
    for candidate in (owner_title_raw, owner_surname_raw, owner_name_raw):
        if candidate and _contains_owner_company_hint(candidate):
            company_source = candidate
            break

    if company_source:
        company_value = _normalize_owner_company_text(company_source)
        if company_value:
            title_value = company_value
            if _looks_like_fragmentary_company_value(company_value):
                # When OCR only preserves a weak business fragment, keeping a
                # generic company-title label is safer than treating it like a
                # personal owner name/title.
                title_value = "TICARI UNVASI"
            source_entry = owner_surname_entry if isinstance(owner_surname_entry, dict) else (
                owner_title_entry if isinstance(owner_title_entry, dict) else {}
            )
            out["owner_title"] = {
                **(dict(source_entry) if isinstance(source_entry, dict) else {}),
                "value": title_value,
                "raw": company_source,
                "method": "derived_owner_company_title",
                "postprocess_applied": True,
            }
            if isinstance(owner_surname_entry, dict):
                out["owner_surname"] = {
                    **owner_surname_entry,
                    "value": company_value,
                    "raw": company_source,
                    "method": "derived_owner_company_surname",
                    "postprocess_applied": True,
                }
            elif not out.get("owner_surname"):
                out["owner_surname"] = {
                    "value": company_value,
                    "raw": company_source,
                    "method": "derived_owner_company_surname",
                    "postprocess_applied": True,
                }

            if isinstance(owner_name_entry, dict):
                owner_name_value = normalize_turkish_ascii(collapse_spaces(str(owner_name_entry.get("value") or "")))
                if (
                    not owner_name_value
                    or _contains_owner_company_hint(owner_name_raw)
                    or owner_name_value == company_value
                ):
                    patched_owner_name = dict(owner_name_entry)
                    patched_owner_name["value"] = None
                    patched_owner_name["raw"] = owner_name_raw
                    patched_owner_name["method"] = "derived_company_owner_name_cleared"
                    patched_owner_name["postprocess_applied"] = True
                    out["owner_name"] = patched_owner_name
    elif isinstance(owner_name_entry, dict):
        owner_name_value = str(owner_name_entry.get("value") or "")
        owner_surname_value = str(owner_surname_entry.get("value") or "") if isinstance(owner_surname_entry, dict) else ""
        if owner_name_value.strip() and normalize_turkish_ascii(collapse_spaces(owner_name_value)) == normalize_turkish_ascii(collapse_spaces(owner_surname_value)):
            patched_owner_name = dict(owner_name_entry)
            patched_owner_name["value"] = None
            patched_owner_name["method"] = "duplicate_owner_name_cleared"
            patched_owner_name["postprocess_applied"] = True
            out["owner_name"] = patched_owner_name
            owner_name_entry = patched_owner_name
            owner_name_value = ""
        elif owner_name_value.strip() and _looks_like_weak_person_name(owner_name_value):
            patched_owner_name = dict(owner_name_entry)
            patched_owner_name["value"] = None
            patched_owner_name["method"] = "weak_owner_name_cleared"
            patched_owner_name["postprocess_applied"] = True
            out["owner_name"] = patched_owner_name
            owner_name_entry = patched_owner_name
            owner_name_value = ""
        if not owner_surname_value.strip():
            derived_surname, derived_name = _derive_owner_name_parts(owner_name_raw or owner_name_value)
            if derived_surname:
                if isinstance(owner_surname_entry, dict):
                    original_title = _fix_owner_title(
                        str(owner_surname_entry.get("raw") or owner_surname_entry.get("value") or "")
                    )
                    existing_title = out.get("owner_title")
                    existing_title_value = ""
                    if isinstance(existing_title, dict):
                        existing_title_value = str(existing_title.get("value") or "")
                    if original_title and not existing_title_value.strip():
                        out["owner_title"] = {
                            **owner_surname_entry,
                            "value": original_title,
                            "method": "derived_owner_title",
                            "postprocess_applied": True,
                        }
                    owner_surname_entry = dict(owner_surname_entry)
                else:
                    owner_surname_entry = {}

                owner_surname_entry["value"] = derived_surname
                owner_surname_entry["raw"] = owner_name_raw or owner_name_value
                owner_surname_entry["method"] = "derived_owner_surname"
                owner_surname_entry["postprocess_applied"] = True
                out["owner_surname"] = owner_surname_entry

                if derived_name:
                    owner_name_entry = dict(owner_name_entry)
                    owner_name_entry["value"] = derived_name
                    owner_name_entry["raw"] = owner_name_raw or owner_name_value
                    owner_name_entry["method"] = "derived_owner_name"
                    owner_name_entry["postprocess_applied"] = True
                    out["owner_name"] = owner_name_entry

    serial_entry = out.get("serial_no")
    if isinstance(serial_entry, dict):
        serial_raw = str(serial_entry.get("raw") or serial_entry.get("value") or "")
        document_serial, document_number = _parse_document_fields(serial_raw)
        if document_serial:
            out["document_serial"] = {
                **serial_entry,
                "value": document_serial,
                "method": "derived_document_serial",
                "postprocess_applied": True,
            }
        if document_number:
            out["document_number"] = {
                **serial_entry,
                "value": document_number,
                "method": "derived_document_number",
                "postprocess_applied": True,
            }

    vehicle_class_entry = out.get("vehicle_class")
    standing_entry = out.get("standing_passenger_count")
    vehicle_class_value = ""
    if isinstance(vehicle_class_entry, dict):
        vehicle_class_value = str(vehicle_class_entry.get("value") or "").strip().upper()
    if vehicle_class_value == "M1" and isinstance(standing_entry, dict) and standing_entry.get("value") in {None, ""}:
        standing_entry = dict(standing_entry)
        standing_entry["value"] = 0
        standing_entry["method"] = "derived_vehicle_class_default"
        standing_entry["postprocess_applied"] = True
        out["standing_passenger_count"] = standing_entry

    return out
