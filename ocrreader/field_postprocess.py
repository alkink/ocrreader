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
            if c in _VIN_CHAR_MAP:
                result[i] = _VIN_CHAR_MAP[c]
            elif c == "I":
                result[i] = "1"  # pos 3+ → I hep 1
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


# ──────────────────────────────────────────────────────────────────────
# Ana giriş noktası
# ──────────────────────────────────────────────────────────────────────

def postprocess_field(field_name: str, raw_value: str) -> str | None:
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

    if field_name in ("first_registration_date", "registration_date", "inspection_date"):
        return _fix_date(s)

    if field_name == "model_year":
        return _fix_model_year(s)

    if field_name == "tax_or_id_no":
        return _fix_tax_id(s)

    if field_name == "serial_no":
        return _fix_serial_no(s)

    if field_name == "brand":
        # Çok kısa veya sayı ağırlıklıysa ABSTAIN
        letters = re.sub(r"[^A-ZÇĞİÖŞÜ\-]", "", s.upper())
        return s.upper() if len(letters) >= 2 else None

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
    return out
