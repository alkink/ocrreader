from __future__ import annotations

from datetime import datetime
import re
from typing import Any

from .text_utils import collapse_spaces, normalize_turkish_ascii


PLATE_RE = re.compile(r"^\d{2}[A-Z]{1,3}\d{2,4}$")
VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")
ENGINE_RE = re.compile(r"^[A-Z0-9]{6,25}$")
DATE_RE = re.compile(r"^(\d{2})[/-](\d{2})[/-](\d{4})$")
REG_SERIAL_RE = re.compile(r"^[A-Z0-9]{16,22}$")
APPROVAL_RE = re.compile(r"^[A-Z0-9][A-Z0-9*/.\-]+(?:\*[A-Z0-9/.\-]+)+$", re.IGNORECASE)
DOC_SERIAL_RE = re.compile(r"^[A-Z]{1,4}$")
DOC_NUMBER_RE = re.compile(r"^\d{4,10}$")
APPROVER_RE = re.compile(r"^\d{1,6}$")

VALID_VEHICLE_CLASSES = {
    "L1", "L2", "L3", "L4", "L5", "L6", "L7",
    "M1", "M2", "M3",
    "N1", "N2", "N3",
    "O1", "O2", "O3", "O4",
}
KNOWN_COLORS = {"BEYAZ", "SIYAH", "GRI", "GUMUS", "MAVI", "KIRMIZI", "YESIL", "SARI", "LACIVERT", "TURUNCU", "KAHVERENGI"}
KNOWN_FUELS = {"DIZEL", "BENZIN", "ELEKTRIK", "HIBRIT", "LPG", "CNG"}
HEADER_BLEED_TOKENS = {
    "VERILDIGI", "IL", "ILCE", "NOTER", "NOTERLIGI", "TICARI", "UNVANI", "SOYADI", "ADI", "ADRESI"
}
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
    "KIRTASIYE",
    "OFIS",
    "UR",
    "URUN",
    "IMAL",
    "INSAAT",
    "A.S",
    "A.S.",
}

CRITICAL_FIELDS = {
    "plate",
    "engine_no",
    "chassis_no",
    "tax_or_id_no",
    "brand",
    "type",
    "owner_surname",
    "owner_name",
    "first_registration_date",
    "registration_date",
    "province_district",
    "registration_serial_no",
    "vehicle_type",
    "document_serial",
    "document_number",
}

IMPORTANT_FIELDS = {
    "model_year",
    "vehicle_class",
    "color",
    "owner_title",
    "owner_address",
    "inspection_date",
    "net_weight_kg",
    "max_loaded_weight_kg",
    "trailer_weight_kg",
    "seat_count",
    "standing_passenger_count",
    "engine_power_kw",
    "cylinder_volume_cm3",
    "fuel_type",
    "usage_purpose",
    "approval_type_no",
    "approver_registration_no",
}


def _entry_value(entry: Any) -> Any:
    if isinstance(entry, dict):
        return entry.get("value")
    return entry


def _entry_method(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("method") or "")
    return ""


def _entry_confidence(entry: Any) -> int | None:
    if isinstance(entry, dict):
        raw = entry.get("confidence_score")
        if raw is None:
            return None
        try:
            return int(raw)
        except Exception:
            return None
    return None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return collapse_spaces(normalize_turkish_ascii(str(value)))


def _parse_date(value: str) -> datetime | None:
    m = DATE_RE.fullmatch(value)
    if not m:
        return None
    try:
        return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def _owner_tokens(value: str) -> list[str]:
    return [tok for tok in re.findall(r"[A-Z0-9.]+", _normalize_text(value)) if tok]


def _contains_owner_company_hint(value: Any) -> bool:
    text = _normalize_text(value)
    if not text:
        return False
    tokens = _owner_tokens(text)
    if any(tok in OWNER_COMPANY_HINTS for tok in tokens):
        return True
    joined = " ".join(tokens)
    return any(fragment in joined for fragment in ("LIMITED", "SIRKET", "A.S", "A.S."))


def _owner_mode(flat: dict[str, Any]) -> str:
    owner_title = flat.get("owner_title")
    owner_surname = flat.get("owner_surname")
    owner_name = flat.get("owner_name")
    if any(_contains_owner_company_hint(v) for v in (owner_title, owner_surname, owner_name)):
        return "company"
    if _normalize_text(owner_name) or _normalize_text(owner_surname):
        return "person"
    return "unknown"


def _penalty(field_name: str, status: str) -> int:
    if status == "ok":
        return 0
    if field_name in CRITICAL_FIELDS:
        return 12 if status == "error" else 5
    if field_name in IMPORTANT_FIELDS:
        return 6 if status == "error" else 3
    return 3 if status == "error" else 1


def _missing_status(field_name: str) -> tuple[str, str]:
    if field_name in CRITICAL_FIELDS:
        return "error", "required_field_missing"
    if field_name in IMPORTANT_FIELDS:
        return "warning", "important_field_missing"
    return "warning", "field_missing"


def _validate_field(field_name: str, value: Any, flat: dict[str, Any]) -> tuple[str, str]:
    text = _normalize_text(value)
    owner_mode = _owner_mode(flat)
    if not text:
        if field_name == "owner_name" and owner_mode == "company":
            return "ok", "not_required_for_company_owner"
        if field_name == "owner_surname" and owner_mode == "company" and (
            _contains_owner_company_hint(flat.get("owner_title")) or _contains_owner_company_hint(flat.get("owner_surname"))
        ):
            return "ok", "covered_by_company_title"
        if field_name == "owner_title" and owner_mode == "company" and _contains_owner_company_hint(flat.get("owner_surname")):
            return "ok", "covered_by_company_surname"
        return _missing_status(field_name)

    if field_name == "plate":
        return ("ok", "valid") if PLATE_RE.fullmatch(text) else ("error", "invalid_plate_format")

    if field_name == "chassis_no":
        return ("ok", "valid") if VIN_RE.fullmatch(text) else ("error", "invalid_vin")

    if field_name == "engine_no":
        return ("ok", "valid") if ENGINE_RE.fullmatch(text) else ("error", "invalid_engine_no")

    if field_name == "tax_or_id_no":
        return ("ok", "valid") if re.fullmatch(r"\d{10,11}", text) else ("error", "invalid_tax_or_id")

    if field_name == "brand":
        if any(token in text for token in HEADER_BLEED_TOKENS):
            return "error", "header_bleed_brand"
        return ("ok", "valid") if re.fullmatch(r"[A-Z][A-Z0-9 .-]{1,40}", text) else ("error", "invalid_brand")

    if field_name == "type":
        if any(token in text for token in {"VERGI", "NOTER", "ILCE", "ADRESI"}):
            return "error", "header_bleed_type"
        compact = re.sub(r"[^A-Z0-9]", "", text)
        return ("ok", "valid") if len(compact) >= 4 else ("error", "type_too_short")

    if field_name in {"owner_surname", "owner_name"}:
        if field_name == "owner_surname" and (owner_mode == "company" or _contains_owner_company_hint(text)):
            return ("ok", "valid_company_owner") if len(text) >= 3 else ("warning", "company_owner_unusual")
        if any(token in text.split() for token in HEADER_BLEED_TOKENS):
            return "error", "header_bleed_owner_name"
        other = _normalize_text(flat.get("owner_name" if field_name == "owner_surname" else "owner_surname"))
        if other and text == other:
            return "warning", "owner_name_duplicate"
        if field_name == "owner_name" and owner_mode == "company":
            if _contains_owner_company_hint(text):
                return "warning", "company_text_in_owner_name"
            return ("ok", "valid") if re.fullmatch(r"[A-Z ]{2,60}", text) else ("warning", "owner_name_unusual")
        return ("ok", "valid") if re.fullmatch(r"[A-Z ]{2,60}", text) else ("warning", "owner_name_unusual")

    if field_name == "owner_title":
        if "NOTERLIGI" in text or "ILCE" in text:
            return "error", "owner_title_header_bleed"
        if owner_mode == "company" or _contains_owner_company_hint(text):
            return ("ok", "valid_company_title") if len(text) >= 3 else ("warning", "company_title_unusual")
        return ("ok", "valid") if len(text) >= 2 else ("warning", "owner_title_short")

    if field_name in {"first_registration_date", "registration_date", "inspection_date"}:
        return ("ok", "valid") if _parse_date(text) else ("error", "invalid_date")

    if field_name == "province_district":
        if "/" not in text or "NOTERLIGI" not in text:
            return "warning", "province_district_unusual"
        return "ok", "valid"

    if field_name == "registration_serial_no":
        return ("ok", "valid") if REG_SERIAL_RE.fullmatch(text) else ("warning", "registration_serial_unusual")

    if field_name == "vehicle_class":
        return ("ok", "valid") if text in VALID_VEHICLE_CLASSES else ("error", "invalid_vehicle_class")

    if field_name == "vehicle_type":
        compact = re.sub(r"[^A-Z0-9]", "", text)
        return ("ok", "valid") if len(compact) >= 4 else ("error", "vehicle_type_too_short")

    if field_name == "color":
        return ("ok", "valid") if text in KNOWN_COLORS else ("warning", "unknown_color")

    if field_name == "owner_address":
        return ("ok", "valid") if len(text) >= 12 else ("warning", "address_too_short")

    if field_name in {"net_weight_kg", "max_loaded_weight_kg", "trailer_weight_kg", "seat_count", "standing_passenger_count", "engine_power_kw", "cylinder_volume_cm3"}:
        try:
            number = int(value)
        except Exception:
            return "error", "invalid_numeric_field"
        if field_name == "seat_count":
            return ("ok", "valid") if 1 <= number <= 99 else ("error", "seat_count_out_of_range")
        if field_name == "standing_passenger_count":
            if _normalize_text(flat.get("vehicle_class")) == "M1" and number != 0:
                return "error", "standing_passenger_inconsistent_with_m1"
            return ("ok", "valid") if 0 <= number <= 99 else ("error", "standing_passenger_out_of_range")
        if field_name == "engine_power_kw":
            return ("ok", "valid") if 1 <= number <= 2000 else ("error", "engine_power_out_of_range")
        if field_name == "cylinder_volume_cm3":
            return ("ok", "valid") if 1 <= number <= 10000 else ("error", "cylinder_volume_out_of_range")
        if field_name in {"net_weight_kg", "max_loaded_weight_kg"}:
            return ("ok", "valid") if 100 <= number <= 10000 else ("error", "weight_out_of_range")
        if field_name == "trailer_weight_kg":
            return ("ok", "valid") if 0 <= number <= 10000 else ("error", "trailer_weight_out_of_range")

    if field_name == "fuel_type":
        return ("ok", "valid") if text in KNOWN_FUELS else ("warning", "unknown_fuel_type")

    if field_name == "usage_purpose":
        return ("ok", "valid") if re.search(r"(YOLCU|YUK|HUSUSI|TICARI)", text) else ("warning", "usage_purpose_unusual")

    if field_name == "approval_type_no":
        return ("ok", "valid") if APPROVAL_RE.fullmatch(text) else ("error", "invalid_approval_type_no")

    if field_name == "document_serial":
        return ("ok", "valid") if DOC_SERIAL_RE.fullmatch(text) else ("warning", "document_serial_unusual")

    if field_name == "document_number":
        return ("ok", "valid") if DOC_NUMBER_RE.fullmatch(text) else ("error", "invalid_document_number")

    if field_name == "approver_registration_no":
        return ("ok", "valid") if APPROVER_RE.fullmatch(text) else ("warning", "invalid_approver_registration_no")

    if field_name == "model_year":
        try:
            year = int(text)
        except Exception:
            return "error", "invalid_model_year"
        return ("ok", "valid") if 1950 <= year <= 2035 else ("error", "model_year_out_of_range")

    return "ok", "not_checked"


def _detect_profile(flat: dict[str, Any], anchor_count: int, field_status: dict[str, dict[str, Any]]) -> dict[str, Any]:
    strong_hits = sum(
        1
        for key in ("plate", "province_district", "registration_serial_no", "brand", "type", "vehicle_type")
        if field_status.get(key, {}).get("status") == "ok"
    )
    if anchor_count >= 8 and strong_hits >= 5:
        return {
            "selected_profile": "v29_photo_structured",
            "confidence": round(min(0.98, 0.55 + anchor_count * 0.03 + strong_hits * 0.04), 2),
        }
    if strong_hits >= 3:
        return {
            "selected_profile": "v29_photo_variant_b_candidate",
            "confidence": round(min(0.85, 0.35 + strong_hits * 0.06), 2),
        }
    return {
        "selected_profile": "unknown_or_degraded_photo",
        "confidence": 0.25,
    }


def evaluate_extraction_review(
    fields: dict[str, Any],
    anchors: dict[str, Any],
) -> dict[str, Any]:
    flat = {name: _entry_value(entry) for name, entry in fields.items()}
    field_status: dict[str, dict[str, Any]] = {}
    penalties = 0

    for field_name, value in flat.items():
        status, reason = _validate_field(field_name, value, flat)
        penalties += _penalty(field_name, status)
        field_status[field_name] = {
            "status": status,
            "reason": reason,
            "critical": field_name in CRITICAL_FIELDS,
            "important": field_name in IMPORTANT_FIELDS,
            "method": _entry_method(fields.get(field_name)),
            "confidence_score": _entry_confidence(fields.get(field_name)),
        }

    first_date = _parse_date(_normalize_text(flat.get("first_registration_date")))
    reg_date = _parse_date(_normalize_text(flat.get("registration_date")))
    if first_date and reg_date and reg_date < first_date:
        penalties += 8
        field_status["registration_date"]["status"] = "error"
        field_status["registration_date"]["reason"] = "registration_before_first_registration"

    score = max(0, 100 - penalties)
    critical_errors = [name for name, meta in field_status.items() if meta["critical"] and meta["status"] == "error"]
    warnings = [name for name, meta in field_status.items() if meta["status"] == "warning"]
    errors = [name for name, meta in field_status.items() if meta["status"] == "error"]
    anchor_count = len(anchors)
    profile = _detect_profile(flat, anchor_count, field_status)

    needs_review = bool(
        critical_errors
        or score < 92
        or len(errors) >= 3
        or (profile["selected_profile"] == "unknown_or_degraded_photo")
    )

    suggested_action = "manual_review" if needs_review else "save_to_database"

    return {
        "needs_review": needs_review,
        "review_score": score,
        "suggested_action": suggested_action,
        "critical_error_fields": critical_errors,
        "warning_fields": warnings,
        "error_fields": errors,
        "anchor_count": anchor_count,
        "profile": profile,
        "field_status": field_status,
    }
