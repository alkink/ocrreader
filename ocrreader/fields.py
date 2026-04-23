from __future__ import annotations

from difflib import SequenceMatcher
import re

import cv2
import numpy as np

from .anchors import AnchorMatch
from .config import FieldConfig, OCRConfig
from .field_value_locator import locate_value_from_anchor, locate_value_from_roi_words
from .ocr_engine import OCREngine, TextReadEngine
from .preprocess import crop
from .text_utils import collapse_spaces, normalize_turkish_ascii
from .types import Rect, union_rects


LABEL_NOISE = {
    "PLAKA",
    "MARKASI",
    "MARKA",
    "TIPI",
    "TIP",
    "MODEL",
    "MODEL",
    "YILI",
    "MOTOR",
    "SASE",
    "SERI",
    "BELGE",
    "TESCIL",
    "TARIHI",
    "DIGER",
    "BILGILER",
    "NOTER",
    "SOYADI",
    "ADI",
    "VERGI",
    "NO",
}

VEHICLE_TYPE_BLOCK = {
    "M",
    "S",
    "NO",
    "KG",
    "KW",
    "CM3",
    "MODEL",
    "YILI",
    "ARAC",
    "SINIFI",
    "SASE",
    "MOTOR",
}

OWNER_COMPANY_HINTS = {
    "TICARI",
    "UNVANI",
    "UNVAN",
    "LIMITED",
    "LTD",
    "SIRKETI",
    "SAN",
    "SANAYI",
    "TURIZM",
    "OTOMOTIV",
    "KIRTASIYE",
    "PROMOSYON",
    "OFIS",
    "UR",
    "URUN",
    "IMAL",
}

OWNER_COMPANY_FRAGMENTS = (
    "LIMITED",
    "SIRKET",
    "OFIS",
    "KIRTASIYE",
    "PROMOSYON",
    "OTOMOTIV",
    "EMLAK",
    "TASIMACILIK",
    "SANAYI",
    "INSAAT",
    "TURIZM",
    "UNVAN",
)

GENERIC_VEHICLE_TYPE_TOKENS = {
    "OTOMOBIL",
    "OTOBUS",
    "MOTOSIKLET",
    "KAMYON",
    "KAMYONET",
    "MINIBUS",
    "PANELVAN",
    "SEDAN",
    "HATCHBACK",
    "STATIONWAGON",
    "TEK",
    "CIFT",
    "KATLI",
    "AB",
    "AA",
    "AC",
    "CA",
    "HB",
}

VEHICLE_TYPE_NOISE = {
    "D",
    "D4",
    "D.4",
    "TABLE",
    "BORDER",
    "TARIHI",
    "YONET",
    "MARKDOWN",
    "UTF-8",
    "ADI",
    "SOYADI",
    "MODEL",
    "YILI",
    "BEYAZ",
    "SIYAH",
    "GRI",
    "KIRMIZI",
    "MAVI",
    "SARI",
    "YESIL",
    "LACIVERT",
    "TURUNCU",
    "KAHVERENGI",
    "RENGI",
    "L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7",
    "M1", "M2", "M3",
    "N1", "N2", "N3",
    "O1", "O2", "O3", "O4",
}

PLATE_PATTERN = re.compile(r"\d{2}[A-Z]{1,3}\d{2,4}")
PLATE_TEXT_PATTERN = re.compile(r"\b\d{2}\s?[A-Z]{1,3}\s?\d{2,4}\b")
VIN_PATTERN = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
DATE_PATTERN = re.compile(r"\b(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{2,4})\b")
YEAR_PATTERN = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
TAX_ID_PATTERN = re.compile(r"\b\d{10,11}\b")

KNOWN_BRANDS = {
    "RENAULT",
    "FORD",
    "MERCEDES-BENZ",
    "VOLKSWAGEN",
    "TOYOTA",
    "HYUNDAI",
    "KIA",
    "BMW",
    "AUDI",
    "FIAT",
    "OPEL",
    "PEUGEOT",
    "CITROEN",
    "NISSAN",
    "HONDA",
    "MAZDA",
    "VOLVO",
    "SKODA",
    "SEAT",
    "DACIA",
    "MITSUBISHI",
    "SUZUKI",
    "ISUZU",
    "TATA",
    "SYM",
    "MONDIAL",
    "ARORA",
    "CITYCOCO",
    "CUSWA",
}

BRAND_STOPWORDS = {
    "MARKASI",
    "MARKA",
    "ARAC",
    "ARACI",
    "OTOMOBIL",
    "OTOMOBILI",
    "CINSI",
    "SINIFI",
    "TIPI",
    "TIP",
}


def _norm_box_to_rect(box: tuple[float, float, float, float], doc_w: int, doc_h: int) -> Rect:
    x = int(box[0] * doc_w)
    y = int(box[1] * doc_h)
    w = int(box[2] * doc_w)
    h = int(box[3] * doc_h)
    return Rect(x=x, y=y, w=max(1, w), h=max(1, h))


def resolve_field_rois(
    document_shape: tuple[int, int, int],
    fields: dict[str, FieldConfig],
    anchors: dict[str, object],
) -> dict[str, Rect]:
    doc_h, doc_w = document_shape[:2]
    rois: dict[str, Rect] = {}

    for field_name, cfg in fields.items():
        roi: Rect | None = None

        if cfg.anchor and cfg.offset_from_anchor_norm and cfg.anchor in anchors:
            anchor_box = anchors[cfg.anchor].bbox
            dx, dy, rw, rh = cfg.offset_from_anchor_norm
            roi = Rect(
                x=anchor_box.x + int(dx * doc_w),
                y=anchor_box.y + int(dy * doc_h),
                w=max(1, int(rw * doc_w)),
                h=max(1, int(rh * doc_h)),
            )

        if roi is None and cfg.fallback_norm:
            roi = _norm_box_to_rect(cfg.fallback_norm, doc_w, doc_h)

        if roi is not None:
            rois[field_name] = roi.clip(doc_w, doc_h)

    return rois


def preprocess_field_crop(crop_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return bw


def _preprocess_field_crop_alt(crop_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def cleanup_text(raw_text: str, strategy: str) -> str:
    text = collapse_spaces(raw_text)

    if strategy == "plate":
        normalized = normalize_turkish_ascii(text)
        tokens = re.findall(r"[A-Z0-9]+", normalized)
        if not tokens:
            return ""

        seen: set[str] = set()
        candidates: list[str] = []
        for i in range(len(tokens)):
            for j in range(i + 1, min(len(tokens), i + 4) + 1):
                cand = "".join(tokens[i:j])
                if not cand or cand in seen:
                    continue
                seen.add(cand)
                candidates.append(cand)

        for cand in candidates:
            compact = re.sub(r"[^A-Z0-9]", "", cand)
            if PLATE_PATTERN.fullmatch(compact):
                return compact

        return ""

    if strategy == "digits":
        groups = re.findall(r"\d+", text)
        if not groups:
            return ""
        return max(groups, key=len)

    if strategy == "date":
        normalized = normalize_turkish_ascii(text)
        normalized = normalized.replace(".", "/").replace("-", "/")
        m = re.search(r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{2,4})", normalized)
        if not m:
            return ""
        d, mo, y = m.group(1), m.group(2), m.group(3)
        if len(y) == 2:
            y = f"20{y}"
        return f"{int(d):02d}/{int(mo):02d}/{y}"

    if strategy == "alnum_upper":
        normalized = normalize_turkish_ascii(text)
        tokens = re.findall(r"[A-Z0-9-]+", normalized)
        if not tokens:
            return ""

        mixed = [t for t in tokens if re.search(r"[A-Z]", t) and re.search(r"\d", t)]
        source = mixed if mixed else tokens
        source = sorted(source, key=lambda t: len(re.sub(r"[^A-Z0-9]", "", t)), reverse=True)
        return re.sub(r"[^A-Z0-9]+", "", source[0])

    if strategy == "alnum_hyphen_upper":
        normalized = normalize_turkish_ascii(text)
        tokens = re.findall(r"[A-Z0-9-]+", normalized)
        if not tokens:
            return ""

        mixed = [t for t in tokens if re.search(r"[A-Z]", t) and re.search(r"\d", t)]
        source = mixed if mixed else tokens
        source = sorted(source, key=lambda t: len(re.sub(r"[^A-Z0-9]", "", t)), reverse=True)
        return re.sub(r"[^A-Z0-9-]+", "", source[0])

    if strategy == "text_upper":
        normalized = normalize_turkish_ascii(text)
        normalized = re.sub(r"[^A-Z0-9 /.-]+", " ", normalized)
        return collapse_spaces(normalized)

    if strategy == "owner_text":
        normalized = normalize_turkish_ascii(text)
        normalized = re.sub(r"[^A-Z0-9 /.-]+", " ", normalized)
        normalized = collapse_spaces(normalized)
        normalized = re.sub(r"\b(?:SOYADI|ADI|TICARI|UNVANI|UNVAN|ADRESI|ADRES|NOTER)\b", " ", normalized)
        normalized = collapse_spaces(normalized)
        parts = [p for p in normalized.split(" ") if len(p) > 1]
        if not parts:
            return ""
        return collapse_spaces(" ".join(parts[:10]))

    if strategy == "vehicle_type":
        normalized = normalize_turkish_ascii(text)
        normalized = re.sub(r"[^A-Z0-9 /.-]+", " ", normalized)
        normalized = collapse_spaces(normalized)
        normalized = re.sub(r"\b(?:TIPI|TIP|TYPE|MODEL|YILI|MARKASI|MARKA)\b", " ", normalized)
        normalized = collapse_spaces(normalized)
        tokens = re.findall(r"[A-Z0-9.-]+", normalized)
        if not tokens:
            return ""
        if len(tokens) == 1:
            return tokens[0]
        if any(bool(re.search(r"[A-Z]", t)) and bool(re.search(r"\d", t)) for t in tokens):
            mixed = [t for t in tokens if bool(re.search(r"[A-Z]", t)) and bool(re.search(r"\d", t))]
            return collapse_spaces(" ".join(mixed[:3]))
        return collapse_spaces(" ".join(tokens[:4]))

    return text


def _score_cleaned(cleaned: str, strategy: str) -> int:
    if not cleaned:
        return 0

    if strategy == "plate":
        compact = re.sub(r"[^A-Z0-9]", "", cleaned)
        if not compact:
            return 0
        return len(compact) + (12 if PLATE_PATTERN.fullmatch(compact) else 0)

    if strategy == "digits":
        return len(cleaned)

    if strategy == "date":
        return 100 if re.fullmatch(r"\d{2}/\d{2}/\d{4}", cleaned) else 0

    if strategy == "alnum_upper":
        has_alpha = bool(re.search(r"[A-Z]", cleaned))
        has_digit = bool(re.search(r"\d", cleaned))
        bonus = 8 if has_alpha and has_digit else 0
        return len(cleaned) + bonus

    if strategy == "alnum_hyphen_upper":
        pure = re.sub(r"[^A-Z0-9]", "", cleaned)
        has_alpha = bool(re.search(r"[A-Z]", pure))
        has_digit = bool(re.search(r"\d", pure))
        bonus = 8 if has_alpha and has_digit else 0
        return len(pure) + bonus

    if strategy == "text_upper":
        tokens = re.findall(r"[A-Z0-9]+", cleaned)
        if not tokens:
            return 0
        longest = max(len(t) for t in tokens)
        mixed = any(bool(re.search(r"[A-Z]", t)) and bool(re.search(r"\d", t)) for t in tokens)
        return longest + (4 if mixed else 0)

    if strategy == "owner_text":
        tokens = re.findall(r"[A-Z0-9]+", cleaned)
        if not tokens:
            return 0
        letter_tokens = [t for t in tokens if re.search(r"[A-Z]", t)]
        penalty = 0
        if re.search(r"\b(?:SOYADI|ADI|TICARI|UNVAN|ADRES|NOTER)\b", cleaned):
            penalty += 12
        return max(0, sum(len(t) for t in letter_tokens[:6]) - penalty)

    if strategy == "vehicle_type":
        tokens = re.findall(r"[A-Z0-9.-]+", cleaned)
        if not tokens:
            return 0
        mixed = [t for t in tokens if bool(re.search(r"[A-Z]", t)) and bool(re.search(r"\d", t))]
        bonus = 8 if mixed else 0
        return sum(min(len(t), 12) for t in tokens[:4]) + bonus

    return len(cleaned)


def _looks_like_label_noise(value: str) -> bool:
    if not value:
        return True
    up = normalize_turkish_ascii(value)
    toks = re.findall(r"[A-Z0-9]+", up)
    if not toks:
        return True
    if len(toks) == 1 and len(toks[0]) <= 2:
        return True
    noise = sum(1 for t in toks if t in LABEL_NOISE)
    return noise >= max(1, len(toks) - 1)


def _canonicalize_brand(value: str) -> str:
    up = normalize_turkish_ascii(value)
    toks = [
        t
        for t in re.findall(r"[A-Z0-9-]+", up)
        if t and t not in BRAND_STOPWORDS and t not in LABEL_NOISE
    ]
    if not toks:
        return ""

    cand = collapse_spaces(" ".join(toks[:3]))
    cand_key = re.sub(r"[^A-Z0-9]", "", cand)
    if not cand_key:
        return ""

    best = ""
    best_score = 0.0
    for brand in KNOWN_BRANDS:
        brand_key = re.sub(r"[^A-Z0-9]", "", brand)
        if cand_key == brand_key:
            return brand

        score = SequenceMatcher(None, cand_key, brand_key).ratio()
        if len(cand_key) >= 4 and (cand_key[:4] in brand_key or brand_key[:4] in cand_key):
            score = max(score, 0.72)

        if score > best_score:
            best_score = score
            best = brand

    if best and best_score >= 0.72:
        return best
    return ""


def _post_field_filters(field_name: str, value: str) -> str:
    up = normalize_turkish_ascii(value)
    if field_name == "plate":
        up = cleanup_text(up, "plate")
        if re.fullmatch(r"\d{5,}", up):
            return ""
        if up in {"MARK", "PLAKA", "TESCIL"}:
            return ""
        # strict TR plate pattern: 2 digits + 1-3 letters + 2-4 digits
        if not PLATE_PATTERN.fullmatch(up):
            return ""
        return up

    if field_name in {"serial_no", "owner_surname", "owner_name", "brand"}:
        if _looks_like_label_noise(up):
            return ""

    if field_name == "brand":
        return _canonicalize_brand(up)

    if field_name == "serial_no":
        nums = re.findall(r"\d{4,7}", up)
        if nums:
            return nums[-1]
        return ""

    if field_name == "owner_surname":
        # Reject common label bleed that GLM hallucinates as surname
        if up in {"TESCIL TARIHI", "TESCIL TARIH", "TARIHI", "TESCIL", "VERGI DAIRESI", "VERGI NO", "MARKA", "MARKASI"}:
            return ""
        if re.search(r"\b(?:TESCIL|NOTER|VERGI|ILCE|PLAKA)\b", up) and not any(
            hint in up for hint in OWNER_COMPANY_HINTS
        ):
            return ""
        # preserve company-style surnames/unvan as-is; only remove obvious field label tokens.
        up = re.sub(r"\b(?:SOYADI|ADI)\b", " ", up)
        up = re.sub(r"\bTICARI\s+UNVAN[I1]?\b", "TICARI UNVANI", up)
        # strip trailing address tail when it starts (very common OCR bleed from lower rows)
        up = re.split(r"\b(?:ADRESI|ADRES|MAH\.?|CAD\.?|SOK\.?|NO\b)\b", up)[0]
        up = re.sub(r"\b\d{2,}\b", " ", up)
        up = collapse_spaces(up)
        toks = re.findall(r"[A-Z]+", up)
        if len(up) < 3 or not toks:
            return ""

        company_hits = [t for t in toks if t in OWNER_COMPANY_HINTS]
        if company_hits:
            cleaned_toks = [t for t in toks if len(t) >= 2 and t not in {"NOTER", "TESCIL", "TARIHI"}]
            if not cleaned_toks:
                return ""
            return collapse_spaces(" ".join(cleaned_toks[:8]))

        # if not a company title line, keep compact personal surname-like chunk
        if "TICARI" not in up and "LIMITED" not in up and "SIRKETI" not in up:
            alpha = [t for t in toks if len(t) >= 3]
            if not alpha:
                return ""
            # prefer first meaningful token(s), e.g. YILMAZ / AKIN
            return collapse_spaces(" ".join(alpha[:2]))

        return up

    if field_name == "owner_name":
        toks = re.findall(r"[A-Z]+", up)
        if not toks:
            return ""
        if any(t in OWNER_COMPANY_HINTS for t in toks):
            return ""
        alpha = [t for t in toks if len(t) >= 2]
        if not alpha:
            return ""
        if len(alpha) >= 2 and len(alpha[0]) < 3 and len(alpha[1]) < 3:
            return ""
        if len(alpha) == 1 and len(alpha[0]) < 3:
            return ""
        return collapse_spaces(" ".join(alpha[:3]))

    if field_name == "type":
        if up in {"KLET", "MOBIL", "OTOMOBIL", "MOTOSIKLET", "KAMYONET", "MINIBUS", "TIP", "TIPI", "TPI", "TYPE"}:
            # These are typically generic types or cut-off fragments from 'vehicle_class' row. 
            # They shouldn't be the 'commercial type / model'.
            return ""
        
        toks = re.findall(r"[A-Z0-9.-]+", up)
        toks = [
            t
            for t in toks
            if t not in VEHICLE_TYPE_BLOCK
            and t not in VEHICLE_TYPE_NOISE
            and t not in {"TIP", "TIPI", "TPI", "TYPE"}
            and not re.fullmatch(r"D\.?\d+", t)
        ]
        if not toks:
            return ""
        meaningful = [t for t in toks if t not in GENERIC_VEHICLE_TYPE_TOKENS]
        if meaningful:
            toks = meaningful
        else:
            return ""
        # keep richer D.2 phrases and also short alnum codes such as 3C / UMG / LT019 / 0403 SHD.
        if len(toks) >= 2:
            return collapse_spaces(" ".join(toks[:6]))
        return toks[0]

    if field_name in {"engine_no", "chassis_no"}:
        compact = re.sub(r"[^A-Z0-9]", "", up)
        if len(compact) < 8:
            return ""
        if field_name == "chassis_no" and len(compact) < 12:
            return ""
        # VIN-like cleanup for chassis (usually 17 chars): I/O/Q are not valid VIN chars.
        if field_name == "chassis_no" and len(compact) >= 16:
            chars = list(compact)
            for idx, ch in enumerate(chars):
                if ch in {"O", "Q"}:
                    chars[idx] = "0"
                elif ch == "I" and idx >= 3:
                    chars[idx] = "1"
            compact = "".join(chars)
        elif field_name == "engine_no":
            # Engine numbers vary by maker; conservative ambiguity cleanup.
            chars = list(compact)
            for idx, ch in enumerate(chars):
                if ch in {"O", "Q"} and idx > 0 and idx < len(chars) - 1:
                    left = chars[idx - 1]
                    right = chars[idx + 1]
                    if left.isdigit() or right.isdigit():
                        chars[idx] = "0"
                elif ch == "I" and idx > 2:
                    left = chars[idx - 1]
                    right = chars[idx + 1] if idx + 1 < len(chars) else ""
                    if left.isdigit() or right.isdigit():
                        chars[idx] = "1"
            compact = "".join(chars)
        if compact in LABEL_NOISE:
            return ""
        return compact

    if field_name == "tax_or_id_no":
        digits = re.sub(r"\D", "", up)
        if len(digits) in {10, 11}:
            return digits
        return ""

    if field_name in {"first_registration_date", "registration_date", "inspection_date"}:
        m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", up)
        if not m:
            return ""
        dd, mm, yy = map(int, m.groups())
        if not (1 <= dd <= 31 and 1 <= mm <= 12 and 1950 <= yy <= 2035):
            return ""
        return up

    return up


def _post_cleanup(cleaned: str, cfg: FieldConfig) -> str:
    if not cleaned:
        return ""

    value = collapse_spaces(cleaned)

    if cfg.strip_prefixes:
        for prefix in cfg.strip_prefixes:
            p = normalize_turkish_ascii(prefix)
            if value.startswith(p):
                value = value[len(p) :].lstrip(" :-/")

    if cfg.min_len > 0 and len(re.sub(r"\s+", "", value)) < cfg.min_len:
        return ""

    if cfg.prefer_mixed_alnum:
        has_alpha = bool(re.search(r"[A-Z]", value))
        has_digit = bool(re.search(r"\d", value))
        if not (has_alpha and has_digit):
            # keep if it's still a likely short code like JT8
            if not re.fullmatch(r"(?:[A-Z]{2,6}\d{0,3}|\d+[A-Z]{1,4}|[A-Z0-9-]{2,12})", value):
                return ""

    return value


_OWNER_ADDRESS_HINTS = re.compile(
    r"\b(?:MAH\.?|MAHALLESI|CAD\.?|CADDESI|SOK\.?|SOKAK|NO\b|TEPEBASI|ESKISEHIR|ADRES(?:I)?)\b"
)
_OWNER_NOISE_HINTS = re.compile(
    r"\b(?:NOTER|VERILDIGI|IL/ILCE|ILCE|TESCIL|TARIHI|VERGI|PLAKA|RENGI|MARKASI|MODEL|SINIFI)\b"
)
_TYPE_BLEED_HINTS = re.compile(r"\b(?:ADRESI|SOYADI|ADI|VERGI|TESCIL)\b")


def _should_run_secondary_ocr(field_name: str, entry: object, min_confidence: int) -> bool:
    if not isinstance(entry, dict):
        return True

    value = str(entry.get("value") or "").strip()
    if not value:
        return True
    if bool(entry.get("low_confidence", False)):
        return True

    score = int(entry.get("confidence_score", 0) or 0)
    up = normalize_turkish_ascii(value)
    if field_name in {"owner_name", "owner_surname"}:
        return bool(
            _OWNER_ADDRESS_HINTS.search(up)
            or _OWNER_NOISE_HINTS.search(up)
            or (field_name == "owner_name" and any(hint in up for hint in OWNER_COMPANY_HINTS))
        )
    if field_name == "type" and _TYPE_BLEED_HINTS.search(up):
        return True
    if field_name == "chassis_no":
        compact = re.sub(r"[^A-Z0-9]", "", up)
        if len(compact) < 17:
            return True
        if any(ch in compact for ch in {"I", "O", "Q"}):
            return True
    if field_name == "engine_no":
        compact = re.sub(r"[^A-Z0-9]", "", up)
        if len(compact) < 8:
            return True
        if compact.endswith(("000", "OOO")):
            return True
    if score < min_confidence:
        return True
    return False


def _expanded_secondary_roi(field_name: str, roi: Rect, doc_shape: tuple[int, int, int]) -> Rect:
    doc_h, doc_w = doc_shape[:2]
    # Keep ROI expansion conservative by default; wide expansions caused
    # neighboring cells to bleed into ROI-VL reads on the ruhsat layout.
    left_pad = max(6, int(doc_w * 0.006))
    right_pad = max(6, int(doc_w * 0.006))
    top_pad = max(5, int(doc_h * 0.005))
    bottom_pad = max(5, int(doc_h * 0.005))

    field_padding: dict[str, tuple[float, float, float, float]] = {
        # Label/value on the same row: include just a bit of label context.
        "type": (0.018, 0.012, 0.006, 0.010),
        "owner_name": (0.018, 0.010, 0.004, 0.008),
        "owner_surname": (0.018, 0.010, 0.004, 0.008),
        "owner_title": (0.018, 0.010, 0.004, 0.008),
        "owner_address": (0.014, 0.012, 0.006, 0.010),
        "province_district": (0.010, 0.010, 0.004, 0.008),
        # Value boxes stacked under labels: do not pull in the next row.
        "vehicle_type": (0.008, 0.008, 0.004, 0.008),
        "engine_no": (0.006, 0.006, 0.004, 0.006),
        "chassis_no": (0.006, 0.006, 0.004, 0.006),
        "registration_serial_no": (0.008, 0.008, 0.004, 0.006),
        "approval_type_no": (0.008, 0.008, 0.004, 0.006),
        "usage_purpose": (0.010, 0.010, 0.004, 0.008),
        "serial_no": (0.008, 0.008, 0.004, 0.006),
    }
    if field_name in field_padding:
        lp, rp, tp, bp = field_padding[field_name]
        left_pad = max(left_pad, int(doc_w * lp))
        right_pad = max(right_pad, int(doc_w * rp))
        top_pad = max(top_pad, int(doc_h * tp))
        bottom_pad = max(bottom_pad, int(doc_h * bp))

    expanded = Rect(
        x=max(0, roi.x - left_pad),
        y=max(0, roi.y - top_pad),
        w=roi.w + left_pad + right_pad,
        h=roi.h + top_pad + bottom_pad,
    )
    return expanded.clip(doc_w, doc_h)


def apply_secondary_ocr_fallback(
    document: np.ndarray,
    rois: dict[str, Rect],
    fields: dict[str, dict[str, object]],
    field_configs: dict[str, FieldConfig],
    engine: TextReadEngine | None,
    allowed_fields: tuple[str, ...],
    force_fields: tuple[str, ...],
    min_confidence: int,
    method_name: str = "glm_roi_fallback",
    extra_read_kwargs: dict[str, object] | None = None,
) -> None:
    if engine is None or not allowed_fields:
        return

    forced = {str(name).strip() for name in force_fields}
    kwargs_key = tuple(sorted((str(k), str(v)) for k, v in (extra_read_kwargs or {}).items()))
    roi_cache: dict[tuple[int, int, int, int, int | None, str | None, tuple[tuple[str, str], ...]], str] = {}

    def quality_score(field_name: str, value: str, base_score: int) -> int:
        normalized = normalize_turkish_ascii(value)
        compact = re.sub(r"[^A-Z0-9]", "", normalize_turkish_ascii(value))
        score = int(base_score)
        if field_name == "chassis_no":
            if len(compact) == 17:
                score += 30
            elif len(compact) >= 15:
                score += 10
            if VIN_PATTERN.fullmatch(compact):
                score += 25
            if any(ch in compact for ch in {"I", "O", "Q"}):
                score -= 20
            return score
        if field_name == "engine_no":
            if 8 <= len(compact) <= 20:
                score += 15
            elif len(compact) >= 6:
                score += 6
            if re.fullmatch(r"[A-Z0-9]{6,20}", compact):
                score += 12
            if re.search(r"[A-Z]", compact) and re.search(r"\d", compact):
                score += 8
            return score
        if field_name == "type":
            if _TYPE_BLEED_HINTS.search(normalized):
                score -= 28
            if re.fullmatch(r"[A-Z0-9-]{2,12}", compact):
                score += 18
            if re.search(r"(POLO|TDI|PS|COMFORTLINE|SYMBOL|AUTH|DCI|\d\.\d)", normalized):
                score += 18
            if re.search(r"\b(?:NOTER|VERGI|ILCE|ADRESI|MAH|CAD|SOK|NO|TESCIL)\b", normalized):
                score -= 18
            if len(compact) < 2:
                score -= 20
            return score
        if field_name == "vehicle_type":
            if re.fullmatch(r"[A-Z0-9-]{2,12}", compact):
                score += 20
            if re.search(r"\b(?:OTOMOBIL|MOTOSIKLET|KAMYONET|MINIBUS|KAMYON|TRAKTOR)\b", normalized):
                score += 12
            if re.search(r"(YENI SYMBOL|PALIO|POLO|AUTH|DCI)", normalized):
                score += 10
            if re.search(r"\b20\d{2}\b", normalized):
                score -= 10
            if re.search(r"\b(?:NOTER|VERGI|ILCE|ADRESI|MAH|CAD|SOK|NO|TESCIL)\b", normalized):
                score -= 24
            if _TYPE_BLEED_HINTS.search(normalized):
                score -= 20
            return score
        return score

    for field_name in allowed_fields:
        cfg = field_configs.get(field_name)
        roi = rois.get(field_name)
        if cfg is None or roi is None:
            continue

        current = fields.get(field_name)
        if field_name not in forced and not _should_run_secondary_ocr(field_name, current, min_confidence):
            continue

        secondary_roi = _expanded_secondary_roi(field_name, roi, document.shape)
        cache_key = (
            secondary_roi.x,
            secondary_roi.y,
            secondary_roi.w,
            secondary_roi.h,
            cfg.psm,
            cfg.whitelist,
            kwargs_key,
        )
        cached = roi_cache.get(cache_key)
        if cached is None:
            patch = crop(document, secondary_roi)
            if extra_read_kwargs:
                try:
                    cached = collapse_spaces(engine.read_text(patch, psm=cfg.psm, whitelist=cfg.whitelist, **extra_read_kwargs))
                except TypeError:
                    cached = collapse_spaces(engine.read_text(patch, psm=cfg.psm, whitelist=cfg.whitelist))
            else:
                cached = collapse_spaces(engine.read_text(patch, psm=cfg.psm, whitelist=cfg.whitelist))
            roi_cache[cache_key] = cached
        raw = cached

        # If the GLM model (via explicit prompt) tells us this is an empty patch or noise,
        # we should confidently wipe out any low-confidence noise PaddleOCR found earlier.
        if not raw or "EMPTY_PATCH" in raw:
            entry = dict(current) if isinstance(current, dict) else {}
            entry["value"] = ""
            entry["raw"] = raw if "EMPTY_PATCH" not in raw else ""
            entry["method"] = method_name
            entry["confidence_score"] = 100  # Confidently empty
            entry["low_confidence"] = False
            entry["secondary_ocr_applied"] = True
            fields[field_name] = entry
            continue

        cleaned = cleanup_text(raw, cfg.cleanup)
        cleaned = _post_cleanup(cleaned, cfg)
        cleaned = _post_field_filters(field_name, cleaned)
        if not cleaned:
            continue

        score = _score_cleaned(cleaned, cfg.cleanup)
        current_value = ""
        current_score = 0
        if isinstance(current, dict):
            current_value = str(current.get("value") or "")
            current_score = int(current.get("confidence_score", 0) or 0)

        if field_name in {"engine_no", "chassis_no", "type", "vehicle_type"} and current_value:
            current_quality = quality_score(field_name, current_value, current_score)
            secondary_quality = quality_score(field_name, cleaned, score)
            if secondary_quality + 4 < current_quality:
                continue

        entry = dict(current) if isinstance(current, dict) else {}
        entry["value"] = cleaned
        entry["raw"] = raw
        entry["method"] = method_name
        entry["confidence_score"] = max(score, int(entry.get("confidence_score", 0) or 0))
        entry["low_confidence"] = bool(cfg.confidence_threshold > 0 and score < cfg.confidence_threshold)
        entry["roi"] = roi.to_dict()
        entry["value_bbox"] = roi.to_dict()
        entry["secondary_roi"] = secondary_roi.to_dict()
        entry["secondary_ocr_applied"] = True
        fields[field_name] = entry


def _extract_plate_from_words(
    field_name: str,
    cfg: FieldConfig,
    roi: Rect,
    page_words: list[object],
    doc_shape: tuple[int, int, int],
) -> dict[str, object] | None:
    if field_name != "plate":
        return None
    if not page_words:
        return None

    doc_h, doc_w = doc_shape[:2]

    # Plate is usually near top area, but avoid scanning too far right where many
    # false plate-like strings appear in other fields.
    y_cap = min(doc_h, max(roi.y + 2 * roi.h, int(doc_h * 0.46)))
    x_cap = int(doc_w * 0.68)
    scoped_words = [
        w
        for w in page_words
        if (w.bbox.y + w.bbox.h / 2) <= y_cap
        and (w.bbox.x + w.bbox.w / 2) <= x_cap
        and float(getattr(w, "conf", 0.0)) >= 10.0
    ]
    if not scoped_words:
        return None

    by_line: dict[tuple[int, int, int], list[object]] = {}
    for w in scoped_words:
        key = (int(w.block_num), int(w.par_num), int(w.line_num))
        by_line.setdefault(key, []).append(w)

    roi_cx = roi.x + roi.w / 2.0
    roi_cy = roi.y + roi.h / 2.0

    best: dict[str, object] | None = None

    for line_words in by_line.values():
        ordered = sorted(line_words, key=lambda ww: ww.bbox.x)
        tokens: list[tuple[str, Rect]] = []

        for w in ordered:
            up = normalize_turkish_ascii(str(w.text or ""))
            parts = [re.sub(r"[^A-Z0-9]", "", p) for p in re.findall(r"[A-Z0-9]+", up)]
            for p in parts:
                if p:
                    tokens.append((p, w.bbox))

        if not tokens:
            continue

        for i in range(len(tokens)):
            combined = ""
            boxes: list[Rect] = []
            for j in range(i, min(i + 4, len(tokens))):
                tok, box = tokens[j]
                combined += tok
                boxes.append(box)

                cleaned = cleanup_text(combined, "plate")
                cleaned = _post_cleanup(cleaned, cfg)
                cleaned = _post_field_filters("plate", cleaned)
                if not cleaned:
                    continue

                cand_bbox = union_rects(boxes).clip(doc_w, doc_h)
                cx = cand_bbox.x + cand_bbox.w / 2.0
                cy = cand_bbox.y + cand_bbox.h / 2.0
                proximity_penalty = int(
                    round(
                        10
                        * (
                            abs(cx - roi_cx) / max(1, doc_w)
                            + abs(cy - roi_cy) / max(1, doc_h)
                        )
                    )
                )
                score = _score_cleaned(cleaned, cfg.cleanup) + 12 - proximity_penalty

                # Penalize suspicious alnum tails that often come from vehicle-type
                # strings (e.g., DCI65 / 15OCI65), which look like plates by regex
                # but are semantically wrong.
                if re.search(r"(?:DCI|TDI|HDI|CDI|VVT|TSI|OIL|DIZEL|DIESEL)\d{1,3}$", cleaned):
                    score -= 12
                if re.search(r"[A-Z]{3,}\d{3,4}$", cleaned) and not re.match(r"^\d{2}", cleaned):
                    score -= 6

                cand = {
                    "raw": combined,
                    "cleaned": cleaned,
                    "score": score,
                    "bbox": cand_bbox,
                    "method": "semantic_plate_page_words",
                    "priority": 6,
                }

                if best is None:
                    best = cand
                else:
                    if int(cand["score"]) > int(best["score"]):
                        best = cand
                    elif int(cand["score"]) == int(best["score"]):
                        by = int(best["bbox"].y)
                        bx = int(best["bbox"].x)
                        cy_ = int(cand["bbox"].y)
                        cx_ = int(cand["bbox"].x)
                        if (cy_, cx_) < (by, bx):
                            best = cand

    return best


def _extract_owner_from_words(
    field_name: str,
    cfg: FieldConfig,
    roi: Rect,
    page_words: list[object],
    doc_shape: tuple[int, int, int],
) -> dict[str, object] | None:
    if field_name not in {"owner_name", "owner_surname", "owner_title"}:
        return None
    if not page_words:
        return None

    doc_h, doc_w = doc_shape[:2]
    sx = roi.x
    sy = roi.y
    ex = roi.x + roi.w
    ey = roi.y + roi.h

    region_words = [
        w
        for w in page_words
        if (w.bbox.x >= sx and w.bbox.x + w.bbox.w <= ex and w.bbox.y >= sy and w.bbox.y + w.bbox.h <= ey)
    ]
    if not region_words:
        return None

    region_words = sorted(region_words, key=lambda w: (w.bbox.y, w.bbox.x))
    text = collapse_spaces(" ".join(str(w.text) for w in region_words))
    cleaned = cleanup_text(text, cfg.cleanup)
    cleaned = _post_cleanup(cleaned, cfg)
    cleaned = _post_field_filters(field_name, cleaned)
    if not cleaned:
        return None

    candidate_boxes = [w.bbox for w in region_words]
    return {
        "raw": text,
        "cleaned": cleaned,
        "score": _score_cleaned(cleaned, cfg.cleanup) + 6,
        "bbox": union_rects(candidate_boxes).clip(doc_w, doc_h),
        "method": "semantic_owner_roi_words",
        "priority": 5,
    }


def _extract_vehicle_type_from_words(
    field_name: str,
    cfg: FieldConfig,
    roi: Rect,
    page_words: list[object],
    doc_shape: tuple[int, int, int],
) -> dict[str, object] | None:
    if field_name not in {"type", "vehicle_type"}:
        return None
    if not page_words:
        return None

    doc_h, doc_w = doc_shape[:2]
    sx = roi.x
    sy = roi.y
    ex = roi.x + roi.w
    ey = roi.y + roi.h

    def inside_strict(word: object) -> bool:
        return (
            word.bbox.x >= sx
            and word.bbox.x + word.bbox.w <= ex
            and word.bbox.y >= sy
            and word.bbox.y + word.bbox.h <= ey
        )

    # D.2/D.5 values often sit slightly to the right of the configured ROI or
    # share the same row with the label. Expand horizontally, but keep the scan
    # tightly banded vertically so we don't wander into owner/date rows.
    scan_x_min = sx
    scan_x_max = min(doc_w, ex + max(220, int(doc_w * 0.12)))
    scan_y_min = max(0, sy - max(18, int(doc_h * 0.01)))
    scan_y_max = min(doc_h, ey + max(36, int(doc_h * 0.03)))

    strict_words = [w for w in page_words if inside_strict(w)]
    band_words = [
        w
        for w in page_words
        if (
            scan_x_min <= (w.bbox.x + w.bbox.w / 2.0) <= scan_x_max
            and scan_y_min <= (w.bbox.y + w.bbox.h / 2.0) <= scan_y_max
        )
    ]
    words = strict_words or band_words
    if not words:
        return None

    def token_looks_like_type_code(token: str) -> bool:
        compact = re.sub(r"[^A-Z0-9-]", "", token)
        pure = re.sub(r"[^A-Z0-9]", "", compact)
        if not pure or len(pure) < 2 or len(pure) > 10:
            return False
        if pure in VEHICLE_TYPE_BLOCK or pure in LABEL_NOISE or pure in VEHICLE_TYPE_NOISE:
            return False
        if pure in GENERIC_VEHICLE_TYPE_TOKENS:
            return False
        if PLATE_PATTERN.fullmatch(pure) or VIN_PATTERN.fullmatch(pure):
            return False
        if YEAR_PATTERN.fullmatch(pure) or TAX_ID_PATTERN.fullmatch(pure):
            return False
        if pure.isdigit():
            return False
        if pure.startswith(("YIL", "VIL", "MOD")):
            return False
        if re.fullmatch(r"[A-Z]{2,4}", pure):
            return True
        if "-" in compact and re.search(r"[A-Z]", compact) and re.search(r"\d", compact):
            return True
        if re.fullmatch(r"[A-Z]{1,5}\d{1,5}", pure):
            return True
        return False

    line_groups: list[list[object]] = []
    line_tol = max(14, int(roi.h * 0.8))
    for word in sorted(words, key=lambda ww: (ww.bbox.y, ww.bbox.x)):
        cy = word.bbox.y + word.bbox.h / 2.0
        if not line_groups:
            line_groups.append([word])
            continue
        prev = line_groups[-1][-1]
        prev_cy = prev.bbox.y + prev.bbox.h / 2.0
        if abs(cy - prev_cy) <= line_tol:
            line_groups[-1].append(word)
        else:
            line_groups.append([word])

    roi_cy = sy + roi.h / 2.0
    sorted_groups = sorted(
        line_groups,
        key=lambda group: abs((sum(w.bbox.y + w.bbox.h / 2.0 for w in group) / len(group)) - roi_cy),
    )

    for group in sorted_groups:
        tokens: list[tuple[str, Rect]] = []
        for w in sorted(group, key=lambda ww: ww.bbox.x):
            up = normalize_turkish_ascii(str(w.text or ""))
            for tok in re.findall(r"[A-Z0-9.-]+", up):
                if tok in VEHICLE_TYPE_BLOCK or tok in LABEL_NOISE or len(tok) <= 1:
                    continue
                if not token_looks_like_type_code(tok):
                    continue
                tokens.append((tok, w.bbox))

        if not tokens:
            continue

        preferred_tokens = [
            (t, b)
            for t, b in tokens
            if "-" in t or (re.search(r"[A-Z]", t) and re.search(r"\d", t))
        ]
        if preferred_tokens:
            phrase_tokens = [preferred_tokens[0][0]]
            phrase_boxes = [preferred_tokens[0][1]]
        else:
            phrase_tokens = [tokens[0][0]]
            phrase_boxes = [tokens[0][1]]
        cleaned = collapse_spaces(" ".join(phrase_tokens))
        cleaned = _post_cleanup(cleaned, cfg)
        cleaned = _post_field_filters(field_name, cleaned)
        if not cleaned:
            continue

        return {
            "raw": cleaned,
            "cleaned": cleaned,
            "score": _score_cleaned(cleaned, cfg.cleanup) + 8,
            "bbox": union_rects(phrase_boxes).clip(doc_w, doc_h),
            "method": "semantic_vehicle_type_roi_words",
            "priority": 5,
        }

    return None


def _extract_date_from_words(
    field_name: str,
    cfg: FieldConfig,
    roi: Rect,
    page_words: list[object],
    doc_shape: tuple[int, int, int],
) -> dict[str, object] | None:
    if field_name not in {"first_registration_date", "registration_date", "inspection_date"}:
        return None
    if not page_words:
        return None

    doc_h, doc_w = doc_shape[:2]
    scan_x_min = max(0, roi.x - max(24, int(doc_w * 0.01)))
    scan_x_max = min(doc_w, roi.x + roi.w + max(220, int(doc_w * 0.14)))
    scan_y_min = max(0, roi.y - max(18, int(doc_h * 0.015)))
    scan_y_max = min(doc_h, roi.y + roi.h + max(70, int(doc_h * 0.07)))

    words = [
        w
        for w in page_words
        if (
            scan_x_min <= (w.bbox.x + w.bbox.w / 2.0) <= scan_x_max
            and scan_y_min <= (w.bbox.y + w.bbox.h / 2.0) <= scan_y_max
        )
    ]
    if not words:
        return None

    ordered = sorted(words, key=lambda ww: (ww.bbox.y, ww.bbox.x))
    text = normalize_turkish_ascii(collapse_spaces(" ".join(str(w.text or "") for w in ordered)))
    matches = list(DATE_PATTERN.finditer(text))
    if not matches:
        return None

    for match in matches:
        candidate = cleanup_text(match.group(0), cfg.cleanup)
        candidate = _post_cleanup(candidate, cfg)
        candidate = _post_field_filters(field_name, candidate)
        if not candidate:
            continue
        return {
            "raw": match.group(0),
            "cleaned": candidate,
            "score": _score_cleaned(candidate, cfg.cleanup) + 12,
            "bbox": union_rects([w.bbox for w in ordered]).clip(doc_w, doc_h),
            "method": "semantic_date_roi_words",
            "priority": 5,
        }

    return None


def _extract_engine_from_words(
    field_name: str,
    cfg: FieldConfig,
    roi: Rect,
    page_words: list[object],
    doc_shape: tuple[int, int, int],
) -> dict[str, object] | None:
    if field_name != "engine_no":
        return None
    if not page_words:
        return None

    doc_h, doc_w = doc_shape[:2]
    scan_x_min = max(0, roi.x - max(32, int(doc_w * 0.02)))
    scan_x_max = min(doc_w, roi.x + roi.w + max(180, int(doc_w * 0.10)))
    scan_y_min = max(0, roi.y - max(18, int(doc_h * 0.015)))
    scan_y_max = min(doc_h, roi.y + roi.h + max(55, int(doc_h * 0.05)))

    words = [
        w
        for w in page_words
        if (
            scan_x_min <= (w.bbox.x + w.bbox.w / 2.0) <= scan_x_max
            and scan_y_min <= (w.bbox.y + w.bbox.h / 2.0) <= scan_y_max
        )
    ]
    if not words:
        return None

    bad_fragments = ("TICAR", "ADI", "TESCIL", "MARKA", "MODEL", "SASE", "PANEL", "ROMORK")
    best_token = ""
    best_box: Rect | None = None
    best_score = -1

    for w in sorted(words, key=lambda ww: (ww.bbox.y, ww.bbox.x)):
        up = normalize_turkish_ascii(str(w.text or ""))
        for tok in re.findall(r"[A-Z0-9]+", up):
            if len(tok) < 6 or len(tok) > 18:
                continue
            if tok in LABEL_NOISE or tok in VEHICLE_TYPE_BLOCK:
                continue
            if any(fragment in tok for fragment in bad_fragments):
                continue
            if not (re.search(r"[A-Z]", tok) and re.search(r"\d", tok)):
                continue
            if PLATE_PATTERN.fullmatch(tok) or VIN_PATTERN.fullmatch(tok) or TAX_ID_PATTERN.fullmatch(tok):
                continue
            score = len(tok)
            if tok[0].isalpha():
                score += 2
            if re.search(r"[A-Z]{1,4}\d{3,}", tok):
                score += 4
            if score > best_score:
                best_score = score
                best_token = tok
                best_box = w.bbox

    if not best_token or best_box is None:
        return None

    cleaned = cleanup_text(best_token, cfg.cleanup)
    cleaned = _post_cleanup(cleaned, cfg)
    cleaned = _post_field_filters(field_name, cleaned)
    if not cleaned:
        return None

    return {
        "raw": best_token,
        "cleaned": cleaned,
        "score": _score_cleaned(cleaned, cfg.cleanup) + 12,
        "bbox": best_box.clip(doc_w, doc_h),
        "method": "semantic_engine_roi_words",
        "priority": 5,
    }


def _extract_serial_from_text(
    field_name: str,
    cfg: FieldConfig,
    roi: Rect,
    page_words: list[object],
    doc_shape: tuple[int, int, int],
) -> dict[str, object] | None:
    if field_name != "serial_no":
        return None

    doc_h, doc_w = doc_shape[:2]
    sx = roi.x
    sy = roi.y
    ex = roi.x + roi.w
    ey = roi.y + roi.h
    words = [
        w
        for w in page_words
        if (w.bbox.x >= sx and w.bbox.x + w.bbox.w <= ex and w.bbox.y >= sy and w.bbox.y + w.bbox.h <= ey)
    ]

    if not words:
        return None

    ordered = sorted(words, key=lambda ww: (ww.bbox.y, ww.bbox.x))
    text = normalize_turkish_ascii(collapse_spaces(" ".join(str(w.text) for w in ordered)))

    patterns = [
        r"\b([A-Z]{1,3}\s*(?:NO|N|NUMARA|NR|№)?\s*[:.]?\s*\d{3,7})\b",
        r"\b([A-Z]{1,3}\s*\d{3,7})\b",
    ]
    hit = ""
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            hit = collapse_spaces(m.group(1))
            break

    if not hit:
        return None

    cleaned = cleanup_text(hit, cfg.cleanup)
    cleaned = _post_cleanup(cleaned, cfg)
    cleaned = _post_field_filters(field_name, cleaned)
    if not cleaned:
        return None

    return {
        "raw": hit,
        "cleaned": cleaned,
        "score": _score_cleaned(cleaned, cfg.cleanup) + 8,
        "bbox": union_rects([w.bbox for w in ordered]).clip(doc_w, doc_h),
        "method": "semantic_serial_roi_words",
        "priority": 5,
    }


def _extract_structured_candidates_from_page_text(
    field_configs: dict[str, FieldConfig],
    page_words: list[object],
    anchor_matches: dict[str, AnchorMatch],
    enabled: bool,
) -> dict[str, dict[str, str]]:
    if not enabled:
        return {}
    if not page_words:
        return {}

    by_line: dict[tuple[int, int, int], list[object]] = {}
    for w in page_words:
        key = (
            int(getattr(w, "block_num", 0)),
            int(getattr(w, "par_num", 0)),
            int(getattr(w, "line_num", 0)),
        )
        by_line.setdefault(key, []).append(w)

    def line_sort_key(words: list[object]) -> tuple[int, int]:
        min_y = min((int(w.bbox.y) for w in words), default=0)
        min_x = min((int(w.bbox.x) for w in words), default=0)
        return min_y, min_x

    lines = sorted(by_line.values(), key=line_sort_key)
    line_entries: list[dict[str, object]] = []
    for line_words in lines:
        ordered = sorted(line_words, key=lambda ww: int(ww.bbox.x))
        txt = collapse_spaces(" ".join(normalize_turkish_ascii(str(getattr(w, "text", "") or "")) for w in ordered))
        if txt:
            line_entries.append(
                {
                    "text": txt,
                    "x": min((int(w.bbox.x) for w in ordered), default=0),
                    "y": min((int(w.bbox.y) for w in ordered), default=0),
                }
            )

    line_texts = [str(entry["text"]) for entry in line_entries]

    full_text = "\n".join(line_texts)
    full_text = full_text.replace("О", "O").replace("І", "I")
    if not full_text:
        return {}

    out: dict[str, dict[str, str]] = {}

    inline_label_specs: dict[str, tuple[str, ...]] = {
        "province_district": ("VERILDIGI IL/ILCE", "VERILDIGIILILCE"),
        "registration_serial_no": ("TESCIL SIRA NO", "TESCILSIRANO"),
        "type": ("TIPI",),
        "brand": ("MARKASI", "MARKA"),
        "first_registration_date": ("ILK TESCIL TARIHI", "ILK TESCIL TAR", "ILKTESCILTARIHI"),
        "registration_date": ("TESCIL TARIHI", "TESCIL TAR", "TESCILTARIHI"),
        "inspection_date": ("MUA GEC TRH", "MUA.GEC.TRH", "MUA GEC", "MUAYENE"),
        "vehicle_class": ("ARAC SINIFI", "ARACSINIFI"),
        "vehicle_type": ("CINSI",),
        "color": ("RENGI",),
        "engine_no": ("MOTOR NO", "MOTORNO", "MOTOR N0"),
        "chassis_no": ("SASE NO", "SASENO", "ESASENO"),
        "tax_or_id_no": ("T C KIMLIK NO", "TC KIMLIK NO", "VERGI NO", "TCKIMLIKNO", "VERGINO"),
        "owner_title": ("SOYADI TICARI UNVANI", "SOYADI", "TICARI UNVANI"),
        "owner_surname": ("SOYADI TICARI UNVANI", "SOYADI", "TICARI UNVANI"),
        "owner_name": ("ADI",),
        "owner_address": ("ADRESI",),
        "approval_type_no": ("TIP ONAY NO", "TIPONAYNO"),
        "approver_registration_no": ("ONAYLAYAN SICIL-IMZA", "ONAYLAYAN SICIL IMZA", "ONAYLAYANSICILIMZA"),
    }

    label_field_specs: dict[str, tuple[tuple[str, ...], int]] = {
        "province_district": (("VERILDIGI IL/ILCE", "VERILDIGIILILCE"), 1),
        "registration_serial_no": (("TESCIL SIRA NO", "TESCILSIRANO"), 1),
        "type": (("TIPI",), 2),
        "owner_title": (("SOYADI TICARI UNVANI", "SOYADITICARIUNVANI"), 1),
        "owner_surname": (("SOYADI TICARI UNVANI", "SOYADITICARIUNVANI"), 1),
        "owner_name": (("ADI",), 1),
        "owner_address": (("ADRESI",), 1),
        "vehicle_type": (("CINSI",), 2),
        "color": (("RENGI",), 1),
        "net_weight_kg": (("NET AGIRLIGI", "NETAGIRLIGI"), 1),
        "max_loaded_weight_kg": (("AZAMI YUKLU AGIRLIGI", "AZAMIYUKLUAGIRLIGI"), 1),
        "trailer_weight_kg": (("ROMORK AZAMI YUKLU AGIRLIGI", "ROMORKAZAMIYUKLUAGIRLIGI", "ROMORKAZAMIYUKLU"), 1),
        "seat_count": (("KOLTUK SAYISI", "KOLTUKSAYISI"), 1),
        "standing_passenger_count": (("AYAKTA YOLCU SAYISI", "AYAKTAYOLCUSAYISI"), 1),
        "cylinder_volume_cm3": (("SILINDIR HACMI", "SILINDIRHACMI"), 1),
        "engine_power_kw": (("MOTOR GUCU", "MOTORGUCU"), 1),
        "fuel_type": (("YAKIT CINSI", "YAKITCINSI"), 1),
        "usage_purpose": (("KULLANIM AMACI", "KULLANIMAMACI"), 1),
        "approval_type_no": (("TIP ONAY NO", "TIPONAYNO"), 1),
        "approver_registration_no": (("ONAYLAYAN SICIL-IMZA", "ONAYLAYAN SICIL IMZA", "ONAYLAYANSICILIMZA"), 1),
    }
    label_alias_compacts = {
        re.sub(r"[^A-Z0-9]", "", alias)
        for aliases, _wanted in label_field_specs.values()
        for alias in aliases
    }

    def maybe_add(field_name: str, raw: str, method: str) -> None:
        if field_name in out:
            return
        cfg = field_configs.get(field_name)
        if cfg is None:
            return
        # Safe mode: only use global regex fallback when anchor is not detected.
        if cfg.anchor and cfg.anchor in anchor_matches and not method.startswith("page_label_"):
            return

        cleaned = cleanup_text(raw, cfg.cleanup)
        cleaned = _post_cleanup(cleaned, cfg)
        cleaned = _post_field_filters(field_name, cleaned)
        if not cleaned:
            return
        out[field_name] = {
            "raw": raw,
            "cleaned": cleaned,
            "method": method,
        }

    plate_match = PLATE_TEXT_PATTERN.search(full_text)
    if plate_match:
        maybe_add("plate", plate_match.group(0), "page_regex_plate")

    for vin_m in VIN_PATTERN.finditer(full_text):
        vin = vin_m.group(0)
        if re.search(r"[A-Z]", vin) and re.search(r"\d", vin):
            maybe_add("chassis_no", vin, "page_regex_vin")
            break

    dates: list[str] = []
    seen_dates: set[str] = set()
    for m in DATE_PATTERN.finditer(full_text):
        try:
            dd = int(m.group(1))
            mm = int(m.group(2))
            yy_raw = m.group(3)
            yy = int(f"20{yy_raw}" if len(yy_raw) == 2 else yy_raw)
        except ValueError:
            continue
        if not (1 <= dd <= 31 and 1 <= mm <= 12 and 1950 <= yy <= 2035):
            continue
        norm = f"{dd:02d}/{mm:02d}/{yy:04d}"
        if norm in seen_dates:
            continue
        seen_dates.add(norm)
        dates.append(norm)

    for field_name, date_value in zip(
        ("first_registration_date", "registration_date", "inspection_date"),
        dates[:3],
    ):
        maybe_add(field_name, date_value, "page_regex_date_sequence")

    years = [int(m.group(1)) for m in YEAR_PATTERN.finditer(full_text)]
    if years:
        maybe_add("model_year", str(min(years)), "page_regex_model_year")

    tax_hits = [m.group(0) for m in TAX_ID_PATTERN.finditer(full_text)]
    if tax_hits:
        maybe_add("tax_or_id_no", tax_hits[0], "page_regex_tax_or_id")

    label_regex_specs: dict[str, tuple[str, ...]] = {
        "engine_no": (
            r"MOTOR\s*N[O0]\s*([A-Z0-9]{8,25})",
            r"P\.?5\s*MOTOR\s*N[O0]\s*([A-Z0-9]{8,25})",
        ),
        "chassis_no": (
            r"(?:SASE\s*N[O0]|SASENO|ESASENO)\s*([A-Z0-9]{15,18})",
            r"E\s*(?:SASE\s*N[O0]|SASENO)\s*([A-Z0-9]{15,18})",
        ),
        "registration_serial_no": (
            r"TESCIL\s*SIRA\s*N[O0]\s*(\d{10,})",
        ),
        "first_registration_date": (
            r"ILK\s*TESCIL\s*TARIHI\s*([0-9./-]{8,10})",
        ),
        "registration_date": (
            r"TESCIL\s*TARIHI\s*([0-9./-]{8,10})",
        ),
        "inspection_date": (
            r"(?:MUA\.?GEC\.?TRH|MUA\s*GEC\s*TRH|MUAYENE)\s*[:.]?\s*([0-9./-]{8,10})",
        ),
        "type": (
            r"TIPI(?:\s+|\n+)([A-Z0-9-]{2,12})",
            r"TIPI(?:\s+|\n+)([A-Z0-9 ./-]{4,40})",
        ),
        "brand": (
            r"MARKASI\s*([A-Z][A-Z0-9 -]{1,30})",
        ),
        "vehicle_class": (
            r"ARAC\s*SINIFI\s*(L[1-7]|M[1-3]|N[1-3]|O[1-4])",
        ),
        "color": (
            r"RENGI\s*(BEYAZ|SIYAH|GRI|GUMUS|MAVI|KIRMIZI|YESIL|SARI)",
        ),
        "tax_or_id_no": (
            r"(?:T\.?C\.?\s*KIMLIK\s*N[O0]|VERGI\s*N[O0])\s*([0-9]{10,11})",
        ),
        "owner_surname": (
            r"(?:SOYADI/?\s*TICARI\s*UNVANI|SOYADI)(?:\s+|\n+)([A-Z0-9(). /-]{8,160})",
        ),
        "serial_no": (
            r"BELGE\s*SERI\s*[:.]?\s*([A-Z]{1,3}\s*(?:NO|N|NUMARA|NR|№)?\s*\d{3,7})",
        ),
    }
    for field_name, patterns in label_regex_specs.items():
        if field_name in out:
            continue
        for pattern in patterns:
            match = re.search(pattern, full_text)
            if match:
                maybe_add(field_name, collapse_spaces(match.group(1)), "page_label_regex")
                if field_name in out:
                    break

    if "type" not in out and line_entries:
        upper_lines = [
            entry
            for entry in line_entries
            if 180 <= int(entry["y"]) <= 420
        ]
        standalone_candidates: list[str] = []
        for entry in upper_lines:
            line = normalize_turkish_ascii(collapse_spaces(str(entry["text"])))
            compact_line = re.sub(r"[^A-Z0-9]", "", line)
            if not line or any(alias in compact_line for alias in label_alias_compacts):
                continue
            if any(
                marker in line
                for marker in (
                    "MARKASI",
                    "TICARADI",
                    "TICARI ADI",
                    "MODEL YILI",
                    "ARAC SINIFI",
                    "CINSI",
                    "RENGI",
                    "MOTORNO",
                    "SASENO",
                    "SASE NO",
                )
            ):
                continue

            typeish_tokens = [
                tok
                for tok in re.findall(r"[A-Z0-9.-]+", line)
                if tok not in VEHICLE_TYPE_BLOCK
                and tok not in VEHICLE_TYPE_NOISE
                and tok not in GENERIC_VEHICLE_TYPE_TOKENS
                and tok not in {"TIP", "TIPI", "TPI", "TYPE"}
                and len(tok) >= 2
            ]
            if not typeish_tokens:
                continue

            if any(re.search(r"[A-Z]", tok) and re.search(r"\d", tok) for tok in typeish_tokens):
                standalone_candidates.append(collapse_spaces(" ".join(typeish_tokens[:4])))
                continue

            if len(typeish_tokens) >= 2 and len(collapse_spaces(" ".join(typeish_tokens[:4]))) <= 40:
                standalone_candidates.append(collapse_spaces(" ".join(typeish_tokens[:4])))

        for candidate in standalone_candidates:
            maybe_add("type", candidate, "page_line_standalone_type")
            if "type" in out:
                break

    def extract_inline_value(field_name: str, line: str, aliases: tuple[str, ...]) -> str:
        up = normalize_turkish_ascii(collapse_spaces(line))
        compact = re.sub(r"[^A-Z0-9*/().:-]", "", up)

        if field_name == "owner_name" and any(marker in compact for marker in {"TICARIADI", "NOTERINADI", "NOTERADI"}):
            return ""

        alias_hit = ""
        for alias in aliases:
            alias_norm = normalize_turkish_ascii(alias)
            if alias_norm in up:
                alias_hit = alias_norm
                break
        if not alias_hit:
            return ""

        remainder = up.split(alias_hit, 1)[1].strip(" :.-/")
        if not remainder:
            return ""

        if field_name == "province_district":
            return remainder if "/" in remainder and "NOTER" in remainder else ""

        if field_name == "registration_serial_no":
            hits = re.findall(r"\d{10,}", remainder)
            return max(hits, key=len) if hits else ""

        if field_name == "type":
            # Variant-B documents often carry a short D.2 code (3C, LT019, UMG, CG, CH50T-8A).
            code_hits = re.findall(r"[A-Z0-9-]{2,12}", remainder)
            code_hits = [
                hit for hit in code_hits
                if hit not in VEHICLE_TYPE_BLOCK
                and hit not in VEHICLE_TYPE_NOISE
            ]
            if code_hits:
                return collapse_spaces(" ".join(code_hits[:3]))
            if re.search(r"(POLO|TDI|PS|COMFORTLINE|\d\.\d)", remainder):
                return remainder
            return ""

        if field_name == "brand":
            brand_hits = re.findall(r"[A-Z0-9-]+", remainder)
            brand_hits = [hit for hit in brand_hits if hit not in BRAND_STOPWORDS and hit not in LABEL_NOISE]
            return collapse_spaces(" ".join(brand_hits[:3])) if brand_hits else ""

        if field_name in {"first_registration_date", "registration_date", "inspection_date"}:
            m = DATE_PATTERN.search(remainder)
            if not m:
                return ""
            dd = int(m.group(1))
            mm = int(m.group(2))
            yy_raw = m.group(3)
            yy = int(f"20{yy_raw}" if len(yy_raw) == 2 else yy_raw)
            if not (1 <= dd <= 31 and 1 <= mm <= 12 and 1950 <= yy <= 2035):
                return ""
            return f"{dd:02d}/{mm:02d}/{yy:04d}"

        if field_name == "vehicle_class":
            m = re.search(r"\b(?:L[1-7]|M[1-3]|N[1-3]|O[1-4])\b", remainder)
            return m.group(0) if m else ""

        if field_name == "vehicle_type":
            return remainder if len(remainder) >= 4 else ""

        if field_name == "color":
            for color in ("BEYAZ", "SIYAH", "GRI", "GUMUS", "MAVI", "KIRMIZI", "YESIL", "SARI"):
                if color in remainder:
                    return color
            return ""

        if field_name == "engine_no":
            hits = re.findall(r"[A-Z0-9]{8,25}", re.sub(r"[^A-Z0-9]", " ", remainder))
            return max(hits, key=len) if hits else ""

        if field_name == "chassis_no":
            hits = re.findall(r"[A-Z0-9]{15,18}", re.sub(r"[^A-Z0-9]", " ", remainder))
            return max(hits, key=len) if hits else ""

        if field_name == "tax_or_id_no":
            hits = re.findall(r"\d{10,11}", remainder)
            return hits[0] if hits else ""

        if field_name in {"owner_title", "owner_name", "owner_address"}:
            return remainder

        if field_name == "approval_type_no":
            compact_rest = re.sub(r"[^A-Z0-9*/.\- ]", " ", remainder)
            compact_rest = collapse_spaces(compact_rest)
            if "*" in compact_rest or "/" in compact_rest:
                return compact_rest
            return ""

        if field_name == "approver_registration_no":
            hits = re.findall(r"\d{1,6}", remainder)
            return hits[-1] if hits else ""

        return remainder

    for entry in line_entries:
        line = str(entry["text"])
        for field_name, aliases in inline_label_specs.items():
            if field_name in out:
                continue
            value = extract_inline_value(field_name, line, aliases)
            if value:
                maybe_add(field_name, value, "page_label_inline")

    def is_label_line(line_text: str) -> bool:
        compact = re.sub(r"[^A-Z0-9]", "", line_text)
        if re.match(r"^[A-Z]\d", compact):
            return True
        return any(alias in compact for alias in label_alias_compacts)

    x_windows: dict[str, tuple[int, int]] = {
        "province_district": (-40, 360),
        "registration_serial_no": (-20, 220),
        "type": (-40, 260),
        "owner_title": (100, 240),
        "owner_name": (100, 260),
        "owner_address": (-20, 520),
        "vehicle_type": (-20, 260),
        "color": (80, 220),
        "net_weight_kg": (0, 120),
        "max_loaded_weight_kg": (140, 320),
        "trailer_weight_kg": (140, 320),
        "seat_count": (0, 120),
        "standing_passenger_count": (140, 320),
        "cylinder_volume_cm3": (0, 120),
        "engine_power_kw": (140, 320),
        "fuel_type": (0, 160),
        "usage_purpose": (-30, 360),
        "approval_type_no": (-40, 260),
        "approver_registration_no": (120, 240),
    }
    max_y_gaps: dict[str, int] = {
        "type": 70,
        "vehicle_type": 55,
        "standing_passenger_count": 40,
        "usage_purpose": 60,
        "approval_type_no": 60,
        "approver_registration_no": 60,
    }
    known_colors = ("BEYAZ", "SIYAH", "GRI", "GUMUS", "MAVI", "KIRMIZI", "YESIL", "SARI")
    known_fuels = ("DIZEL", "BENZIN", "ELEKTRIK", "HIBRIT", "LPG", "CNG")
    numeric_fields = {
        "net_weight_kg",
        "max_loaded_weight_kg",
        "trailer_weight_kg",
        "seat_count",
        "standing_passenger_count",
        "cylinder_volume_cm3",
        "engine_power_kw",
        "approver_registration_no",
    }

    def normalize_follow_line(field_name: str, line: str) -> str:
        up = normalize_turkish_ascii(collapse_spaces(line))
        compact = re.sub(r"[^A-Z0-9*/().:-]", "", up)

        if field_name == "registration_serial_no":
            hits = re.findall(r"\d{10,}", up)
            return max(hits, key=len) if hits else ""

        if field_name == "type":
            if re.search(r"(POLO|TDI|PS|COMFORTLINE|\d\.\d)", up):
                return up
            code_hits = re.findall(r"[A-Z0-9-]{2,12}", up)
            code_hits = [
                hit for hit in code_hits
                if hit not in VEHICLE_TYPE_BLOCK
                and hit not in VEHICLE_TYPE_NOISE
                and not re.fullmatch(r"\d{4}", hit)
            ]
            if code_hits:
                return collapse_spaces(" ".join(code_hits[:3]))
            return ""

        if field_name in {"owner_title", "owner_surname"}:
            return up if re.search(r"\b[A-Z]{2,}\b", up) else ""

        if field_name == "owner_name":
            return up if re.search(r"[A-Z]{2,}", up) else ""

        if field_name == "owner_address":
            return up

        if field_name == "vehicle_type":
            if "OTOMOBIL" in up or re.search(r"\bAB\b", up) or "AB" in compact:
                return up
            if len(up) >= 4 and not is_label_line(up):
                return up
            return ""

        if field_name == "color":
            for color in known_colors:
                if color in up:
                    return color
            return ""

        if field_name in numeric_fields:
            nums = re.findall(r"\d+", up)
            if not nums:
                return ""
            if field_name in {"seat_count", "standing_passenger_count", "engine_power_kw", "approver_registration_no"}:
                return nums[-1]
            return max(nums, key=len)

        if field_name == "fuel_type":
            for fuel in known_fuels:
                if fuel in up:
                    return fuel
            return ""

        if field_name == "usage_purpose":
            return up if re.search(r"(YOLCU|YUK|HUSUSI|TICARI)", up) else ""

        if field_name == "approval_type_no":
            return up if "*" in up else ""

        return up

    def gather_following_lines(field_name: str, start_idx: int, wanted: int) -> str:
        values: list[str] = []
        label_x = int(line_entries[start_idx]["x"])
        label_y = int(line_entries[start_idx]["y"])
        rel_min, rel_max = x_windows.get(field_name, (-80, 260))
        max_gap = max_y_gaps.get(field_name, 120)
        scan_count = 0
        for entry in line_entries[start_idx + 1 :]:
            line = str(entry["text"])
            if not line:
                continue
            scan_count += 1
            if scan_count > 14:
                break
            line_x = int(entry["x"])
            line_y = int(entry["y"])
            if line_y - label_y > max_gap:
                break
            if line_x < label_x + rel_min or line_x > label_x + rel_max:
                continue
            compact = re.sub(r"[^A-Z0-9]", "", line)
            if compact in {"AGIRLIGI", "YUKLU", "AZAMI", "HECREKI", "O"}:
                continue
            normalized = normalize_follow_line(field_name, line)
            if not normalized:
                if is_label_line(line):
                    continue
                if re.fullmatch(r"(?:KG\.?|KW/?KG|CM3|KW|---)", normalize_turkish_ascii(line)):
                    continue
                continue
            values.append(normalized)
            if len(values) >= wanted:
                break
        return collapse_spaces(" ".join(values))

    for field_name, (aliases, wanted_lines) in label_field_specs.items():
        if field_name in out:
            continue
        for idx, line in enumerate(line_texts):
            compact_line = re.sub(r"[^A-Z0-9]", "", line)
            if field_name == "owner_name" and any(marker in compact_line for marker in {"TICARIADI", "NOTERINADI", "NOTERADI"}):
                continue
            if any(re.sub(r"[^A-Z0-9]", "", alias) in compact_line for alias in aliases):
                gathered = gather_following_lines(field_name, idx, wanted_lines)
                if gathered:
                    maybe_add(field_name, gathered, "page_label_follow_line")
                break

    return out


def apply_profile_overrides(
    fields: dict[str, dict[str, object]],
    field_configs: dict[str, FieldConfig],
    page_words: list[object],
    anchor_matches: dict[str, AnchorMatch],
    profile_name: str,
) -> list[str]:
    if profile_name not in {"v29_photo_variant_b_candidate", "unknown_or_degraded_photo"}:
        return []
    if not page_words:
        return []

    structured_page_candidates = _extract_structured_candidates_from_page_text(
        field_configs=field_configs,
        page_words=page_words,
        anchor_matches=anchor_matches,
        enabled=True,
    )
    if not structured_page_candidates:
        return []

    preferred_fields = {
        "province_district",
        "registration_serial_no",
        "brand",
        "first_registration_date",
        "registration_date",
        "inspection_date",
        "vehicle_class",
        "color",
        "engine_no",
        "chassis_no",
        "tax_or_id_no",
        "owner_address",
        "approval_type_no",
        "approver_registration_no",
    }
    replace_if_methods = {
        "empty_fallback",
        "postprocess_validation_failed",
        "glm_roi_fallback",
        "roi_tesseract_preprocessed",
        "roi_tesseract_preprocessed_alt",
        "roi_tesseract_raw",
        "roi_line",
    }

    changed_fields: list[str] = []
    for field_name in preferred_fields:
        candidate = structured_page_candidates.get(field_name)
        current = fields.get(field_name)
        if not candidate or not isinstance(current, dict):
            continue

        current_value = str(current.get("value") or "").strip()
        current_method = str(current.get("method") or "")
        current_score = int(current.get("confidence_score", 0) or 0)
        candidate_score = _score_cleaned(str(candidate["cleaned"]), field_configs[field_name].cleanup) + 24
        current_up = normalize_turkish_ascii(current_value)
        candidate_up = normalize_turkish_ascii(str(candidate["cleaned"]))

        def has_company_hint(text: str) -> bool:
            if not text:
                return False
            tokens = re.findall(r"[A-Z0-9.]+", text)
            if any(tok in OWNER_COMPANY_HINTS for tok in tokens):
                return True
            joined = " ".join(tokens)
            return any(fragment in joined for fragment in OWNER_COMPANY_FRAGMENTS)

        def has_owner_noise(text: str) -> bool:
            if not text:
                return False
            return bool(_OWNER_ADDRESS_HINTS.search(text) or _OWNER_NOISE_HINTS.search(text) or "/" in text)

        owner_name_entry = fields.get("owner_name")
        owner_surname_entry = fields.get("owner_surname")
        owner_name_current = normalize_turkish_ascii(
            str(owner_name_entry.get("value") or "") if isinstance(owner_name_entry, dict) else ""
        )
        owner_surname_current = normalize_turkish_ascii(
            str(owner_surname_entry.get("value") or "") if isinstance(owner_surname_entry, dict) else ""
        )

        should_replace = (
            not current_value
            or bool(current.get("low_confidence", False))
            or current_method in replace_if_methods
            or field_name in {"province_district", "owner_address"}
        )
        if field_name == "owner_title":
            should_replace = (
                should_replace
                or current_up in {"ER", "ILERLE"}
                or (not current_value and (candidate_up == "TAP" or has_company_hint(candidate_up)))
            )
        elif field_name == "owner_surname":
            current_tokens = re.findall(r"[A-Z]+", current_up)
            should_replace = (
                should_replace
                or len(current_tokens) == 0
                or (len(current_tokens) == 1 and len(current_tokens[0]) < 3)
                or (has_company_hint(candidate_up) and not has_company_hint(current_up))
                or current_up == owner_name_current
            )
        elif field_name == "owner_name":
            current_tokens = re.findall(r"[A-Z]+", current_up)
            should_replace = (
                should_replace
                or current_up == owner_surname_current
                or has_company_hint(current_up)
                or (len(current_tokens) >= 2 and len(current_tokens[0]) < 3 and len(current_tokens[1]) < 3)
                or (len(current_tokens) == 1 and len(current_tokens[0]) < 3)
            )
        if field_name in {"owner_title", "owner_surname", "owner_name"}:
            if has_owner_noise(candidate_up) and not has_company_hint(candidate_up):
                continue
            if field_name == "owner_name" and has_company_hint(candidate_up):
                continue
        if not should_replace and candidate_score <= current_score:
            continue
        if not should_replace:
            continue

        patched = dict(current)
        patched["value"] = str(candidate["cleaned"])
        patched["raw"] = str(candidate["raw"])
        patched["method"] = str(candidate["method"])
        patched["confidence_score"] = max(candidate_score, current_score)
        patched["low_confidence"] = False
        patched["profile_override_applied"] = profile_name
        fields[field_name] = patched
        changed_fields.append(field_name)

    return changed_fields


def _pick_best_candidate(
    field_name: str,
    cfg: FieldConfig,
    candidates: list[dict[str, object]],
) -> tuple[dict[str, object] | None, str]:
    if cfg.force_method:
        forced = [c for c in candidates if str(c["method"]) == cfg.force_method]
        non_forced = [c for c in candidates if str(c["method"]) != cfg.force_method]
        ordered = sorted(forced, key=lambda c: (int(c["score"]), int(c["priority"])), reverse=True) + sorted(
            non_forced,
            key=lambda c: (int(c["score"]), int(c["priority"])),
            reverse=True,
        )
    else:
        ordered = sorted(candidates, key=lambda c: (int(c["score"]), int(c["priority"])), reverse=True)

    best: dict[str, object] | None = None
    cleaned = ""

    for cand in ordered:
        cand_cleaned = _post_field_filters(field_name, str(cand["cleaned"]))
        if not cand_cleaned:
            continue
        if cfg.confidence_threshold > 0 and int(cand["score"]) < cfg.confidence_threshold:
            continue
        best = cand
        cleaned = cand_cleaned
        break

    if best is None and cfg.confidence_threshold <= 0:
        for cand in ordered:
            cand_cleaned = _post_field_filters(field_name, str(cand["cleaned"]))
            if cand_cleaned:
                best = cand
                cleaned = cand_cleaned
                break

    return best, cleaned


def _should_run_crop_ocr(
    field_name: str,
    cfg: FieldConfig,
    ocr_cfg: OCRConfig,
    best_non_ocr: dict[str, object] | None,
    cleaned_non_ocr: str,
) -> bool:
    mode = str(ocr_cfg.crop_ocr_mode or "always").strip().lower()
    if mode == "disabled":
        return False
    if mode == "always":
        return True

    if not cleaned_non_ocr or best_non_ocr is None:
        return True

    score = int(best_non_ocr.get("score", 0) or 0)
    threshold = int(cfg.confidence_threshold or 0)
    margin = max(0, int(ocr_cfg.crop_ocr_skip_margin or 0))

    # Keep OCR fallback for a few fragile text-heavy fields when page candidates
    # barely pass the threshold.
    if field_name in {"owner_name", "owner_surname", "brand"} and score < threshold + max(2, margin):
        return True

    if threshold > 0:
        return score < (threshold + margin)
    return False


def _is_good_enough_crop_result(field_name: str, cfg: FieldConfig, cleaned: str, score: int) -> bool:
    if not cleaned:
        return False

    compact = re.sub(r"[^A-Z0-9]", "", normalize_turkish_ascii(cleaned))
    threshold = int(cfg.confidence_threshold or 0)

    if field_name == "chassis_no":
        return (
            len(compact) >= 12
            and bool(re.search(r"[A-Z]", compact))
            and bool(re.search(r"\d", compact))
            and score >= max(threshold + 10, 22)
        )
    if field_name == "engine_no":
        return (
            len(compact) >= 6
            and bool(re.search(r"[A-Z]", compact))
            and bool(re.search(r"\d", compact))
            and score >= max(threshold + 8, 16)
        )
    if field_name == "registration_serial_no":
        return len(compact) >= 10 and score >= max(threshold + 8, 18)
    if field_name == "tax_or_id_no":
        return len(re.findall(r"\d", cleaned)) >= 10 and score >= max(threshold + 8, 14)
    if field_name == "document_number":
        return bool(re.fullmatch(r"\d{4,10}", compact)) and score >= max(threshold + 6, 12)
    if field_name == "document_serial":
        return bool(re.fullmatch(r"[A-Z]{1,4}", compact)) and score >= max(threshold + 5, 8)

    if field_name in {
        "owner_name",
        "owner_surname",
        "owner_title",
        "owner_address",
        "province_district",
        "brand",
        "type",
        "vehicle_type",
        "color",
        "fuel_type",
        "usage_purpose",
    }:
        return len(compact) >= 3 and score >= max(threshold + 3, 8)

    return len(compact) >= 2 and score >= max(threshold + 5, 10)


def extract_fields(
    document: np.ndarray,
    rois: dict[str, Rect],
    field_configs: dict[str, FieldConfig],
    ocr_config: OCRConfig,
    engine: OCREngine,
    page_words: list[object] | None = None,
    anchor_matches: dict[str, AnchorMatch] | None = None,
    page_regex_fallback_enabled: bool = False,
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    page_words = page_words or []
    anchor_matches = anchor_matches or {}
    structured_page_candidates = _extract_structured_candidates_from_page_text(
        field_configs=field_configs,
        page_words=page_words,
        anchor_matches=anchor_matches,
        enabled=page_regex_fallback_enabled,
    )

    for field_name, roi in rois.items():
        cfg = field_configs[field_name]

        patch = crop(document, roi)
        prepared = preprocess_field_crop(patch)

        candidates: list[dict[str, object]] = []

        semantic_candidate = _extract_plate_from_words(field_name, cfg, roi, page_words, document.shape)
        if semantic_candidate is None:
            semantic_candidate = _extract_owner_from_words(field_name, cfg, roi, page_words, document.shape)
        if semantic_candidate is None:
            semantic_candidate = _extract_vehicle_type_from_words(field_name, cfg, roi, page_words, document.shape)
        if semantic_candidate is None:
            semantic_candidate = _extract_date_from_words(field_name, cfg, roi, page_words, document.shape)
        if semantic_candidate is None:
            semantic_candidate = _extract_engine_from_words(field_name, cfg, roi, page_words, document.shape)
        if semantic_candidate is None:
            semantic_candidate = _extract_serial_from_text(field_name, cfg, roi, page_words, document.shape)
        if semantic_candidate is not None:
            candidates.append(semantic_candidate)

        regex_candidate = structured_page_candidates.get(field_name)
        if regex_candidate is not None:
            regex_cleaned = str(regex_candidate["cleaned"])
            regex_method = str(regex_candidate["method"])
            regex_bonus = 18 if regex_method == "page_label_follow_line" else 0
            candidates.append(
                {
                    "raw": str(regex_candidate["raw"]),
                    "cleaned": regex_cleaned,
                    "score": _score_cleaned(regex_cleaned, cfg.cleanup) + regex_bonus,
                    "bbox": roi,
                    "method": regex_method,
                    "priority": 1,
                }
            )

        if cfg.prefer_anchor and cfg.anchor and cfg.anchor in anchor_matches and page_words:
            candidate = locate_value_from_anchor(
                field_cfg=cfg,
                anchor=anchor_matches[cfg.anchor],
                words=page_words,
                doc_shape=document.shape,
            )
            if candidate is not None:
                raw_c = collapse_spaces(candidate.text)
                cleaned_c = cleanup_text(raw_c, cfg.cleanup)
                cleaned_c = _post_cleanup(cleaned_c, cfg)
                candidates.append(
                    {
                        "raw": raw_c,
                        "cleaned": cleaned_c,
                        "score": _score_cleaned(cleaned_c, cfg.cleanup),
                        "bbox": candidate.bbox.clip(document.shape[1], document.shape[0]),
                        "method": candidate.strategy,
                        "priority": 4,
                    }
                )

        if page_words:
            candidate2 = locate_value_from_roi_words(roi, page_words)
            if candidate2 is not None:
                raw_c = collapse_spaces(candidate2.text)
                cleaned_c = cleanup_text(raw_c, cfg.cleanup)
                cleaned_c = _post_cleanup(cleaned_c, cfg)
                candidates.append(
                    {
                        "raw": raw_c,
                        "cleaned": cleaned_c,
                        "score": _score_cleaned(cleaned_c, cfg.cleanup),
                        "bbox": candidate2.bbox.clip(document.shape[1], document.shape[0]),
                        "method": candidate2.strategy,
                        "priority": 3,
                    }
                )

        best, cleaned = _pick_best_candidate(field_name, cfg, candidates)

        if _should_run_crop_ocr(field_name, cfg, ocr_config, best, cleaned):
            field_variants = cfg.crop_ocr_variants or ocr_config.crop_ocr_variants
            variants = {str(v).strip().lower() for v in field_variants}
            stop_crop_variants = False

            if "preprocessed" in variants and not stop_crop_variants:
                raw_t = engine.read_text(prepared, psm=cfg.psm, whitelist=cfg.whitelist)
                raw_t = collapse_spaces(raw_t)
                cleaned_t = cleanup_text(raw_t, cfg.cleanup)
                cleaned_t = _post_cleanup(cleaned_t, cfg)
                score_t = _score_cleaned(cleaned_t, cfg.cleanup)
                candidates.append(
                    {
                        "raw": raw_t,
                        "cleaned": cleaned_t,
                        "score": score_t,
                        "bbox": roi,
                        "method": "roi_tesseract_preprocessed",
                        "priority": 2,
                    }
                )
                stop_crop_variants = _is_good_enough_crop_result(field_name, cfg, cleaned_t, score_t)

            if "preprocessed_alt" in variants and not stop_crop_variants:
                prepared_alt = _preprocess_field_crop_alt(patch)
                raw_t_alt = engine.read_text(prepared_alt, psm=cfg.psm, whitelist=cfg.whitelist)
                raw_t_alt = collapse_spaces(raw_t_alt)
                cleaned_t_alt = cleanup_text(raw_t_alt, cfg.cleanup)
                cleaned_t_alt = _post_cleanup(cleaned_t_alt, cfg)
                score_t_alt = _score_cleaned(cleaned_t_alt, cfg.cleanup)
                candidates.append(
                    {
                        "raw": raw_t_alt,
                        "cleaned": cleaned_t_alt,
                        "score": score_t_alt,
                        "bbox": roi,
                        "method": "roi_tesseract_preprocessed_alt",
                        "priority": 2,
                    }
                )
                stop_crop_variants = _is_good_enough_crop_result(field_name, cfg, cleaned_t_alt, score_t_alt)

            if "raw" in variants and not stop_crop_variants:
                raw_t2 = engine.read_text(patch, psm=cfg.psm, whitelist=cfg.whitelist)
                raw_t2 = collapse_spaces(raw_t2)
                cleaned_t2 = cleanup_text(raw_t2, cfg.cleanup)
                cleaned_t2 = _post_cleanup(cleaned_t2, cfg)
                score_t2 = _score_cleaned(cleaned_t2, cfg.cleanup)
                candidates.append(
                    {
                        "raw": raw_t2,
                        "cleaned": cleaned_t2,
                        "score": score_t2,
                        "bbox": roi,
                        "method": "roi_tesseract_raw",
                        "priority": 2,
                    }
                )

            best, cleaned = _pick_best_candidate(field_name, cfg, candidates)

        if best is None:
            best = {
                "raw": "",
                "cleaned": "",
                "score": 0,
                "bbox": roi,
                "method": "empty_fallback",
                "priority": 0,
            }
            cleaned = ""

        raw = str(best["raw"])
        chosen_roi = best["bbox"]
        method = str(best["method"])
        confidence_score = int(best.get("score", 0))
        low_confidence = bool(cfg.confidence_threshold > 0 and confidence_score < cfg.confidence_threshold)

        output[field_name] = {
            "value": cleaned,
            "raw": collapse_spaces(raw),
            "roi": roi.to_dict(),
            "value_bbox": chosen_roi.to_dict(),
            "method": method,
            "confidence_score": confidence_score,
            "low_confidence": low_confidence,
        }

    return output

