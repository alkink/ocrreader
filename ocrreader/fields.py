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

    if field_name in {"serial_no", "owner_surname", "owner_name", "type", "brand"}:
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
        return collapse_spaces(" ".join(alpha[:3]))

    if field_name == "type":
        if up in {"KLET", "MOBIL", "OTOMOBIL", "MOTOSIKLET", "KAMYONET", "MINIBUS"}:
            # These are typically generic types or cut-off fragments from 'vehicle_class' row. 
            # They shouldn't be the 'commercial type / model'.
            return ""
        
        toks = re.findall(r"[A-Z0-9.-]+", up)
        toks = [
            t
            for t in toks
            if t not in VEHICLE_TYPE_BLOCK
            and t not in VEHICLE_TYPE_NOISE
            and not re.fullmatch(r"\d{4}", t)
            and not re.fullmatch(r"D\.?\d+", t)
        ]
        if not toks:
            return ""
        # keep richer type phrases to better match GT (e.g., "YENI SYMBOL AUTH.1.5 DCI 65")
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
            if not re.fullmatch(r"[A-Z]{2,6}\d{0,3}", value):
                return ""

    return value


_OWNER_ADDRESS_HINTS = re.compile(
    r"\b(?:MAH\.?|MAHALLESI|CAD\.?|CADDESI|SOK\.?|SOKAK|NO\b|TEPEBASI|ESKISEHIR|ADRES(?:I)?)\b"
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
        return bool(_OWNER_ADDRESS_HINTS.search(up))
    if field_name == "type" and _TYPE_BLEED_HINTS.search(up):
        return True
    if score < min_confidence:
        return True
    return False


def apply_secondary_ocr_fallback(
    document: np.ndarray,
    rois: dict[str, Rect],
    fields: dict[str, dict[str, object]],
    field_configs: dict[str, FieldConfig],
    engine: TextReadEngine | None,
    allowed_fields: tuple[str, ...],
    min_confidence: int,
    method_name: str = "glm_roi_fallback",
) -> None:
    if engine is None or not allowed_fields:
        return

    for field_name in allowed_fields:
        cfg = field_configs.get(field_name)
        roi = rois.get(field_name)
        if cfg is None or roi is None:
            continue

        current = fields.get(field_name)
        if not _should_run_secondary_ocr(field_name, current, min_confidence):
            continue

        patch = crop(document, roi)
        raw = collapse_spaces(engine.read_text(patch, psm=cfg.psm, whitelist=cfg.whitelist))
        
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
        entry = dict(current) if isinstance(current, dict) else {}
        entry["value"] = cleaned
        entry["raw"] = raw
        entry["method"] = method_name
        entry["confidence_score"] = max(score, int(entry.get("confidence_score", 0) or 0))
        entry["low_confidence"] = bool(cfg.confidence_threshold > 0 and score < cfg.confidence_threshold)
        entry["roi"] = roi.to_dict()
        entry["value_bbox"] = roi.to_dict()
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
    if field_name not in {"owner_name", "owner_surname"}:
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
    if field_name not in {"type"}:
        return None
    if not page_words:
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

    tokens: list[tuple[str, Rect]] = []
    for w in sorted(words, key=lambda ww: (ww.bbox.y, ww.bbox.x)):
        up = normalize_turkish_ascii(str(w.text or ""))
        for tok in re.findall(r"[A-Z0-9.-]+", up):
            if tok in VEHICLE_TYPE_BLOCK:
                continue
            if re.fullmatch(r"\d{4}", tok):
                continue
            if tok in LABEL_NOISE:
                continue
            if len(tok) <= 1:
                continue
            tokens.append((tok, w.bbox))

    if not tokens:
        return None

    mixed = [(t, b) for t, b in tokens if bool(re.search(r"[A-Z]", t)) and bool(re.search(r"\d", t))]
    chosen = mixed[0] if mixed else tokens[0]
    cleaned = chosen[0]
    cleaned = _post_cleanup(cleaned, cfg)
    cleaned = _post_field_filters(field_name, cleaned)
    if not cleaned:
        return None

    return {
        "raw": cleaned,
        "cleaned": cleaned,
        "score": _score_cleaned(cleaned, cfg.cleanup) + 8,
        "bbox": chosen[1].clip(doc_w, doc_h),
        "method": "semantic_vehicle_type_roi_words",
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
    line_texts: list[str] = []
    for line_words in lines:
        ordered = sorted(line_words, key=lambda ww: int(ww.bbox.x))
        txt = collapse_spaces(" ".join(normalize_turkish_ascii(str(getattr(w, "text", "") or "")) for w in ordered))
        if txt:
            line_texts.append(txt)

    full_text = "\n".join(line_texts)
    full_text = full_text.replace("О", "O").replace("І", "I")
    if not full_text:
        return {}

    out: dict[str, dict[str, str]] = {}

    def maybe_add(field_name: str, raw: str, method: str) -> None:
        if field_name in out:
            return
        cfg = field_configs.get(field_name)
        if cfg is None:
            return
        # Safe mode: only use global regex fallback when anchor is not detected.
        if cfg.anchor and cfg.anchor in anchor_matches:
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

    return out


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
            semantic_candidate = _extract_serial_from_text(field_name, cfg, roi, page_words, document.shape)
        if semantic_candidate is not None:
            candidates.append(semantic_candidate)

        regex_candidate = structured_page_candidates.get(field_name)
        if regex_candidate is not None:
            regex_cleaned = str(regex_candidate["cleaned"])
            candidates.append(
                {
                    "raw": str(regex_candidate["raw"]),
                    "cleaned": regex_cleaned,
                    "score": _score_cleaned(regex_cleaned, cfg.cleanup),
                    "bbox": roi,
                    "method": str(regex_candidate["method"]),
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
            variants = {str(v).strip().lower() for v in ocr_config.crop_ocr_variants}

            if "preprocessed" in variants:
                raw_t = engine.read_text(prepared, psm=cfg.psm, whitelist=cfg.whitelist)
                raw_t = collapse_spaces(raw_t)
                cleaned_t = cleanup_text(raw_t, cfg.cleanup)
                cleaned_t = _post_cleanup(cleaned_t, cfg)
                candidates.append(
                    {
                        "raw": raw_t,
                        "cleaned": cleaned_t,
                        "score": _score_cleaned(cleaned_t, cfg.cleanup),
                        "bbox": roi,
                        "method": "roi_tesseract_preprocessed",
                        "priority": 2,
                    }
                )

            if "preprocessed_alt" in variants:
                prepared_alt = _preprocess_field_crop_alt(patch)
                raw_t_alt = engine.read_text(prepared_alt, psm=cfg.psm, whitelist=cfg.whitelist)
                raw_t_alt = collapse_spaces(raw_t_alt)
                cleaned_t_alt = cleanup_text(raw_t_alt, cfg.cleanup)
                cleaned_t_alt = _post_cleanup(cleaned_t_alt, cfg)
                candidates.append(
                    {
                        "raw": raw_t_alt,
                        "cleaned": cleaned_t_alt,
                        "score": _score_cleaned(cleaned_t_alt, cfg.cleanup),
                        "bbox": roi,
                        "method": "roi_tesseract_preprocessed_alt",
                        "priority": 2,
                    }
                )

            if "raw" in variants:
                raw_t2 = engine.read_text(patch, psm=cfg.psm, whitelist=cfg.whitelist)
                raw_t2 = collapse_spaces(raw_t2)
                cleaned_t2 = cleanup_text(raw_t2, cfg.cleanup)
                cleaned_t2 = _post_cleanup(cleaned_t2, cfg)
                candidates.append(
                    {
                        "raw": raw_t2,
                        "cleaned": cleaned_t2,
                        "score": _score_cleaned(cleaned_t2, cfg.cleanup),
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

