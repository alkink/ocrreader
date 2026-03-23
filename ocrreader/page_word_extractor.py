from __future__ import annotations

from dataclasses import dataclass
import re

from .text_utils import normalize_for_match, normalize_turkish_ascii
from .types import Rect, union_rects


_VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
_PLATE_RE = re.compile(r"^\d{2}[A-Z]{1,3}\d{2,4}$")
_DATE_RE = re.compile(r"\b(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{2,4})\b")
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
_TAX_RE = re.compile(r"\b\d{10,11}\b")
_SERIAL_RE = re.compile(r"\b\d{4,7}\b")


_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "chassis_no": (0.00, 0.42, 0.60, 0.92),
    "engine_no": (0.00, 0.34, 0.62, 0.68),
    "plate": (0.00, 0.04, 0.48, 0.44),
    "model_year": (0.15, 0.20, 0.58, 0.58),
    "tax_or_id_no": (0.48, 0.00, 1.00, 0.32),
    "serial_no": (0.46, 0.68, 1.00, 1.00),
    "inspection_date": (0.46, 0.48, 1.00, 1.00),
    "first_registration_date": (0.12, 0.04, 0.62, 0.30),
    "registration_date": (0.12, 0.14, 0.64, 0.42),
    "brand": (0.00, 0.22, 0.52, 0.56),
}


_DATE_LABEL_HINTS: dict[str, tuple[str, ...]] = {
    "first_registration_date": ("ILK", "ILKTESCIL", "ILK TESCIL"),
    "registration_date": ("TESCIL TARIHI", "TESCILTARIHI", "TESCIL TAR"),
    "inspection_date": ("MUAYENE", "DIGER", "MUA GEC"),
}


_NOISE_WORDS = {
    "PLAKA",
    "MARKASI",
    "MARKA",
    "TIPI",
    "TIP",
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
    "VERGI",
    "NO",
}


@dataclass(frozen=True)
class PageCandidate:
    value: str
    raw: str
    bbox: Rect
    score: float
    method: str


def _bbox(word: object) -> Rect:
    box = getattr(word, "bbox", None)
    if isinstance(box, Rect):
        return box
    return Rect(0, 0, 1, 1)


def _text(word: object) -> str:
    return str(getattr(word, "text", "") or "")


def _norm(word: object) -> str:
    return normalize_for_match(_text(word))


def _doc_size(words: list[object], doc_shape: tuple[int, int, int] | tuple[int, int] | None) -> tuple[int, int]:
    if doc_shape and len(doc_shape) >= 2:
        return int(doc_shape[1]), int(doc_shape[0])
    max_x = max((_bbox(w).x + _bbox(w).w for w in words), default=1)
    max_y = max((_bbox(w).y + _bbox(w).h for w in words), default=1)
    return max(1, int(max_x)), max(1, int(max_y))


def _region_rect(field_name: str, doc_w: int, doc_h: int) -> Rect:
    x1, y1, x2, y2 = _REGIONS.get(field_name, (0.0, 0.0, 1.0, 1.0))
    x = int(x1 * doc_w)
    y = int(y1 * doc_h)
    w = max(1, int((x2 - x1) * doc_w))
    h = max(1, int((y2 - y1) * doc_h))
    return Rect(x=x, y=y, w=w, h=h).clip(doc_w, doc_h)


def _inside_center(box: Rect, region: Rect) -> bool:
    cx = box.x + box.w / 2.0
    cy = box.y + box.h / 2.0
    return region.x <= cx <= (region.x + region.w) and region.y <= cy <= (region.y + region.h)


def _center(box: Rect) -> tuple[float, float]:
    return box.x + box.w / 2.0, box.y + box.h / 2.0


def _label_positions(words: list[object], hints: tuple[str, ...]) -> list[tuple[float, float]]:
    norm_hints = [normalize_for_match(h) for h in hints if h]
    out: list[tuple[float, float]] = []
    for w in words:
        nw = _norm(w)
        if not nw:
            continue
        if any(h in nw for h in norm_hints):
            out.append(_center(_bbox(w)))
    return out


def _label_bonus(cx: float, cy: float, labels: list[tuple[float, float]], doc_w: int, doc_h: int) -> float:
    if not labels:
        return 0.0
    best = 99.0
    for lx, ly in labels:
        dist = abs(cx - lx) / max(1.0, float(doc_w)) + abs(cy - ly) / max(1.0, float(doc_h))
        if dist < best:
            best = dist
    if best <= 0.08:
        return 0.28
    if best <= 0.14:
        return 0.18
    if best <= 0.20:
        return 0.10
    return 0.0


def _fix_vin_ocr(text: str) -> str:
    s = re.sub(r"[^A-Z0-9]", "", normalize_turkish_ascii(text))
    chars = list(s)
    for i, ch in enumerate(chars):
        if ch in {"O", "Q"}:
            chars[i] = "0"
        elif ch == "I" and i >= 3:
            chars[i] = "1"
    return "".join(chars)


def _extract_chassis(words: list[object], doc_w: int, doc_h: int) -> list[PageCandidate]:
    region = _region_rect("chassis_no", doc_w, doc_h)
    labels = _label_positions(words, ("SASE", "SASE NO", "SASENO", "ESASENO", "CHASSIS"))
    candidates: list[PageCandidate] = []

    sorted_words = sorted(words, key=lambda w: (int(_bbox(w).y), int(_bbox(w).x)))

    def add_candidate(raw: str, box: Rect, merge_bonus: float = 0.0) -> None:
        fixed = _fix_vin_ocr(raw)
        m = _VIN_RE.search(fixed)
        if not m:
            return
        vin = m.group(0)
        cx, cy = _center(box)
        score = 0.62 + merge_bonus
        if _inside_center(box, region):
            score += 0.20
        score += _label_bonus(cx, cy, labels, doc_w, doc_h)
        candidates.append(
            PageCandidate(
                value=vin,
                raw=raw,
                bbox=box.clip(doc_w, doc_h),
                score=min(score, 1.0),
                method="page_second_pass_chassis_vin",
            )
        )

    for w in sorted_words:
        add_candidate(_text(w), _bbox(w))

    for i in range(len(sorted_words) - 1):
        w1 = sorted_words[i]
        w2 = sorted_words[i + 1]
        b1 = _bbox(w1)
        b2 = _bbox(w2)

        cy1 = b1.y + b1.h / 2.0
        cy2 = b2.y + b2.h / 2.0
        if abs(cy1 - cy2) > max(10.0, doc_h * 0.02):
            continue

        merged_raw = f"{_text(w1)}{_text(w2)}"
        merged_box = union_rects([b1, b2])
        add_candidate(merged_raw, merged_box, merge_bonus=0.05)

    return candidates


def _extract_serial(words: list[object], doc_w: int, doc_h: int) -> list[PageCandidate]:
    region = _region_rect("serial_no", doc_w, doc_h)
    labels = _label_positions(words, ("BELGE", "SERI", "SERI NO", "BELGE SERI"))
    fallback_region = Rect(
        x=int(0.52 * doc_w),
        y=int(0.78 * doc_h),
        w=max(1, doc_w - int(0.52 * doc_w)),
        h=max(1, doc_h - int(0.78 * doc_h)),
    ).clip(doc_w, doc_h)
    out: list[PageCandidate] = []
    seen: set[tuple[str, int, int, int, int]] = set()

    def _append(value: str, raw: str, box: Rect, score: float, method: str) -> None:
        key = (value, box.x, box.y, box.w, box.h)
        if key in seen:
            return
        seen.add(key)
        out.append(
            PageCandidate(
                value=value,
                raw=raw,
                bbox=box.clip(doc_w, doc_h),
                score=min(score, 1.0),
                method=method,
            )
        )

    for w in words:
        box = _bbox(w)
        if not _inside_center(box, region):
            continue
        nw = _norm(w)
        if not nw:
            continue

        for m in _SERIAL_RE.finditer(nw):
            val = m.group(0)
            if len(val) < 4 or len(val) > 7:
                continue

            cx, cy = _center(box)
            score = 0.48
            if len(val) == 6:
                score += 0.18
            score += _label_bonus(cx, cy, labels, doc_w, doc_h)

            _append(val, _text(w), box, score, "page_second_pass_serial")

    # Label-less spatial fallback for lower-right serial zone.
    # Intended for motorcycle-like layouts where serial label OCR is frequently missed.
    for w in words:
        box = _bbox(w)
        if not _inside_center(box, fallback_region):
            continue
        nw = _norm(w)
        if not nw:
            continue
        for m in _SERIAL_RE.finditer(nw):
            val = m.group(0)
            if len(val) < 4 or len(val) > 7:
                continue
            _append(val, _text(w), box, 0.42, "page_second_pass_serial_region")

    return out


def _extract_tax(words: list[object], doc_w: int, doc_h: int) -> list[PageCandidate]:
    region = _region_rect("tax_or_id_no", doc_w, doc_h)
    labels = _label_positions(words, ("VERGI", "VERGI NO", "VERGINO", "KIMLIK"))
    out: list[PageCandidate] = []

    for w in words:
        box = _bbox(w)
        if not _inside_center(box, region):
            continue
        nw = _norm(w)
        if not nw:
            continue

        for m in _TAX_RE.finditer(nw):
            val = m.group(0)
            cx, cy = _center(box)
            score = 0.50 + _label_bonus(cx, cy, labels, doc_w, doc_h)
            if len(val) == 11:
                score += 0.05
            out.append(
                PageCandidate(
                    value=val,
                    raw=_text(w),
                    bbox=box.clip(doc_w, doc_h),
                    score=min(score, 1.0),
                    method="page_second_pass_tax",
                )
            )

    return out


def _extract_engine(words: list[object], doc_w: int, doc_h: int) -> list[PageCandidate]:
    region = _region_rect("engine_no", doc_w, doc_h)
    labels = _label_positions(words, ("MOTOR", "MOTOR NO", "MOTORNO"))
    out: list[PageCandidate] = []

    for w in words:
        box = _bbox(w)
        if not _inside_center(box, region):
            continue

        token = re.sub(r"[^A-Z0-9]", "", normalize_turkish_ascii(_text(w)))
        if len(token) < 6 or len(token) > 20:
            continue
        if token.isdigit() or token in _NOISE_WORDS:
            continue
        if not (re.search(r"[A-Z]", token) and re.search(r"\d", token)):
            continue

        cx, cy = _center(box)
        score = 0.50 + _label_bonus(cx, cy, labels, doc_w, doc_h)
        out.append(
            PageCandidate(
                value=token,
                raw=_text(w),
                bbox=box.clip(doc_w, doc_h),
                score=min(score, 1.0),
                method="page_second_pass_engine",
            )
        )

    return out


def _extract_date(field_name: str, words: list[object], doc_w: int, doc_h: int) -> list[PageCandidate]:
    region = _region_rect(field_name, doc_w, doc_h)
    labels = _label_positions(words, _DATE_LABEL_HINTS.get(field_name, ()))
    out: list[PageCandidate] = []

    for w in words:
        box = _bbox(w)
        if not _inside_center(box, region):
            continue
        nw = _norm(w)
        if not nw:
            continue

        for m in _DATE_RE.finditer(nw):
            try:
                dd = int(m.group(1))
                mm = int(m.group(2))
                yy_raw = m.group(3)
                yy = int(f"20{yy_raw}" if len(yy_raw) == 2 else yy_raw)
            except ValueError:
                continue

            if not (1 <= dd <= 31 and 1 <= mm <= 12 and 1950 <= yy <= 2035):
                continue

            value = f"{dd:02d}/{mm:02d}/{yy:04d}"
            cx, cy = _center(box)
            score = 0.50 + _label_bonus(cx, cy, labels, doc_w, doc_h)

            out.append(
                PageCandidate(
                    value=value,
                    raw=_text(w),
                    bbox=box.clip(doc_w, doc_h),
                    score=min(score, 1.0),
                    method=f"page_second_pass_{field_name}",
                )
            )

    return out


def _extract_model_year(words: list[object], doc_w: int, doc_h: int) -> list[PageCandidate]:
    region = _region_rect("model_year", doc_w, doc_h)
    labels = _label_positions(words, ("MODEL", "MODEL YILI", "YILI"))
    out: list[PageCandidate] = []

    for w in words:
        box = _bbox(w)
        if not _inside_center(box, region):
            continue
        nw = _norm(w)
        if not nw:
            continue
        for m in _YEAR_RE.finditer(nw):
            value = m.group(1)
            cx, cy = _center(box)
            score = 0.52 + _label_bonus(cx, cy, labels, doc_w, doc_h)
            out.append(
                PageCandidate(
                    value=value,
                    raw=_text(w),
                    bbox=box.clip(doc_w, doc_h),
                    score=min(score, 1.0),
                    method="page_second_pass_model_year",
                )
            )

    return out


def _extract_plate(words: list[object], doc_w: int, doc_h: int) -> list[PageCandidate]:
    region = _region_rect("plate", doc_w, doc_h)
    labels = _label_positions(words, ("PLAKA",))
    out: list[PageCandidate] = []

    by_line: dict[tuple[int, int, int], list[object]] = {}
    for w in words:
        box = _bbox(w)
        if not _inside_center(box, region):
            continue
        key = (
            int(getattr(w, "block_num", 0)),
            int(getattr(w, "par_num", 0)),
            int(getattr(w, "line_num", 0)),
        )
        by_line.setdefault(key, []).append(w)

    for line_words in by_line.values():
        ordered = sorted(line_words, key=lambda ww: int(_bbox(ww).x))
        tokens: list[tuple[str, Rect, str]] = []
        for w in ordered:
            raw = _text(w)
            token = re.sub(r"[^A-Z0-9]", "", normalize_turkish_ascii(raw))
            if token:
                tokens.append((token, _bbox(w), raw))

        for i in range(len(tokens)):
            combined = ""
            boxes: list[Rect] = []
            raw_parts: list[str] = []
            for j in range(i, min(i + 3, len(tokens))):
                tok, box, raw = tokens[j]
                combined += tok
                boxes.append(box)
                raw_parts.append(raw)
                if _PLATE_RE.fullmatch(combined):
                    cand_box = union_rects(boxes).clip(doc_w, doc_h)
                    cx, cy = _center(cand_box)
                    score = 0.54 + _label_bonus(cx, cy, labels, doc_w, doc_h)
                    out.append(
                        PageCandidate(
                            value=combined,
                            raw=" ".join(raw_parts),
                            bbox=cand_box,
                            score=min(score, 1.0),
                            method="page_second_pass_plate",
                        )
                    )

    return out


def _extract_brand(words: list[object], doc_w: int, doc_h: int) -> list[PageCandidate]:
    region = _region_rect("brand", doc_w, doc_h)
    labels = _label_positions(words, ("MARKASI", "MARKA"))
    out: list[PageCandidate] = []
    pat = re.compile(r"^[A-Z][A-Z0-9.-]{1,19}$")

    for w in words:
        box = _bbox(w)
        if not _inside_center(box, region):
            continue
        token = re.sub(r"[^A-Z0-9.-]", "", normalize_turkish_ascii(_text(w)))
        if not token or token in _NOISE_WORDS:
            continue
        if not pat.fullmatch(token):
            continue

        cx, cy = _center(box)
        score = 0.52 + _label_bonus(cx, cy, labels, doc_w, doc_h)
        out.append(
            PageCandidate(
                value=token,
                raw=_text(w),
                bbox=box.clip(doc_w, doc_h),
                score=min(score, 1.0),
                method="page_second_pass_brand",
            )
        )

    return out


def extract_field_from_page(
    field_name: str,
    words: list[object],
    doc_shape: tuple[int, int, int] | tuple[int, int] | None,
    *,
    min_score: float = 0.55,
) -> PageCandidate | None:
    if not words:
        return None

    doc_w, doc_h = _doc_size(words, doc_shape)

    candidates: list[PageCandidate] = []
    if field_name == "chassis_no":
        candidates = _extract_chassis(words, doc_w, doc_h)
    elif field_name == "serial_no":
        candidates = _extract_serial(words, doc_w, doc_h)
    elif field_name == "engine_no":
        candidates = _extract_engine(words, doc_w, doc_h)
    elif field_name == "tax_or_id_no":
        candidates = _extract_tax(words, doc_w, doc_h)
    elif field_name in {"inspection_date", "first_registration_date", "registration_date"}:
        candidates = _extract_date(field_name, words, doc_w, doc_h)
    elif field_name == "model_year":
        candidates = _extract_model_year(words, doc_w, doc_h)
    elif field_name == "plate":
        candidates = _extract_plate(words, doc_w, doc_h)
    elif field_name == "brand":
        candidates = _extract_brand(words, doc_w, doc_h)

    if not candidates:
        return None

    best = max(candidates, key=lambda c: c.score)
    if best.score < min_score:
        return None
    return best

