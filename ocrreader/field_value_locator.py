from __future__ import annotations

from dataclasses import dataclass
import re
from .anchors import AnchorMatch
from .config import FieldConfig
from .ocr_engine import OCRWord
from .types import Rect, union_rects


@dataclass(frozen=True)
class FieldValueCandidate:
    text: str
    bbox: Rect
    words: list[OCRWord]
    strategy: str


@dataclass(frozen=True)
class LineCandidate:
    words: list[OCRWord]
    bbox: Rect
    text: str
    avg_conf: float


def _line_of(word: OCRWord) -> tuple[int, int, int]:
    return (word.block_num, word.par_num, word.line_num)


def _make_window(anchor: Rect, doc_w: int, doc_h: int, direction: str, margin: tuple[float, float, float, float]) -> Rect:
    left, top, right, bottom = margin

    if direction == "right":
        x = anchor.x + anchor.w + int(left * doc_w)
        y = anchor.y - int(top * doc_h)
        w = int((right - left) * doc_w)
        h = anchor.h + int((top + bottom) * doc_h)
        return Rect(x=x, y=y, w=max(1, w), h=max(1, h)).clip(doc_w, doc_h)

    # default: below
    x = anchor.x - int(left * doc_w)
    y = anchor.y + anchor.h + int(top * doc_h)
    w = anchor.w + int((left + right) * doc_w)
    h = int((bottom - top) * doc_h)
    return Rect(x=x, y=y, w=max(1, w), h=max(1, h)).clip(doc_w, doc_h)


def _inside(word_box: Rect, window: Rect) -> bool:
    cx = word_box.x + word_box.w / 2.0
    cy = word_box.y + word_box.h / 2.0
    return window.x <= cx <= (window.x + window.w) and window.y <= cy <= (window.y + window.h)


def _join_words(words: list[OCRWord]) -> str:
    return " ".join(w.text for w in words).strip()


def _group_lines(words: list[OCRWord]) -> list[LineCandidate]:
    if not words:
        return []

    groups: dict[tuple[int, int, int], list[OCRWord]] = {}
    for w in words:
        groups.setdefault(_line_of(w), []).append(w)

    lines: list[LineCandidate] = []
    for line_words in groups.values():
        line_words = sorted(line_words, key=lambda w: w.bbox.x)
        text = _join_words(line_words)
        if not text:
            continue
        bbox = union_rects([w.bbox for w in line_words])
        avg_conf = sum(w.conf for w in line_words) / max(1, len(line_words))
        lines.append(LineCandidate(words=line_words, bbox=bbox, text=text, avg_conf=avg_conf))

    lines.sort(key=lambda ln: (ln.bbox.y, ln.bbox.x))
    return lines


def _is_plausible_text(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 2:
        return False
    alnum_count = sum(ch.isalnum() for ch in stripped)
    return alnum_count >= 2


def _text_quality(text: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9-]+", text)
    if not tokens:
        return 0

    longest = max(len(t) for t in tokens)
    mixed = any(bool(re.search(r"[A-Za-z]", t)) and bool(re.search(r"\d", t)) for t in tokens)
    digits = sum(ch.isdigit() for ch in text)
    letters = sum(ch.isalpha() for ch in text)

    return longest + (5 if mixed else 0) + min(letters, 8) // 3 + min(digits, 8) // 3


def _pick_best_line(direction: str, anchor: AnchorMatch, lines: list[LineCandidate]) -> LineCandidate | None:
    if not lines:
        return None

    anchor_bottom = anchor.bbox.y + anchor.bbox.h
    anchor_right = anchor.bbox.x + anchor.bbox.w
    anchor_cy = anchor.bbox.y + anchor.bbox.h / 2.0

    if direction == "right":
        rights = [ln for ln in lines if ln.bbox.x >= anchor_right - 4 and _is_plausible_text(ln.text)]
        pool = rights if rights else [ln for ln in lines if _is_plausible_text(ln.text)]
        if not pool:
            return None

        def score_right(ln: LineCandidate) -> tuple[float, float, float]:
            cy = ln.bbox.y + ln.bbox.h / 2.0
            dx = max(0.0, anchor_right - ln.bbox.x)
            dy = abs(cy - anchor_cy)
            quality = _text_quality(ln.text)
            return (dy + 0.35 * dx - 2.5 * quality, -ln.avg_conf, ln.bbox.x)

        return min(pool, key=score_right)

    # default: below
    belows = [ln for ln in lines if ln.bbox.y >= anchor_bottom - 4 and _is_plausible_text(ln.text)]
    pool = belows if belows else [ln for ln in lines if _is_plausible_text(ln.text)]
    if not pool:
        return None

    def score_below(ln: LineCandidate) -> tuple[float, float, float]:
        dy = abs(ln.bbox.y - anchor_bottom)
        quality = _text_quality(ln.text)
        return (dy - 2.0 * quality, -ln.avg_conf, ln.bbox.x)

    return min(pool, key=score_below)


def locate_value_from_anchor(
    field_cfg: FieldConfig,
    anchor: AnchorMatch,
    words: list[OCRWord],
    doc_shape: tuple[int, int, int],
) -> FieldValueCandidate | None:
    doc_h, doc_w = doc_shape[:2]

    direction = field_cfg.value_from_anchor if field_cfg.value_from_anchor in {"below", "right"} else "below"
    margin = field_cfg.value_margin_norm or (0.0, 0.0, 0.22, 0.10)
    window = _make_window(anchor.bbox, doc_w, doc_h, direction, margin)

    in_window = [w for w in words if _inside(w.bbox, window)]
    if not in_window:
        return None

    lines = _group_lines(in_window)
    best = _pick_best_line(direction, anchor, lines)
    if best is None:
        return None

    return FieldValueCandidate(text=best.text, bbox=best.bbox, words=best.words, strategy=f"anchor_{direction}_line")


def locate_value_from_roi_words(
    roi: Rect,
    words: list[OCRWord],
) -> FieldValueCandidate | None:
    in_roi = [w for w in words if _inside(w.bbox, roi)]
    if not in_roi:
        return None

    lines = [ln for ln in _group_lines(in_roi) if _is_plausible_text(ln.text)]
    if not lines:
        return None

    roi_cy = roi.y + roi.h / 2.0

    def score(ln: LineCandidate) -> tuple[float, float, float]:
        cy = ln.bbox.y + ln.bbox.h / 2.0
        dy = abs(cy - roi_cy)
        return (dy, -ln.avg_conf, ln.bbox.x)

    best = min(lines, key=score)
    return FieldValueCandidate(text=best.text, bbox=best.bbox, words=best.words, strategy="roi_line")

