from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from .config import AnchorConfig
from .ocr_engine import OCRWord
from .text_utils import normalize_for_match
from .types import Rect, union_rects


@dataclass(frozen=True)
class AnchorMatch:
    name: str
    alias: str
    score: float
    bbox: Rect


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _token_similarity(a: str, b: str) -> float:
    seq_score = _sim(a, b)

    max_len = max(len(a), len(b), 1)
    max_dist = 1 if max_len <= 4 else 2
    if abs(len(a) - len(b)) > (max_dist + 1):
        return seq_score

    dist = _levenshtein(a, b)
    lev_score = max(0.0, 1.0 - (dist / max_len))
    return max(seq_score, lev_score)


def _split_alias(alias: str) -> list[str]:
    norm = normalize_for_match(alias)
    return [w for w in norm.split(" ") if w]


def _match_alias(words: list[OCRWord], alias_tokens: list[str], min_score: float) -> tuple[float, Rect] | None:
    if not alias_tokens:
        return None

    normalized = [normalize_for_match(w.text) for w in words]
    n = len(alias_tokens)

    best_score = -1.0
    best_rect: Rect | None = None

    for i in range(0, len(words) - n + 1):
        seq_words = words[i : i + n]
        seq_norm = normalized[i : i + n]

        if n > 1:
            line_keys = {(w.block_num, w.par_num, w.line_num) for w in seq_words}
            if len(line_keys) != 1:
                continue

        scores = [_token_similarity(seq_norm[j], alias_tokens[j]) for j in range(n)]
        score = sum(scores) / n
        if score < min_score:
            continue

        rect = union_rects([w.bbox for w in seq_words])
        if score > best_score:
            best_score = score
            best_rect = rect

    # Paddle may merge multi-token aliases into a single OCR token
    # (e.g., "MOTOR NO" -> "MOTORNO", "BELGE SERI" -> "BELGESERI").
    if n > 1:
        alias_concat = "".join(alias_tokens)
        max_window = min(3, len(words))

        for win in range(1, max_window + 1):
            for i in range(0, len(words) - win + 1):
                seq_words = words[i : i + win]

                if win > 1:
                    line_keys = {(w.block_num, w.par_num, w.line_num) for w in seq_words}
                    if len(line_keys) != 1:
                        continue

                merged = "".join(normalized[i + j] for j in range(win))
                merged = merged.replace(" ", "")
                if not merged:
                    continue

                if abs(len(merged) - len(alias_concat)) > 2:
                    continue

                score = _token_similarity(merged, alias_concat)
                if score < min_score:
                    continue

                rect = union_rects([w.bbox for w in seq_words])
                if score > best_score:
                    best_score = score
                    best_rect = rect

    if best_rect is None:
        return None
    return best_score, best_rect


def detect_anchors(words: list[OCRWord], anchors: dict[str, AnchorConfig]) -> dict[str, AnchorMatch]:
    found: dict[str, AnchorMatch] = {}

    for name, cfg in anchors.items():
        scoped_words = words
        if cfg.search_region_norm:
            # infer doc size from OCR words extents
            max_x = max((w.bbox.x + w.bbox.w for w in words), default=1)
            max_y = max((w.bbox.y + w.bbox.h for w in words), default=1)
            rx, ry, rw, rh = cfg.search_region_norm
            region = Rect(
                x=int(rx * max_x),
                y=int(ry * max_y),
                w=max(1, int(rw * max_x)),
                h=max(1, int(rh * max_y)),
            )

            def inside(box: Rect, reg: Rect) -> bool:
                cx = box.x + box.w / 2
                cy = box.y + box.h / 2
                return reg.x <= cx <= (reg.x + reg.w) and reg.y <= cy <= (reg.y + reg.h)

            scoped_words = [w for w in words if inside(w.bbox, region)]

        best: AnchorMatch | None = None
        for alias in cfg.aliases:
            alias_tokens = _split_alias(alias)
            candidate = _match_alias(scoped_words, alias_tokens, cfg.min_score)
            if candidate is None:
                continue

            score, rect = candidate
            current = AnchorMatch(name=name, alias=alias, score=score, bbox=rect)
            if best is None or current.score > best.score:
                best = current

        if best is not None:
            found[name] = best

    return found

