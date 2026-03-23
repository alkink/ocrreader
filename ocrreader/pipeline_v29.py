from __future__ import annotations

from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
import re

import cv2
import numpy as np

from .anchors import detect_anchors
from .field_postprocess import postprocess_fields
from .config import AnchorConfig, RuhsatConfig
from .fields import extract_fields, resolve_field_rois
from .io_utils import imread_color, imwrite
from .ocr_engine import OCRWord, create_ocr_engine
from .page_word_extractor import extract_field_from_page
from .preprocess import preprocess_document
from .template_anchor_detector import TemplateAnchorDetector, detect_anchors_hybrid
from .text_utils import normalize_for_match
from .types import Rect


def _infer_doc_size_from_words(words: list[object], fallback_shape: tuple[int, int, int]) -> tuple[int, int]:
    if not words:
        return fallback_shape[1], fallback_shape[0]
    max_x = max((w.bbox.x + w.bbox.w for w in words), default=fallback_shape[1])
    max_y = max((w.bbox.y + w.bbox.h for w in words), default=fallback_shape[0])
    return max(max_x, fallback_shape[1]), max(max_y, fallback_shape[0])


def _inside_center(box: Rect, region: Rect) -> bool:
    cx = box.x + box.w / 2
    cy = box.y + box.h / 2
    return region.x <= cx <= (region.x + region.w) and region.y <= cy <= (region.y + region.h)


def _alias_tokens(alias: str) -> list[str]:
    return [tok for tok in normalize_for_match(alias).split(" ") if tok]


def _best_alias_token_similarity(word_norm: str, aliases: list[str]) -> tuple[str, float]:
    best_alias = ""
    best_score = 0.0
    if not word_norm:
        return best_alias, best_score

    for alias in aliases:
        toks = _alias_tokens(alias)
        if not toks:
            continue
        local_best = max((SequenceMatcher(None, word_norm, tok).ratio() for tok in toks), default=0.0)
        if local_best > best_score:
            best_score = local_best
            best_alias = alias
    return best_alias, best_score


def _collect_anchor_debug_rows(
    image_path: str,
    words: list[OCRWord],
    anchors_cfg: dict[str, AnchorConfig],
    out_rows: list[dict[str, object]],
) -> None:
    max_x = max((w.bbox.x + w.bbox.w for w in words), default=1)
    max_y = max((w.bbox.y + w.bbox.h for w in words), default=1)

    for anchor_name, cfg in anchors_cfg.items():
        if cfg.search_region_norm:
            rx, ry, rw, rh = cfg.search_region_norm
            region = Rect(
                x=int(rx * max_x),
                y=int(ry * max_y),
                w=max(1, int(rw * max_x)),
                h=max(1, int(rh * max_y)),
            )
        else:
            region = Rect(x=0, y=0, w=max(1, int(max_x)), h=max(1, int(max_y)))

        scoped_words = [w for w in words if _inside_center(w.bbox, region)]
        for w in scoped_words:
            word_norm = normalize_for_match(w.text)
            best_alias, best_alias_score = _best_alias_token_similarity(word_norm, cfg.aliases)
            cx = w.bbox.x + w.bbox.w / 2
            cy = w.bbox.y + w.bbox.h / 2
            out_rows.append(
                {
                    "image": image_path,
                    "anchor": anchor_name,
                    "word_text": w.text,
                    "word_norm": word_norm,
                    "word_conf": round(float(w.conf), 2),
                    "block_num": int(w.block_num),
                    "par_num": int(w.par_num),
                    "line_num": int(w.line_num),
                    "word_x": int(w.bbox.x),
                    "word_y": int(w.bbox.y),
                    "word_w": int(w.bbox.w),
                    "word_h": int(w.bbox.h),
                    "word_center_x": round(cx, 2),
                    "word_center_y": round(cy, 2),
                    "region_x": int(region.x),
                    "region_y": int(region.y),
                    "region_w": int(region.w),
                    "region_h": int(region.h),
                    "anchor_min_score": round(float(cfg.min_score), 4),
                    "best_alias": best_alias,
                    "best_alias_token_score": round(best_alias_score, 4),
                    "best_alias_pass": int(best_alias_score >= float(cfg.min_score)),
                }
            )


_SECOND_PASS_MIN_SCORE: dict[str, float] = {
    "tax_or_id_no": 0.55,
    "inspection_date": 0.50,
    "first_registration_date": 0.50,
    "registration_date": 0.55,
    "model_year": 0.55,
}


def _vin_char_fix(value: str) -> str:
    s = re.sub(r"[^A-Z0-9]", "", (value or "").upper())
    if not s:
        return s

    chars = list(s)
    n = len(chars)
    for i, ch in enumerate(chars):
        prev = chars[i - 1] if i > 0 else ""
        nxt = chars[i + 1] if i + 1 < n else ""

        # L8N -> LBN, NM8 -> NMB
        if ch == "8" and prev.isalpha():
            chars[i] = "B"
            continue

        # T5V -> TSV
        if ch == "5" and prev.isalpha() and nxt.isalpha():
            chars[i] = "S"
            continue

        # VE1 -> VF1 (Renault prefix)
        if ch == "E" and i == 1 and chars[0] == "V":
            chars[i] = "F"
            continue

        # VF/I -> VF1 (3rd char of Renault VIN prefix)
        if ch == "I" and i == 2 and "".join(chars[:2]) in {"VF", "VE"}:
            chars[i] = "1"

    return "".join(chars)


def _apply_chassis_vin_fix(fields: dict[str, dict[str, object]]) -> None:
    """
    Apply context-aware VIN char correction to chassis_no.
    v29: also rescues entries where postprocess_validation_failed by trying
    to fix the raw OCR value before it was cleared.
    """
    entry = fields.get("chassis_no")
    if not isinstance(entry, dict):
        return

    value = str(entry.get("value") or "")
    raw_clean = re.sub(r"[^A-Z0-9]", "", value.upper())

    # Try to fix an existing value
    if 15 <= len(raw_clean) <= 18:
        fixed = _vin_char_fix(raw_clean)
        if fixed != raw_clean:
            patched = dict(entry)
            patched["value"] = fixed
            patched["vin_char_fix_applied"] = True
            method = str(patched.get("method", "") or "")
            patched["method"] = f"{method}|vin_char_fix_v29" if method else "vin_char_fix_v29"
            fields["chassis_no"] = patched
            return

    # v29: if value is empty and method indicates validation failed,
    # attempt to rescue using the raw OCR text stored in "raw" field
    method_str = str(entry.get("method", "") or "")
    if not value and "validation_failed" in method_str:
        raw_ocr = re.sub(r"[^A-Z0-9]", "", str(entry.get("raw", "") or "").upper())
        if 15 <= len(raw_ocr) <= 18:
            fixed = _vin_char_fix(raw_ocr)
            patched = dict(entry)
            patched["value"] = fixed
            patched["vin_char_fix_applied"] = True
            patched["method"] = f"rescued_from_validation_failed|vin_char_fix_v29"
            patched["low_confidence"] = True
            fields["chassis_no"] = patched


def _is_empty_field_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return True
    value = entry.get("value")
    if value is None:
        return True
    return not str(value).strip()


def _apply_page_second_pass(
    fields: dict[str, dict[str, object]],
    words: list[OCRWord],
    doc_shape: tuple[int, int, int],
    rois: dict[str, Rect],
) -> None:
    if not words:
        return

    for field_name, min_score in _SECOND_PASS_MIN_SCORE.items():
        current = fields.get(field_name)
        if not _is_empty_field_entry(current):
            continue

        candidate = extract_field_from_page(
            field_name,
            words,
            doc_shape,
            min_score=min_score,
        )
        if candidate is None:
            continue

        entry = dict(current) if isinstance(current, dict) else {}
        entry["value"] = candidate.value
        entry["raw"] = candidate.raw
        entry["method"] = candidate.method
        entry["confidence_score"] = max(
            int(round(candidate.score * 100)),
            int(entry.get("confidence_score", 0) or 0),
        )
        entry["low_confidence"] = False

        if "roi" not in entry and field_name in rois:
            entry["roi"] = rois[field_name].to_dict()
        entry["value_bbox"] = candidate.bbox.to_dict()

        fields[field_name] = entry


class RuhsatOcrPipeline:
    def __init__(self, config: RuhsatConfig):
        self.config = config
        self.engine = create_ocr_engine(config.ocr)
        self.template_detector = TemplateAnchorDetector("config/anchor_templates")

    def process_path(
        self,
        image_path: str,
        debug_dir: str | None = None,
        anchor_debug_rows: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        image = imread_color(image_path)

        prep = preprocess_document(image, self.config.pipeline)
        document = prep.normalized_image

        words = self.engine.iter_words(document, psm=self.config.ocr.psm, min_conf=0.0)
        if anchor_debug_rows is not None:
            _collect_anchor_debug_rows(image_path, words, self.config.anchors, anchor_debug_rows)
        gray = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)
        anchors = detect_anchors_hybrid(words, self.config.anchors, gray, self.template_detector)
        rois = resolve_field_rois(document.shape, self.config.fields, anchors)
        fields = extract_fields(
            document,
            rois,
            self.config.fields,
            self.engine,
            page_words=words,
            anchor_matches=anchors,
            page_regex_fallback_enabled=self.config.pipeline.page_regex_fallback_enabled,
        )
        _apply_page_second_pass(fields, words, document.shape, rois)
        _apply_chassis_vin_fix(fields)
        fields = postprocess_fields(fields)

        result = {
            "image": image_path,
            "pipeline": {
                "output_width": self.config.pipeline.output_width,
                "output_height": self.config.pipeline.output_height,
                "skew_angle_deg": prep.skew_angle_deg,
                "document_quad": prep.document_quad,
            },
            "anchors": {
                name: {
                    "alias": match.alias,
                    "score": round(match.score, 4),
                    "bbox": match.bbox.to_dict(),
                }
                for name, match in anchors.items()
            },
            "fields": fields,
        }

        if debug_dir:
            self._write_debug(debug_dir, prep.normalized_image, anchors, rois)

        return result

    def _write_debug(self, debug_dir: str, normalized: np.ndarray, anchors: dict[str, object], rois: dict[str, object]) -> None:
        out = Path(debug_dir)
        out.mkdir(parents=True, exist_ok=True)

        imwrite(str(out / "normalized.png"), normalized)

        overlay = normalized.copy()
        for name, match in anchors.items():
            b = match.bbox
            cv2.rectangle(overlay, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 180, 0), 2)
            cv2.putText(overlay, f"A:{name}", (b.x, max(20, b.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 0), 2)

        for field_name, roi in rois.items():
            cv2.rectangle(overlay, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 0, 220), 2)
            cv2.putText(
                overlay,
                f"F:{field_name}",
                (roi.x, max(20, roi.y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 0, 220),
                2,
            )

        imwrite(str(out / "overlay.png"), overlay)

