from __future__ import annotations

from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path

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


def _vin_char_fix(vin: str) -> str:
    """
    Context-aware OCR char correction for 17-char VINs.

    Rules (v28):
    - Serial positions (12-16): O→0, I→1, Q→0
    - Non-serial positions:
        * 8→B  when left neighbour is alpha (e.g. NM8→NMB, L8N→LBN)
        * 5→S  when BOTH neighbours are alpha   (e.g. BT5V→BTSV)
        * E→F  at WMI position i=1 when chars[0]=='V'  (Renault VF1)
        * I→1  at WMI position i=2 when VF/VE prefix   (Renault VF1)
    """
    if len(vin) != 17:
        return vin
    chars = list(vin)
    for i, c in enumerate(chars):
        if i >= 12:  # serial positions: digits expected
            if c == "O":
                chars[i] = "0"
            elif c == "I":
                chars[i] = "1"
            elif c == "Q":
                chars[i] = "0"
        else:
            left_alpha = i > 0 and chars[i - 1].isalpha()
            right_alpha = i < 16 and chars[i + 1].isalpha()
            if c == "8" and left_alpha:
                chars[i] = "B"
            elif c == "5" and left_alpha and right_alpha:
                chars[i] = "S"
            elif c == "E" and i == 1 and chars[0] == "V":
                chars[i] = "F"
            elif c == "I" and i == 2 and chars[0] == "V" and chars[1] in ("F", "E"):
                chars[i] = "1"
    return "".join(chars)


def _apply_chassis_vin_fix(fields: dict[str, dict[str, object]]) -> None:
    """Post-process chassis_no value with context-aware VIN char correction."""
    entry = fields.get("chassis_no")
    if not isinstance(entry, dict):
        return
    raw_value = entry.get("value")
    if not raw_value or not isinstance(raw_value, str):
        return
    fixed = _vin_char_fix(raw_value.strip().upper())
    if fixed != raw_value:
        entry = dict(entry)
        entry["value"] = fixed
        entry["method"] = (entry.get("method") or "") + "+vin_char_fix"
        fields["chassis_no"] = entry



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


# v28: serial_no / chassis_no / engine_no second-pass DISABLED.
# All three fields produced more FP than TP when run through page_word_extractor
# (serial: 25 FP / 5 TP; chassis VIN: 15 FP / 1 TP; engine: 14 FP / 5 TP).
# chassis_no gains instead come from postprocess char-fix in field_postprocess.py.
_SECOND_PASS_MIN_SCORE: dict[str, float] = {
    "tax_or_id_no": 0.55,
    "inspection_date": 0.50,
    "first_registration_date": 0.50,
    "registration_date": 0.55,
    "model_year": 0.55,
}


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
        _apply_chassis_vin_fix(fields)  # v28: context-aware char fix before postprocess
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

