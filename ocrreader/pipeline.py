from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re

import cv2
import numpy as np

from .field_postprocess import postprocess_fields
from .config import AnchorConfig, RuhsatConfig
from .fields import apply_secondary_ocr_fallback, extract_fields, resolve_field_rois
from .io_utils import imread_color, imwrite
from .ocr_engine import OCRWord, collect_runtime_metadata, create_glm_fallback_engine, create_ocr_engine
from .page_word_extractor import extract_field_from_page
from .preprocess import preprocess_document
from .template_anchor_detector import TemplateAnchorDetector, detect_anchors_hybrid
from .text_utils import normalize_for_match
from .types import Rect


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
    "serial_no": 0.55,
}


_SERIAL_MERGED_RE = re.compile(
    r"(?:Seri?|Sen|Sec|Sc|Se)[A-Za-z]{0,6}(\d{4,8})",
    re.IGNORECASE,
)
_SERIAL_SUFFIX_RE = re.compile(r"[A-Z]{1,4}(\d{4,8})$")
_SERIAL_PURE_RE = re.compile(r"^\d{5,8}$")
_SERIAL_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


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
    entry = fields.get("chassis_no")
    if not isinstance(entry, dict):
        return

    value = str(entry.get("value") or "")
    raw_clean = re.sub(r"[^A-Z0-9]", "", value.upper())
    if not (15 <= len(raw_clean) <= 18):
        return

    fixed = _vin_char_fix(raw_clean)
    if fixed == raw_clean:
        return

    patched = dict(entry)
    patched["value"] = fixed
    patched["vin_char_fix_applied"] = True
    method = str(patched.get("method", "") or "")
    patched["method"] = f"{method}|vin_char_fix_v28" if method else "vin_char_fix_v28"
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


def _rescue_serial_no(
    words: list[OCRWord],
    doc_shape: tuple[int, int, int],
    search_region_norm: list[float] | None = None,
) -> str | None:
    h, w = doc_shape[:2]
    if h == 0 or w == 0:
        return None

    if search_region_norm:
        rx, ry, rw, rh = search_region_norm
    else:
        rx, ry, rw, rh = 0.50, 0.75, 0.48, 0.23

    region_texts: list[str] = []
    for word in words:
        cx = (word.bbox.x + word.bbox.w / 2) / w
        cy = (word.bbox.y + word.bbox.h / 2) / h
        if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
            region_texts.append(word.text)

    for text in region_texts:
        merged = _SERIAL_MERGED_RE.search(text)
        if merged and len(merged.group(1)) >= 4 and not _SERIAL_YEAR_RE.fullmatch(merged.group(1)):
            return merged.group(1)

    for text in region_texts:
        suffix = _SERIAL_SUFFIX_RE.search(text.upper())
        if suffix and len(suffix.group(1)) >= 4 and not _SERIAL_YEAR_RE.fullmatch(suffix.group(1)):
            return suffix.group(1)

    for text in region_texts:
        clean = re.sub(r"[^0-9]", "", text)
        if _SERIAL_PURE_RE.match(clean) and not _SERIAL_YEAR_RE.fullmatch(clean):
            return clean

    return None


def _apply_serial_rescue(
    fields: dict[str, dict[str, object]],
    words: list[OCRWord],
    doc_shape: tuple[int, int, int],
) -> None:
    current = fields.get("serial_no")
    if not _is_empty_field_entry(current):
        return

    recovered = _rescue_serial_no(words, doc_shape, [0.52, 0.78, 0.45, 0.2])
    if not recovered:
        return

    entry = dict(current) if isinstance(current, dict) else {}
    entry["value"] = recovered
    entry["raw"] = recovered
    entry["method"] = "page_serial_rescue"
    entry["confidence_score"] = max(60, int(entry.get("confidence_score", 0) or 0))
    entry["low_confidence"] = False
    fields["serial_no"] = entry


class RuhsatOcrPipeline:
    def __init__(self, config: RuhsatConfig):
        self.config = config
        self.engine = create_ocr_engine(config.ocr)
        self.glm_engine = None
        self.glm_engine_error: str | None = None
        try:
            self.glm_engine = create_glm_fallback_engine(config.ocr)
        except Exception as exc:
            self.glm_engine_error = str(exc)
        self.template_detector = TemplateAnchorDetector("config/anchor_templates")

    def process_path(
        self,
        image_path: str,
        debug_dir: str | None = None,
        anchor_debug_rows: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        image = imread_color(image_path)
        return self.process_image(
            image,
            image_label=image_path,
            debug_dir=debug_dir,
            anchor_debug_rows=anchor_debug_rows,
        )

    def process_image(
        self,
        image: np.ndarray,
        image_label: str = "<memory>",
        debug_dir: str | None = None,
        anchor_debug_rows: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:

        prep = preprocess_document(image, self.config.pipeline)
        document = prep.normalized_image

        words = self.engine.iter_words(document, psm=self.config.ocr.psm, min_conf=0.0)
        if anchor_debug_rows is not None:
            _collect_anchor_debug_rows(image_label, words, self.config.anchors, anchor_debug_rows)
        gray = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)
        anchors = detect_anchors_hybrid(words, self.config.anchors, gray, self.template_detector)
        rois = resolve_field_rois(document.shape, self.config.fields, anchors)
        fields = extract_fields(
            document,
            rois,
            self.config.fields,
            self.config.ocr,
            self.engine,
            page_words=words,
            anchor_matches=anchors,
            page_regex_fallback_enabled=self.config.pipeline.page_regex_fallback_enabled,
        )
        _apply_page_second_pass(fields, words, document.shape, rois)
        _apply_serial_rescue(fields, words, document.shape)
        apply_secondary_ocr_fallback(
            document,
            rois,
            fields,
            self.config.fields,
            self.glm_engine,
            allowed_fields=self.config.ocr.glm_fallback_fields,
            min_confidence=self.config.ocr.glm_fallback_min_confidence,
        )
        _apply_chassis_vin_fix(fields)
        fields = postprocess_fields(fields)

        result = {
            "image": image_label,
            "pipeline": {
                "output_width": self.config.pipeline.output_width,
                "output_height": self.config.pipeline.output_height,
                "skew_angle_deg": prep.skew_angle_deg,
                "document_quad": prep.document_quad,
                "glm_fallback_enabled": bool(self.config.ocr.glm_fallback_enabled),
                "glm_fallback_active": bool(self.glm_engine is not None),
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
            "runtime": collect_runtime_metadata(self.config.ocr, self.engine, self.glm_engine),
        }

        if self.glm_engine_error:
            result["pipeline"]["glm_fallback_error"] = self.glm_engine_error
        elif self.glm_engine is not None:
            result["pipeline"]["glm_fallback_source"] = str(getattr(self.glm_engine, "source_path", ""))

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

