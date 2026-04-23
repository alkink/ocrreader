from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re
import time

import cv2
import numpy as np

from .field_postprocess import postprocess_fields
from .config import AnchorConfig, RuhsatConfig
from .fields import apply_profile_overrides, apply_secondary_ocr_fallback, extract_fields, resolve_field_rois
from .full_page_vl_parser import (
    apply_d_block_vl_route,
    apply_full_page_vl_route,
    apply_right_page_vl_route,
    merge_full_page_vl_fields,
    parse_d_block_vl_text,
    parse_full_page_vl_text,
    parse_right_page_vl_text,
    score_full_page_vl_candidate,
    should_run_full_page_vl_second_pass,
)
from .io_utils import imread_color, imwrite
from .ocr_engine import OCRWord, collect_runtime_metadata, create_glm_fallback_engine, create_ocr_engine
from .page_word_extractor import extract_field_from_page
from .preprocess import preprocess_document
from .review import evaluate_extraction_review
from .template_anchor_detector import TemplateAnchorDetector, detect_anchors_hybrid
from .text_utils import collapse_spaces, normalize_for_match, normalize_turkish_ascii
from .types import Rect, union_rects


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

GENERIC_VEHICLE_TYPE_HINTS = (
    "OTOMOBIL",
    "OTOBUS",
    "KAMYON",
    "KAMYONET",
    "MINIBUS",
    "PANELVAN",
    "SEDAN",
    "HATCHBACK",
    "BENZIN",
    "DIZEL",
    "ELEKTRIK",
    "HIBRIT",
)

_PIPELINE_PLATE_PATTERN = re.compile(r"\d{2}[A-Z]{1,3}\d{2,4}")
_PIPELINE_VIN_PATTERN = re.compile(r"[A-HJ-NPR-Z0-9]{17}")


def _entry_text(fields: dict[str, dict[str, object]], field_name: str) -> str:
    entry = fields.get(field_name)
    if not isinstance(entry, dict):
        return ""
    return normalize_turkish_ascii(collapse_spaces(str(entry.get("value") or "")))


def _entry_method(fields: dict[str, dict[str, object]], field_name: str) -> str:
    entry = fields.get(field_name)
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("method") or "")


def _entry_confidence(fields: dict[str, dict[str, object]], field_name: str) -> int:
    entry = fields.get(field_name)
    if not isinstance(entry, dict):
        return 0
    try:
        return int(entry.get("confidence_score") or 0)
    except Exception:
        return 0


def _entry_has_company_hint(text: str) -> bool:
    if not text:
        return False
    return any(fragment in text for fragment in OWNER_COMPANY_FRAGMENTS)


def _looks_like_plausible_vehicle_type(text: str, fields: dict[str, dict[str, object]]) -> bool:
    if not text:
        return False
    normalized = normalize_turkish_ascii(collapse_spaces(text))
    compact = re.sub(r"[^A-Z0-9]", "", normalized)
    if not compact:
        return False
    if _PIPELINE_PLATE_PATTERN.fullmatch(compact):
        return False
    if _PIPELINE_VIN_PATTERN.fullmatch(compact):
        return False
    if compact.isdigit():
        return False
    if len(compact) > 18:
        return False

    for other_field in ("plate", "engine_no", "chassis_no", "tax_or_id_no"):
        other = _entry_text(fields, other_field)
        other_compact = re.sub(r"[^A-Z0-9]", "", other)
        if other_compact and compact == other_compact:
            return False

    return bool(re.search(r"[A-Z]", compact))


def _looks_like_weak_owner_text(text: str) -> bool:
    tokens = [tok for tok in re.findall(r"[A-Z]+", text) if tok]
    if not tokens:
        return True
    if len(tokens) == 1 and len(tokens[0]) < 3:
        return True
    if len(tokens) >= 2 and len(tokens[0]) < 3 and len(tokens[1]) < 3:
        return True
    return False


def _full_page_vl_max_new_tokens(config: RuhsatConfig) -> int | None:
    raw = getattr(config.ocr, "paddle_vl_max_new_tokens", None)
    if raw is None:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    if value <= 0:
        return None
    return min(value, 384)


def _patch_field_value(
    fields: dict[str, dict[str, object]],
    target_field: str,
    source_field: str,
    value: str,
    method: str,
) -> None:
    source_entry = fields.get(source_field)
    target_entry = fields.get(target_field)
    base = dict(target_entry) if isinstance(target_entry, dict) else (
        dict(source_entry) if isinstance(source_entry, dict) else {}
    )
    raw_source = ""
    if isinstance(source_entry, dict):
        raw_source = str(source_entry.get("raw") or source_entry.get("value") or "")
    base["value"] = value
    base["raw"] = raw_source or value
    base["method"] = method
    base["postprocess_applied"] = True
    base["low_confidence"] = False
    fields[target_field] = base


def _set_field_empty(
    fields: dict[str, dict[str, object]],
    field_name: str,
    method: str,
) -> None:
    entry = fields.get(field_name)
    patched = dict(entry) if isinstance(entry, dict) else {}
    patched["value"] = ""
    patched["method"] = method
    patched["postprocess_applied"] = True
    patched["low_confidence"] = False
    fields[field_name] = patched


def _apply_profile_harmonization(
    fields: dict[str, dict[str, object]],
    profile_name: str,
) -> list[str]:
    if profile_name not in {"v29_photo_variant_b_candidate", "unknown_or_degraded_photo"}:
        return []

    changed: list[str] = []

    type_text = _entry_text(fields, "type")
    vehicle_type_text = _entry_text(fields, "vehicle_type")
    if _looks_like_plausible_vehicle_type(type_text, fields):
        vehicle_type_generic = (
            not vehicle_type_text
            or any(hint in vehicle_type_text for hint in GENERIC_VEHICLE_TYPE_HINTS)
        )
        if vehicle_type_generic and type_text != vehicle_type_text:
            _patch_field_value(fields, "vehicle_type", "type", type_text, "derived_vehicle_type_from_type")
            changed.append("vehicle_type")

    owner_title = _entry_text(fields, "owner_title")
    owner_surname = _entry_text(fields, "owner_surname")
    owner_name = _entry_text(fields, "owner_name")

    if owner_surname and not owner_title and _entry_has_company_hint(owner_surname):
        _patch_field_value(fields, "owner_title", "owner_surname", owner_surname, "derived_owner_title_from_surname")
        changed.append("owner_title")
        owner_title = owner_surname

    if owner_title and (not owner_surname or _looks_like_weak_owner_text(owner_surname)) and _entry_has_company_hint(owner_title):
        _patch_field_value(fields, "owner_surname", "owner_title", owner_title, "derived_owner_surname_from_title")
        changed.append("owner_surname")
        owner_surname = owner_title

    if owner_name and _entry_has_company_hint(owner_name):
        if not owner_title:
            _patch_field_value(fields, "owner_title", "owner_name", owner_name, "derived_owner_title_from_name")
            changed.append("owner_title")
        if not owner_surname or _looks_like_weak_owner_text(owner_surname):
            _patch_field_value(fields, "owner_surname", "owner_name", owner_name, "derived_owner_surname_from_name")
            changed.append("owner_surname")
        name_entry = fields.get("owner_name")
        if isinstance(name_entry, dict):
            patched = dict(name_entry)
            patched["value"] = None
            patched["method"] = "company_owner_name_cleared"
            patched["postprocess_applied"] = True
            fields["owner_name"] = patched
            changed.append("owner_name")

    return changed


def _normalize_optional_empty_fields(fields: dict[str, dict[str, object]]) -> list[str]:
    changed: list[str] = []
    for field_name in {
        "vehicle_type",
        "owner_name",
        "owner_surname",
        "owner_title",
        "first_registration_date",
        "inspection_date",
        "tax_or_id_no",
        "serial_no",
    }:
        entry = fields.get(field_name)
        if isinstance(entry, dict) and entry.get("value") is None:
            patched = dict(entry)
            patched["value"] = ""
            patched["method"] = str(patched.get("method") or "empty_normalized")
            patched["postprocess_applied"] = True
            fields[field_name] = patched
            changed.append(field_name)
    return changed


def _apply_false_positive_suppression(
    fields: dict[str, dict[str, object]],
    profile_name: str,
) -> list[str]:
    changed: list[str] = []

    owner_title = _entry_text(fields, "owner_title")
    owner_name = _entry_text(fields, "owner_name")
    owner_surname = _entry_text(fields, "owner_surname")

    def _single_short_token(text: str) -> bool:
        tokens = [tok for tok in re.findall(r"[A-Z]+", text) if tok]
        return len(tokens) == 1 and len(tokens[0]) <= 8

    owner_noise_cluster = (
        profile_name == "v29_photo_variant_b_candidate"
        and not owner_title
        and owner_name
        and owner_surname
        and _entry_method(fields, "owner_name") == "anchor_right_line"
        and _entry_method(fields, "owner_surname") == "anchor_right_line"
        and _entry_confidence(fields, "owner_name") <= 6
        and _entry_confidence(fields, "owner_surname") <= 6
        and _single_short_token(owner_name)
        and _single_short_token(owner_surname)
        and not _entry_text(fields, "serial_no")
    )

    if owner_noise_cluster:
        for field_name in ("owner_name", "owner_surname"):
            if _entry_text(fields, field_name):
                _set_field_empty(fields, field_name, "variant_b_owner_noise_cleared")
                changed.append(field_name)

        if (
            _entry_text(fields, "first_registration_date")
            and _entry_text(fields, "first_registration_date") == _entry_text(fields, "registration_date")
            and _entry_method(fields, "first_registration_date") == "semantic_date_roi_words"
            and _entry_method(fields, "registration_date") == "semantic_date_roi_words"
        ):
            _set_field_empty(fields, "first_registration_date", "variant_b_owner_noise_date_cleared")
            changed.append("first_registration_date")

        if _entry_text(fields, "inspection_date") and _entry_method(fields, "inspection_date") == "roi_line":
            _set_field_empty(fields, "inspection_date", "variant_b_owner_noise_date_cleared")
            changed.append("inspection_date")

        if _entry_text(fields, "tax_or_id_no") and _entry_method(fields, "tax_or_id_no") == "page_second_pass_tax":
            _set_field_empty(fields, "tax_or_id_no", "variant_b_owner_noise_tax_cleared")
            changed.append("tax_or_id_no")

    if (
        profile_name == "unknown_or_degraded_photo"
        and _entry_text(fields, "plate")
        and not _entry_text(fields, "type")
        and not _entry_text(fields, "vehicle_type")
        and _entry_text(fields, "owner_title")
        and not _entry_text(fields, "owner_name")
        and _entry_has_company_hint(_entry_text(fields, "owner_title"))
        and _entry_method(fields, "plate") == "semantic_plate_page_words"
        and _entry_confidence(fields, "plate") <= 32
    ):
        _set_field_empty(fields, "plate", "degraded_company_plate_cleared")
        changed.append("plate")

    return changed


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
    "engine_no": 0.55,
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

        # JM7UEY -> JM7UFY (Mazda OCR confusion)
        if ch == "E" and prev == "U" and nxt == "Y":
            chars[i] = "F"
            continue

        # VF/I -> VF1 (3rd char of Renault VIN prefix)
        if ch == "I" and i == 2 and "".join(chars[:2]) in {"VF", "VE"}:
            chars[i] = "1"

        # WSV -> W5V (common OCR confusion in VIN tail)
        if ch == "S" and prev == "W" and nxt == "V":
            chars[i] = "5"

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


def _expand_rect(rect: Rect, doc_shape: tuple[int, int, int], pad_x_norm: float, pad_y_norm: float) -> Rect:
    doc_h, doc_w = doc_shape[:2]
    pad_x = int(doc_w * pad_x_norm)
    pad_y = int(doc_h * pad_y_norm)
    expanded = Rect(
        x=max(0, rect.x - pad_x),
        y=max(0, rect.y - pad_y),
        w=rect.w + pad_x * 2,
        h=rect.h + pad_y * 2,
    )
    return expanded.clip(doc_w, doc_h)


def _build_union_roi(
    rois: dict[str, Rect],
    field_names: tuple[str, ...],
    doc_shape: tuple[int, int, int],
    pad_x_norm: float,
    pad_y_norm: float,
) -> Rect | None:
    rects = [rois[name] for name in field_names if name in rois]
    if not rects:
        return None
    return _expand_rect(union_rects(rects), doc_shape, pad_x_norm=pad_x_norm, pad_y_norm=pad_y_norm)


def _crop_rect(image: np.ndarray, rect: Rect | None) -> np.ndarray | None:
    if rect is None:
        return None
    patch = image[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
    if patch.size == 0:
        return None
    return patch.copy()


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
            force_fields=self.config.ocr.glm_force_fields,
            min_confidence=self.config.ocr.glm_fallback_min_confidence,
            extra_read_kwargs={
                "use_layout_detection": False,
                "max_new_tokens": 256,
            },
        )
        _apply_chassis_vin_fix(fields)
        fields = postprocess_fields(fields)
        preliminary_review = evaluate_extraction_review(fields, anchors)
        profile_route_applied = None
        profile_route_changed_fields: list[str] = []
        right_page_vl_changed_fields: list[str] = []
        d_block_vl_changed_fields: list[str] = []
        right_page_vl_seconds: float | None = None
        d_block_vl_seconds: float | None = None
        full_page_vl_changed_fields: list[str] = []
        full_page_vl_route_changed_fields: list[str] = []
        full_page_vl_seconds: float | None = None
        full_page_vl_retry_seconds: float | None = None
        full_page_vl_selected_pass = "initial"
        selected_profile = str((preliminary_review.get("profile") or {}).get("selected_profile") or "")
        if selected_profile in {"v29_photo_variant_b_candidate", "unknown_or_degraded_photo"}:
            changed_fields = apply_profile_overrides(
                fields=fields,
                field_configs=self.config.fields,
                page_words=words,
                anchor_matches=anchors,
                profile_name=selected_profile,
            )
            harmonized_fields = _apply_profile_harmonization(fields, selected_profile)
            if changed_fields:
                profile_route_applied = selected_profile
                profile_route_changed_fields = changed_fields
                _apply_chassis_vin_fix(fields)
                fields = postprocess_fields(fields)
            if harmonized_fields:
                profile_route_applied = selected_profile
                profile_route_changed_fields = list(dict.fromkeys(profile_route_changed_fields + harmonized_fields))

        enable_right_page_vl = bool(getattr(self.config.ocr, "paddle_vl_enable_right_page_pass", False))
        enable_d_block_vl = bool(getattr(self.config.ocr, "paddle_vl_enable_d_block_pass", False))
        if self.glm_engine is not None and (enable_right_page_vl or enable_d_block_vl):
            right_page_roi = Rect(
                x=int(document.shape[1] * 0.5),
                y=0,
                w=document.shape[1] - int(document.shape[1] * 0.5),
                h=document.shape[0],
            ).clip(document.shape[1], document.shape[0])
            d_block_roi = _build_union_roi(
                rois,
                ("brand", "type", "model_year", "vehicle_class", "vehicle_type", "color"),
                document.shape,
                pad_x_norm=0.02,
                pad_y_norm=0.02,
            )

            right_page_crop = _crop_rect(document, right_page_roi) if enable_right_page_vl else None
            if right_page_crop is not None:
                right_t0 = time.perf_counter()
                right_text = self.glm_engine.read_text(
                    right_page_crop,
                    use_layout_detection=False,
                    max_new_tokens=256,
                )
                right_page_vl_seconds = round(time.perf_counter() - right_t0, 3)
                parsed_right = parse_right_page_vl_text(right_text)
                right_page_vl_changed_fields = apply_right_page_vl_route(
                    fields,
                    parsed_right,
                    force=selected_profile in {"v29_photo_variant_b_candidate", "unknown_or_degraded_photo"},
                )

            d_block_crop = _crop_rect(document, d_block_roi) if enable_d_block_vl else None
            if d_block_crop is not None:
                d_t0 = time.perf_counter()
                d_text = self.glm_engine.read_text(
                    d_block_crop,
                    use_layout_detection=False,
                    max_new_tokens=256,
                )
                d_block_vl_seconds = round(time.perf_counter() - d_t0, 3)
                parsed_d_block = parse_d_block_vl_text(d_text)
                d_block_vl_changed_fields = apply_d_block_vl_route(
                    fields,
                    parsed_d_block,
                    force=selected_profile in {"v29_photo_variant_b_candidate", "unknown_or_degraded_photo"},
                )

            if right_page_vl_changed_fields or d_block_vl_changed_fields:
                _apply_chassis_vin_fix(fields)
                fields = postprocess_fields(fields)

        if self.glm_engine is not None and should_run_full_page_vl_second_pass(fields, selected_profile):
            full_page_vl_input = image if selected_profile in {"v29_photo_variant_b_candidate", "unknown_or_degraded_photo"} else document
            full_page_vl_max_new_tokens = _full_page_vl_max_new_tokens(self.config)
            full_page_vl_t0 = time.perf_counter()
            full_page_vl_text = self.glm_engine.read_text(
                full_page_vl_input,
                max_new_tokens=full_page_vl_max_new_tokens,
            )
            full_page_vl_seconds = round(time.perf_counter() - full_page_vl_t0, 3)
            parsed_full_page_vl = parse_full_page_vl_text(full_page_vl_text)
            initial_full_page_vl_score = score_full_page_vl_candidate(parsed_full_page_vl)
            should_retry_full_page_vl = selected_profile in {"v29_photo_variant_b_candidate", "unknown_or_degraded_photo"} and (
                initial_full_page_vl_score < 18
                or (
                    selected_profile == "unknown_or_degraded_photo"
                    and (
                        not parsed_full_page_vl.get("owner_name")
                        or not parsed_full_page_vl.get("owner_title")
                    )
                )
            )
            if should_retry_full_page_vl:
                retry_t0 = time.perf_counter()
                retry_text = self.glm_engine.read_text(
                    full_page_vl_input,
                    max_new_tokens=full_page_vl_max_new_tokens,
                )
                full_page_vl_retry_seconds = round(time.perf_counter() - retry_t0, 3)
                parsed_retry = parse_full_page_vl_text(retry_text)
                if score_full_page_vl_candidate(parsed_retry) > score_full_page_vl_candidate(parsed_full_page_vl):
                    parsed_full_page_vl = parsed_retry
                    full_page_vl_selected_pass = "retry"
            full_page_vl_changed_fields = merge_full_page_vl_fields(fields, parsed_full_page_vl)
            full_page_vl_route_changed_fields = apply_full_page_vl_route(
                fields,
                parsed_full_page_vl,
                selected_profile=selected_profile,
                force=selected_profile in {"v29_photo_variant_b_candidate", "unknown_or_degraded_photo"},
            )
            if full_page_vl_changed_fields or full_page_vl_route_changed_fields:
                _apply_chassis_vin_fix(fields)
                fields = postprocess_fields(fields)

        suppression_changed_fields = _apply_false_positive_suppression(fields, selected_profile)
        normalized_empty_fields = _normalize_optional_empty_fields(fields)
        if suppression_changed_fields or normalized_empty_fields:
            fields = postprocess_fields(fields)
            normalized_empty_fields = _normalize_optional_empty_fields(fields)

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
        if profile_route_applied:
            result["pipeline"]["profile_route_applied"] = profile_route_applied
            result["pipeline"]["profile_route_changed_fields"] = profile_route_changed_fields
        if right_page_vl_seconds is not None:
            result["pipeline"]["right_page_vl_seconds"] = right_page_vl_seconds
            result["pipeline"]["right_page_vl_changed_fields"] = right_page_vl_changed_fields
        if d_block_vl_seconds is not None:
            result["pipeline"]["d_block_vl_seconds"] = d_block_vl_seconds
            result["pipeline"]["d_block_vl_changed_fields"] = d_block_vl_changed_fields
        if full_page_vl_seconds is not None:
            result["pipeline"]["full_page_vl_second_pass_seconds"] = full_page_vl_seconds
            result["pipeline"]["full_page_vl_selected_pass"] = full_page_vl_selected_pass
            result["pipeline"]["full_page_vl_changed_fields"] = full_page_vl_changed_fields
            result["pipeline"]["full_page_vl_route_changed_fields"] = full_page_vl_route_changed_fields
        if full_page_vl_retry_seconds is not None:
            result["pipeline"]["full_page_vl_retry_seconds"] = full_page_vl_retry_seconds
        if suppression_changed_fields:
            result["pipeline"]["suppression_changed_fields"] = suppression_changed_fields
        if normalized_empty_fields:
            result["pipeline"]["normalized_empty_fields"] = normalized_empty_fields
        result["review"] = evaluate_extraction_review(fields, anchors)

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

