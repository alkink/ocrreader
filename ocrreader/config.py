from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DocumentDetectorConfig:
    min_area_ratio: float = 0.2
    canny_threshold1: int = 60
    canny_threshold2: int = 180


@dataclass(frozen=True)
class DeskewConfig:
    enabled: bool = True
    max_correction_deg: float = 8.0


@dataclass(frozen=True)
class PipelineConfig:
    output_width: int = 2200
    output_height: int = 1400
    orientation_osd_enabled: bool = False
    page_regex_fallback_enabled: bool = False
    document_detector: DocumentDetectorConfig = DocumentDetectorConfig()
    deskew: DeskewConfig = DeskewConfig()


@dataclass(frozen=True)
class OCRConfig:
    engine: str = "tesseract"
    executable: str | None = None
    language: str = "tur+eng"
    oem: int = 3
    psm: int = 6
    crop_engine: str | None = None
    crop_ocr_mode: str = "always"
    crop_ocr_variants: tuple[str, ...] = ("preprocessed", "preprocessed_alt", "raw")
    crop_ocr_skip_margin: int = 0
    onnx_det_model: str | None = None
    onnx_rec_model: str | None = None
    onnx_providers: tuple[str, ...] = ()
    onnx_det_limit_side_len: int | None = None
    onnx_rec_batch_size: int | None = None
    paddle_device: str = "gpu"
    paddle_cpu_threads: int = 1
    paddle_enable_mkldnn: bool = False
    paddle_text_det_limit_side_len: int | None = None
    paddle_text_recognition_batch_size: int | None = None
    # Hybrid mode: when set to "tesseract", crop-level read_text() calls use
    # Tesseract instead of PaddleOCR.  PaddleOCR is kept for iter_words (full
    # page detection) which is its strength; Tesseract handles small, already-
    # preprocessed field crops much faster with no per-call GPU round-trip.
    paddle_crop_engine: str | None = None
    paddle_vl_use_layout_detection: bool = True
    paddle_vl_use_ocr_for_image_block: bool = True
    paddle_vl_runtime_profile: str = "auto"
    paddle_vl_service_url: str | None = None
    paddle_vl_service_model_name: str | None = None
    paddle_vl_service_api_key: str | None = None
    paddle_vl_service_max_concurrency: int | None = None
    paddle_vl_max_side: int | None = None
    paddle_vl_min_pixels: int | None = None
    paddle_vl_max_pixels: int | None = None
    paddle_vl_max_new_tokens: int | None = 512
    paddle_vl_use_cache: bool = True
    paddle_vl_use_queues: bool = True
    paddle_vl_prompt_label: str | None = None
    glm_fallback_enabled: bool = False
    glm_fallback_fields: tuple[str, ...] = ()
    glm_fallback_min_confidence: int = 10
    glm_api_key: str | None = None
    glm_api_url: str | None = None
    glm_model: str | None = None
    glm_mode: str | None = None
    glm_timeout: int | None = None
    glm_enable_layout: bool = False
    glm_env_file: str | None = None
    glm_ocr_api_host: str | None = None
    glm_ocr_api_port: int | None = None
    glm_api_path: str | None = None
    glm_api_mode: str | None = None
    paddle_version: str | None = None


@dataclass(frozen=True)
class AnchorConfig:
    aliases: list[str]
    min_score: float = 0.72
    search_region_norm: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class FieldConfig:
    anchor: str | None = None
    offset_from_anchor_norm: tuple[float, float, float, float] | None = None
    fallback_norm: tuple[float, float, float, float] | None = None
    value_from_anchor: str = "below"
    value_margin_norm: tuple[float, float, float, float] | None = None
    prefer_anchor: bool = True
    force_method: str | None = None
    cleanup: str = "keep"
    min_len: int = 0
    prefer_mixed_alnum: bool = False
    strip_prefixes: tuple[str, ...] = ()
    confidence_threshold: int = 0
    psm: int | None = None
    whitelist: str | None = None


def _str_tuple(value: object, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list[str], got: {value}")
    return tuple(str(v) for v in value)


@dataclass(frozen=True)
class RuhsatConfig:
    pipeline: PipelineConfig
    ocr: OCRConfig
    anchors: dict[str, AnchorConfig]
    fields: dict[str, FieldConfig]


def _tuple4(value: object, name: str) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must be a list[4], got: {value}")
    return tuple(float(v) for v in value)


def load_config(path: str) -> RuhsatConfig:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    pipeline_raw = raw.get("pipeline", {})
    detector_raw = pipeline_raw.get("document_detector", {})
    deskew_raw = pipeline_raw.get("deskew", {})

    pipeline = PipelineConfig(
        output_width=int(pipeline_raw.get("output_width", 2200)),
        output_height=int(pipeline_raw.get("output_height", 1400)),
        orientation_osd_enabled=bool(pipeline_raw.get("orientation_osd_enabled", False)),
        page_regex_fallback_enabled=bool(pipeline_raw.get("page_regex_fallback_enabled", False)),
        document_detector=DocumentDetectorConfig(
            min_area_ratio=float(detector_raw.get("min_area_ratio", 0.2)),
            canny_threshold1=int(detector_raw.get("canny_threshold1", 60)),
            canny_threshold2=int(detector_raw.get("canny_threshold2", 180)),
        ),
        deskew=DeskewConfig(
            enabled=bool(deskew_raw.get("enabled", True)),
            max_correction_deg=float(deskew_raw.get("max_correction_deg", 8.0)),
        ),
    )

    ocr_raw = raw.get("ocr", {})
    ocr = OCRConfig(
        engine=str(ocr_raw.get("engine", "tesseract")).strip().lower(),
        executable=ocr_raw.get("executable"),
        language=str(ocr_raw.get("language", "tur+eng")),
        oem=int(ocr_raw.get("oem", 3)),
        psm=int(ocr_raw.get("psm", 6)),
        crop_engine=(
            str(ocr_raw["crop_engine"]).strip().lower()
            if ocr_raw.get("crop_engine") is not None
            else None
        ),
        crop_ocr_mode=str(ocr_raw.get("crop_ocr_mode", "always")).strip().lower(),
        crop_ocr_variants=_str_tuple(ocr_raw.get("crop_ocr_variants"), "ocr.crop_ocr_variants")
        or ("preprocessed", "preprocessed_alt", "raw"),
        crop_ocr_skip_margin=int(ocr_raw.get("crop_ocr_skip_margin", 0) or 0),
        onnx_det_model=(
            str(ocr_raw["onnx_det_model"]).strip()
            if ocr_raw.get("onnx_det_model") is not None
            else None
        ),
        onnx_rec_model=(
            str(ocr_raw["onnx_rec_model"]).strip()
            if ocr_raw.get("onnx_rec_model") is not None
            else None
        ),
        onnx_providers=_str_tuple(ocr_raw.get("onnx_providers"), "ocr.onnx_providers"),
        onnx_det_limit_side_len=(
            int(ocr_raw["onnx_det_limit_side_len"])
            if ocr_raw.get("onnx_det_limit_side_len") is not None
            else None
        ),
        onnx_rec_batch_size=(
            int(ocr_raw["onnx_rec_batch_size"])
            if ocr_raw.get("onnx_rec_batch_size") is not None
            else None
        ),
        paddle_device=str(ocr_raw.get("paddle_device", "gpu")).strip().lower(),
        paddle_cpu_threads=int(ocr_raw.get("paddle_cpu_threads", 1) or 1),
        paddle_enable_mkldnn=bool(ocr_raw.get("paddle_enable_mkldnn", False)),
        paddle_text_det_limit_side_len=(
            int(ocr_raw["paddle_text_det_limit_side_len"])
            if ocr_raw.get("paddle_text_det_limit_side_len") is not None
            else None
        ),
        paddle_text_recognition_batch_size=(
            int(ocr_raw["paddle_text_recognition_batch_size"])
            if ocr_raw.get("paddle_text_recognition_batch_size") is not None
            else None
        ),
        paddle_crop_engine=(
            str(ocr_raw["paddle_crop_engine"]).strip().lower()
            if ocr_raw.get("paddle_crop_engine") is not None
            else None
        ),
        paddle_vl_use_layout_detection=bool(ocr_raw.get("paddle_vl_use_layout_detection", True)),
        paddle_vl_use_ocr_for_image_block=bool(ocr_raw.get("paddle_vl_use_ocr_for_image_block", True)),
        paddle_vl_runtime_profile=str(ocr_raw.get("paddle_vl_runtime_profile", "auto")).strip().lower(),
        paddle_vl_service_url=(str(ocr_raw.get("paddle_vl_service_url")).strip() if ocr_raw.get("paddle_vl_service_url") is not None else None),
        paddle_vl_service_model_name=(str(ocr_raw.get("paddle_vl_service_model_name")).strip() if ocr_raw.get("paddle_vl_service_model_name") is not None else None),
        paddle_vl_service_api_key=(str(ocr_raw.get("paddle_vl_service_api_key")).strip() if ocr_raw.get("paddle_vl_service_api_key") is not None else None),
        paddle_vl_service_max_concurrency=(int(ocr_raw["paddle_vl_service_max_concurrency"]) if ocr_raw.get("paddle_vl_service_max_concurrency") is not None else None),
        paddle_vl_max_side=(int(ocr_raw["paddle_vl_max_side"]) if ocr_raw.get("paddle_vl_max_side") is not None else None),
        paddle_vl_min_pixels=(int(ocr_raw["paddle_vl_min_pixels"]) if ocr_raw.get("paddle_vl_min_pixels") is not None else None),
        paddle_vl_max_pixels=(int(ocr_raw["paddle_vl_max_pixels"]) if ocr_raw.get("paddle_vl_max_pixels") is not None else None),
        paddle_vl_max_new_tokens=(int(ocr_raw["paddle_vl_max_new_tokens"]) if ocr_raw.get("paddle_vl_max_new_tokens") is not None else 512),
        paddle_vl_use_cache=bool(ocr_raw.get("paddle_vl_use_cache", True)),
        paddle_vl_use_queues=bool(ocr_raw.get("paddle_vl_use_queues", True)),
        paddle_vl_prompt_label=(str(ocr_raw.get("paddle_vl_prompt_label")).strip() if ocr_raw.get("paddle_vl_prompt_label") is not None else None),
        paddle_version=ocr_raw.get("paddle_version"),
        glm_fallback_enabled=bool(ocr_raw.get("glm_fallback_enabled", False)),
        glm_fallback_fields=_str_tuple(ocr_raw.get("glm_fallback_fields"), "ocr.glm_fallback_fields"),
        glm_fallback_min_confidence=int(ocr_raw.get("glm_fallback_min_confidence", 10) or 10),
        glm_api_key=(str(ocr_raw.get("glm_api_key")).strip() if ocr_raw.get("glm_api_key") is not None else None),
        glm_api_url=(str(ocr_raw.get("glm_api_url")).strip() if ocr_raw.get("glm_api_url") is not None else None),
        glm_model=(str(ocr_raw.get("glm_model")).strip() if ocr_raw.get("glm_model") is not None else None),
        glm_mode=(str(ocr_raw.get("glm_mode")).strip().lower() if ocr_raw.get("glm_mode") is not None else None),
        glm_timeout=(int(ocr_raw["glm_timeout"]) if ocr_raw.get("glm_timeout") is not None else None),
        glm_enable_layout=bool(ocr_raw.get("glm_enable_layout", False)),
        glm_env_file=(str(ocr_raw.get("glm_env_file")).strip() if ocr_raw.get("glm_env_file") is not None else None),
        glm_ocr_api_host=(str(ocr_raw.get("glm_ocr_api_host")).strip() if ocr_raw.get("glm_ocr_api_host") is not None else None),
        glm_ocr_api_port=(int(ocr_raw["glm_ocr_api_port"]) if ocr_raw.get("glm_ocr_api_port") is not None else None),
        glm_api_path=(str(ocr_raw.get("glm_api_path")).strip() if ocr_raw.get("glm_api_path") is not None else None),
        glm_api_mode=(str(ocr_raw.get("glm_api_mode")).strip().lower() if ocr_raw.get("glm_api_mode") is not None else None),
    )

    anchors_raw = raw.get("anchors", {})
    anchors: dict[str, AnchorConfig] = {}
    for name, item in anchors_raw.items():
        item = item or {}
        aliases = [str(a) for a in item.get("aliases", [name])]
        anchors[name] = AnchorConfig(
            aliases=aliases,
            min_score=float(item.get("min_score", 0.72)),
            search_region_norm=_tuple4(item.get("search_region_norm"), f"anchors.{name}.search_region_norm"),
        )

    fields_raw = raw.get("fields", {})
    fields: dict[str, FieldConfig] = {}
    for name, item in fields_raw.items():
        item = item or {}
        fields[name] = FieldConfig(
            anchor=item.get("anchor"),
            offset_from_anchor_norm=_tuple4(item.get("offset_from_anchor_norm"), f"fields.{name}.offset_from_anchor_norm"),
            fallback_norm=_tuple4(item.get("fallback_norm"), f"fields.{name}.fallback_norm"),
            value_from_anchor=str(item.get("value_from_anchor", "below")),
            value_margin_norm=_tuple4(item.get("value_margin_norm"), f"fields.{name}.value_margin_norm"),
            prefer_anchor=bool(item.get("prefer_anchor", True)),
            force_method=str(item.get("force_method")) if item.get("force_method") is not None else None,
            cleanup=str(item.get("cleanup", "keep")),
            min_len=int(item.get("min_len", 0)),
            prefer_mixed_alnum=bool(item.get("prefer_mixed_alnum", False)),
            strip_prefixes=_str_tuple(item.get("strip_prefixes"), f"fields.{name}.strip_prefixes"),
            confidence_threshold=int(item.get("confidence_threshold", 0)),
            psm=int(item["psm"]) if item.get("psm") is not None else None,
            whitelist=item.get("whitelist"),
        )

    return RuhsatConfig(
        pipeline=pipeline,
        ocr=ocr,
        anchors=anchors,
        fields=fields,
    )

