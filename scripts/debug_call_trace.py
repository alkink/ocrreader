from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocrreader.config import load_config


TRACK_METHODS = {
    "RuhsatOcrPipeline.__init__",
    "RuhsatOcrPipeline.process_path",
    "PaddleOCREngine.__init__",
    "PaddleOCREngine.iter_words",
    "PaddleOCREngine._predict",
    "PaddleOCREngine.read_text",
    "TesseractEngine.__init__",
    "TesseractEngine.iter_words",
    "TesseractEngine.read_text",
    "TemplateAnchorDetector.__init__",
}

TRACK_FUNCTIONS = {
    "load_config",
    "create_ocr_engine",
    "preprocess_document",
    "detect_document_quad",
    "detect_anchors_hybrid",
    "detect_anchors",
    "resolve_field_rois",
    "extract_fields",
    "locate_value_from_anchor",
    "locate_value_from_roi_words",
    "_apply_page_second_pass",
    "_apply_chassis_vin_fix",
    "postprocess_fields",
}


def _frame_name(frame: object) -> tuple[str, str, str]:
    mod = str(getattr(frame, "f_globals", {}).get("__name__", ""))
    code = getattr(frame, "f_code", None)
    fn = str(getattr(code, "co_name", ""))
    qual = str(getattr(code, "co_qualname", "") or "")

    # Prefer co_qualname for robust method detection (works even if `self` is not
    # yet populated in f_locals at call events).
    if qual and "<locals>" not in qual and "." in qual:
        return mod, fn, qual

    locals_dict = getattr(frame, "f_locals", {})
    self_obj = locals_dict.get("self") if isinstance(locals_dict, dict) else None
    if self_obj is not None:
        cls = type(self_obj).__name__
        return mod, fn, f"{cls}.{fn}"
    return mod, fn, fn


def _should_track(frame: object) -> bool:
    mod, fn, qn = _frame_name(frame)
    if not mod.startswith("ocrreader"):
        return False
    if qn in TRACK_METHODS:
        return True
    if fn in TRACK_FUNCTIONS:
        return True
    return False


def _fmt_call_args(frame: object) -> str:
    locals_dict = getattr(frame, "f_locals", {})
    if not isinstance(locals_dict, dict):
        return ""

    keys = ["path", "image_path", "psm", "min_conf"]
    parts: list[str] = []
    for key in keys:
        if key not in locals_dict:
            continue
        value = locals_dict.get(key)
        text = repr(value)
        if len(text) > 80:
            text = text[:77] + "..."
        parts.append(f"{key}={text}")
    return ", ".join(parts)


def _fmt_return(arg: object) -> str:
    if arg is None:
        return "None"
    if isinstance(arg, dict):
        return f"dict[{len(arg)}]"
    if isinstance(arg, list):
        return f"list[{len(arg)}]"
    if isinstance(arg, tuple):
        return f"tuple[{len(arg)}]"
    shape = getattr(arg, "shape", None)
    if shape is not None:
        return f"shape={tuple(shape)}"
    text = repr(arg)
    if len(text) > 80:
        text = text[:77] + "..."
    return text


def run() -> int:
    parser = argparse.ArgumentParser(description="Run OCR pipeline and dump nested call trace for presentation/debug.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--config", default="config/ruhsat_schema_paddle_v29.yaml", help="Config YAML path")
    parser.add_argument("--pipeline-module", default="ocrreader.pipeline", help="Module exporting RuhsatOcrPipeline")
    parser.add_argument("--output", default="output/result.json", help="Output JSON path")
    parser.add_argument("--trace-out", default="output/debug_call_trace.txt", help="Trace text output path")
    parser.add_argument("--debug-dir", default=None, help="Optional debug-dir for overlays")
    args = parser.parse_args()

    lines: list[str] = []
    active_stack: list[tuple[int, str, float]] = []

    def _prof(frame: object, event: str, arg: object):
        if event == "call":
            if not _should_track(frame):
                return _prof
            _, _, qn = _frame_name(frame)
            call_args = _fmt_call_args(frame)
            depth = len(active_stack)
            prefix = "  " * depth
            if call_args:
                lines.append(f"{prefix}CALL {qn}({call_args})")
            else:
                lines.append(f"{prefix}CALL {qn}()")
            active_stack.append((id(frame), qn, time.perf_counter()))
            return _prof

        if event == "return":
            if not active_stack:
                return _prof
            frame_id, qn, t0 = active_stack[-1]
            if frame_id != id(frame):
                return _prof
            active_stack.pop()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            depth = len(active_stack)
            prefix = "  " * depth
            ret = _fmt_return(arg)
            lines.append(f"{prefix}RET  {qn} -> {ret} [{dt_ms:.1f} ms]")
            return _prof

        return _prof

    sys.setprofile(_prof)
    try:
        cfg = load_config(args.config)
        pipeline_mod = importlib.import_module(args.pipeline_module)
        pipeline_cls = getattr(pipeline_mod, "RuhsatOcrPipeline")
        pipeline = pipeline_cls(cfg)
        result = pipeline.process_path(args.image, debug_dir=args.debug_dir)
    finally:
        sys.setprofile(None)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    trace_path = Path(args.trace_out)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    summary = [
        f"image={args.image}",
        f"config={args.config}",
        f"pipeline_module={args.pipeline_module}",
        f"anchors={len(result.get('anchors', {})) if isinstance(result, dict) else 0}",
        f"fields={len(result.get('fields', {})) if isinstance(result, dict) else 0}",
        "",
        "--- nested call trace ---",
    ]
    trace_path.write_text("\n".join(summary + lines) + "\n", encoding="utf-8")

    print(f"wrote_json={out_path.as_posix()}")
    print(f"wrote_trace={trace_path.as_posix()} lines={len(lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
