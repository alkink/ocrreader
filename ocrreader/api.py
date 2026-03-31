from __future__ import annotations

import argparse
import platform
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import uvicorn

from .config import RuhsatConfig, load_config
from .io_utils import imdecode_color
from .pipeline import RuhsatOcrPipeline


def _fields_only_result(result: dict[str, object]) -> dict[str, object]:
    fields = result.get("fields", {})
    if not isinstance(fields, dict):
        return {}

    flat: dict[str, object] = {}
    for field_name, entry in fields.items():
        if isinstance(entry, dict):
            flat[field_name] = entry.get("value")
        else:
            flat[field_name] = entry
    return flat


def _runtime_info(config: RuhsatConfig, pipeline: RuhsatOcrPipeline | None = None) -> dict[str, object]:
    info: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ocr_engine": config.ocr.engine,
        "paddle_device_requested": config.ocr.paddle_device,
        "crop_ocr_mode": config.ocr.crop_ocr_mode,
    }
    runtime_engine = getattr(pipeline, "engine", None) if pipeline is not None else None
    primary_engine = runtime_engine
    if runtime_engine is not None:
        info["ocr_engine_class"] = type(runtime_engine).__name__
        wrapped = getattr(runtime_engine, "_primary", None)
        if wrapped is not None:
            primary_engine = wrapped
            info["ocr_primary_class"] = type(wrapped).__name__
            crop_engine = getattr(runtime_engine, "_tess", None)
            if crop_engine is not None:
                info["ocr_crop_class"] = type(crop_engine).__name__
    if primary_engine is not None:
        providers = getattr(primary_engine, "_providers", None)
        if providers:
            info["onnx_providers"] = list(providers)
        det_model = getattr(primary_engine, "_det_model_path", None)
        if det_model is not None:
            info["onnx_det_model"] = str(det_model)
        rec_model = getattr(primary_engine, "_rec_model_path", None)
        if rec_model is not None:
            info["onnx_rec_model"] = str(rec_model)
    try:
        import paddle  # type: ignore

        info["paddle_version"] = getattr(paddle, "__version__", "unknown")
        info["paddle_cuda_compiled"] = bool(paddle.device.is_compiled_with_cuda())
        try:
            info["paddle_device"] = paddle.device.get_device()
        except Exception as exc:
            info["paddle_device_error"] = repr(exc)
    except Exception as exc:
        info["paddle_import_error"] = repr(exc)
    return info


class WarmOcrService:
    def __init__(self, config_path: str, debug_root: str | None = None):
        self.config_path = config_path
        self.debug_root = Path(debug_root) if debug_root else None
        self.config = load_config(config_path)
        self.pipeline = RuhsatOcrPipeline(self.config)
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.request_count = 0
        self.total_latency_ms = 0.0

    def _debug_dir(self, request_id: str, enabled: bool) -> str | None:
        if not enabled or self.debug_root is None:
            return None
        return str(self.debug_root / request_id)

    def process_upload(
        self,
        payload: bytes,
        filename: str,
        *,
        full_output: bool = False,
        save_debug: bool = False,
    ) -> dict[str, object]:
        if not payload:
            raise HTTPException(status_code=400, detail="Empty file payload")

        try:
            image = imdecode_color(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc

        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        debug_dir = self._debug_dir(request_id, save_debug)

        with self.lock:
            result = self.pipeline.process_image(
                image,
                image_label=filename or f"upload:{request_id}",
                debug_dir=debug_dir,
            )

        latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
        self.request_count += 1
        self.total_latency_ms += latency_ms

        payload_out = result if full_output else _fields_only_result(result)
        return {
            "request_id": request_id,
            "latency_ms": latency_ms,
            "debug_dir": debug_dir,
            "result": payload_out,
        }

    def metrics(self) -> dict[str, object]:
        avg = (self.total_latency_ms / self.request_count) if self.request_count else 0.0
        return {
            "config_path": self.config_path,
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "request_count": self.request_count,
            "avg_latency_ms": round(avg, 1),
            "runtime": _runtime_info(self.config, self.pipeline),
        }


def create_app(config_path: str, debug_root: str | None = None) -> FastAPI:
    service_box: dict[str, WarmOcrService] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service_box["svc"] = WarmOcrService(config_path=config_path, debug_root=debug_root)
        yield
        service_box.clear()

    app = FastAPI(title="OCR Reader API", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, object]:
        svc = service_box["svc"]
        return {
            "status": "ok",
            "service": "ocrreader",
            "metrics": svc.metrics(),
        }

    @app.get("/metrics")
    def metrics() -> dict[str, object]:
        return service_box["svc"].metrics()

    @app.post("/ocr")
    async def ocr(
        image: UploadFile = File(...),
        full_output: bool = Form(False),
        save_debug: bool = Form(False),
    ) -> dict[str, object]:
        svc = service_box["svc"]
        payload = await image.read()
        return svc.process_upload(
            payload,
            image.filename or "upload.bin",
            full_output=full_output,
            save_debug=save_debug,
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warm OCR API service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default="config/ruhsat_schema_paddle_v29_gpu_lazy.yaml")
    parser.add_argument("--debug-root", default="output/api_debug")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = create_app(config_path=args.config, debug_root=args.debug_root)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
