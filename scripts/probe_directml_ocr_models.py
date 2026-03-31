from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np


def _shape_list(items: list[Any]) -> list[Any]:
    return [getattr(item, "shape", None) for item in items]


def _session_probe(model_path: Path, feed: dict[str, np.ndarray], runs: int) -> dict[str, Any]:
    import onnxruntime as ort  # type: ignore

    t0 = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    init_seconds = time.perf_counter() - t0

    outputs = session.run(None, feed)

    latencies_ms: list[float] = []
    for _ in range(max(1, runs)):
        t1 = time.perf_counter()
        session.run(None, feed)
        latencies_ms.append((time.perf_counter() - t1) * 1000.0)

    return {
        "ok": True,
        "model": str(model_path),
        "providers": session.get_providers(),
        "inputs": [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in session.get_inputs()
        ],
        "outputs": [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in session.get_outputs()
        ],
        "output_shapes": _shape_list(outputs),
        "init_seconds": round(init_seconds, 4),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 3),
        "min_latency_ms": round(float(np.min(latencies_ms)), 3),
        "max_latency_ms": round(float(np.max(latencies_ms)), 3),
    }


def _capture(name: str, fn) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:
        return {"ok": False, "name": name, "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe exported OCR ONNX models with ONNX Runtime DirectML.")
    parser.add_argument(
        "--det-model",
        default="models/onnx/ppocrv4_mobile_det/inference.onnx",
        help="Detection ONNX path",
    )
    parser.add_argument(
        "--rec-model",
        default="models/onnx/en_ppocrv4_mobile_rec/inference.onnx",
        help="Recognition ONNX path",
    )
    parser.add_argument("--runs", type=int, default=5, help="Measured inference runs per model")
    parser.add_argument(
        "--summary-out",
        default="output/directml_ocr_model_probe.json",
        help="Where to write the JSON summary",
    )
    args = parser.parse_args()

    det_model = Path(args.det_model)
    rec_model = Path(args.rec_model)

    summary = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "det": _capture(
            "det",
            lambda: _session_probe(
                det_model,
                {"x": np.random.default_rng(123).random((1, 3, 960, 960), dtype=np.float32)},
                args.runs,
            ),
        ),
        "rec": _capture(
            "rec",
            lambda: _session_probe(
                rec_model,
                {"x": np.random.default_rng(456).random((1, 3, 48, 320), dtype=np.float32)},
                args.runs,
            ),
        ),
    }

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
