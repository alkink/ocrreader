from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _build_model(path: Path) -> None:
    rng = np.random.default_rng(1234)

    input_shape = [1, 3, 640, 640]
    output_shape = [1, 32, 160, 160]

    initializers = [
        numpy_helper.from_array(rng.standard_normal((16, 3, 3, 3), dtype=np.float32) * 0.05, name="w1"),
        numpy_helper.from_array(rng.standard_normal((16,), dtype=np.float32) * 0.05, name="b1"),
        numpy_helper.from_array(rng.standard_normal((32, 16, 3, 3), dtype=np.float32) * 0.05, name="w2"),
        numpy_helper.from_array(rng.standard_normal((32,), dtype=np.float32) * 0.05, name="b2"),
    ]

    nodes = [
        helper.make_node("Conv", ["input", "w1", "b1"], ["x1"], pads=[1, 1, 1, 1], strides=[2, 2]),
        helper.make_node("Relu", ["x1"], ["x2"]),
        helper.make_node("MaxPool", ["x2"], ["x3"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Conv", ["x3", "w2", "b2"], ["x4"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["x4"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes,
        "provider_microbench",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        initializer=initializers,
    )
    model = helper.make_model(graph, producer_name="ocrreader")
    model.opset_import[0].version = 13
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _bench_onnxruntime(model_path: Path, provider: str, runs: int, warmup: int) -> dict[str, Any]:
    import onnxruntime as ort  # type: ignore

    sample = np.random.default_rng(42).standard_normal((1, 3, 640, 640), dtype=np.float32)
    t0 = time.perf_counter()
    session = ort.InferenceSession(str(model_path), providers=[provider])
    init_seconds = time.perf_counter() - t0

    input_name = session.get_inputs()[0].name
    actual_providers = session.get_providers()

    for _ in range(warmup):
        session.run(None, {input_name: sample})

    latencies_ms: list[float] = []
    for _ in range(runs):
        t1 = time.perf_counter()
        session.run(None, {input_name: sample})
        latencies_ms.append((time.perf_counter() - t1) * 1000.0)

    return {
        "ok": True,
        "provider": provider,
        "actual_providers": actual_providers,
        "init_seconds": round(init_seconds, 4),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 3),
        "min_latency_ms": round(float(np.min(latencies_ms)), 3),
        "max_latency_ms": round(float(np.max(latencies_ms)), 3),
    }


def _bench_openvino(model_path: Path, device_name: str, runs: int, warmup: int) -> dict[str, Any]:
    import openvino as ov  # type: ignore

    sample = np.random.default_rng(42).standard_normal((1, 3, 640, 640), dtype=np.float32)
    core = ov.Core()

    t0 = time.perf_counter()
    model = core.read_model(str(model_path))
    compiled = core.compile_model(model, device_name)
    init_seconds = time.perf_counter() - t0

    input_name = compiled.inputs[0].any_name
    output_name = compiled.outputs[0].any_name

    for _ in range(warmup):
        compiled({input_name: sample})

    latencies_ms: list[float] = []
    for _ in range(runs):
        t1 = time.perf_counter()
        compiled({input_name: sample})[output_name]
        latencies_ms.append((time.perf_counter() - t1) * 1000.0)

    return {
        "ok": True,
        "device": device_name,
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
    parser = argparse.ArgumentParser(description="Benchmark non-CUDA runtime providers on a tiny ONNX model.")
    parser.add_argument("--runs", type=int, default=10, help="Number of measured runs per provider.")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs per provider.")
    parser.add_argument(
        "--summary-out",
        default="output/runtime_provider_benchmark.json",
        help="Where to write the JSON summary.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="ocrreader_bench_") as tmpdir:
        model_path = Path(tmpdir) / "provider_microbench.onnx"
        _build_model(model_path)

        results = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "runs": args.runs,
            "warmup": args.warmup,
            "model_path": str(model_path),
            "onnxruntime": {
                "cpu": _capture(
                    "ort_cpu",
                    lambda: _bench_onnxruntime(model_path, "CPUExecutionProvider", args.runs, args.warmup),
                ),
                "directml": _capture(
                    "ort_dml",
                    lambda: _bench_onnxruntime(model_path, "DmlExecutionProvider", args.runs, args.warmup),
                ),
            },
            "openvino": {
                "cpu": _capture(
                    "ov_cpu",
                    lambda: _bench_openvino(model_path, "CPU", args.runs, args.warmup),
                ),
                "gpu": _capture(
                    "ov_gpu",
                    lambda: _bench_openvino(model_path, "GPU", args.runs, args.warmup),
                ),
            },
        }

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
