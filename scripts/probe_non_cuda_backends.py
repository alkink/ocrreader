from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import cv2


def _try_import_onnxruntime() -> dict[str, Any]:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"available": False, "error": repr(exc)}

    out: dict[str, Any] = {
        "available": True,
        "version": getattr(ort, "__version__", "unknown"),
    }
    try:
        out["providers"] = ort.get_available_providers()
    except Exception as exc:  # pragma: no cover - diagnostic only
        out["providers_error"] = repr(exc)
    return out


def main() -> int:
    summary = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "opencv_opencl_have": bool(cv2.ocl.haveOpenCL()),
        "opencv_opencl_use": bool(cv2.ocl.useOpenCL()),
        "onnxruntime": _try_import_onnxruntime(),
    }

    out = Path("output/non_cuda_backend_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
