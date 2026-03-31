from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _trim(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n...[truncated]..."


def _module_info(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"available": False}

    out: dict[str, Any] = {"available": True, "origin": spec.origin}
    try:
        mod = __import__(name)
        out["version"] = getattr(mod, "__version__", "unknown")
    except Exception as exc:
        out["import_error"] = repr(exc)
    return out


def _paddlex_link_info() -> dict[str, Any]:
    spec = importlib.util.find_spec("paddlex")
    if spec is None or spec.origin is None:
        return {"available": False}

    root = Path(spec.origin).resolve().parent
    out: dict[str, Any] = {"available": True, "root": str(root)}
    for filename in ("hpip_links.html", "hpip_links_cu12.html"):
        path = root / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        hrefs = re.findall(r'href="([^"]+)"', text)
        out[filename] = {
            "path": str(path),
            "links": hrefs,
            "windows_links": [link for link in hrefs if "win" in link.lower()],
            "linux_links": [link for link in hrefs if "linux" in link.lower()],
        }
    return out


def _run_command(args: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "cmd": args}

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "cmd": args,
        "stdout": _trim(proc.stdout),
        "stderr": _trim(proc.stderr),
    }


def _paddle_hpi_probe() -> dict[str, Any]:
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}

    try:
        PaddleOCR(
            lang="en",
            ocr_version="PP-OCRv4",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_hpi=True,
            device="cpu",
        )
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def _onnxruntime_probe() -> dict[str, Any]:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    out: dict[str, Any] = {
        "available": True,
        "version": getattr(ort, "__version__", "unknown"),
    }
    try:
        out["providers"] = ort.get_available_providers()
    except Exception as exc:
        out["providers_error"] = repr(exc)
    return out


def _openvino_probe() -> dict[str, Any]:
    try:
        import openvino as ov  # type: ignore
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    out: dict[str, Any] = {
        "available": True,
        "version": getattr(ov, "__version__", "unknown"),
    }

    try:
        core = ov.Core()
        devices: list[dict[str, Any]] = []
        for device_name in core.available_devices:
            item: dict[str, Any] = {"name": device_name}
            try:
                item["full_name"] = core.get_property(device_name, "FULL_DEVICE_NAME")
            except Exception as exc:
                item["full_name_error"] = repr(exc)
            devices.append(item)
        out["devices"] = devices
    except Exception as exc:
        out["device_error"] = repr(exc)

    return out


def _video_controller_probe() -> list[dict[str, Any]]:
    if platform.system() != "Windows":
        return []

    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,DriverVersion,PNPDeviceID | ConvertTo-Json -Compress",
    ]
    result = _run_command(cmd)
    if not result.get("ok"):
        return [{"error": result.get("stderr") or result.get("stdout") or "query_failed"}]

    raw = str(result.get("stdout") or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [{"raw": raw}]
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return [{"raw": raw}]


def _amd_probe() -> dict[str, Any]:
    return {
        "hipinfo": shutil.which("hipInfo.exe") or shutil.which("hipinfo"),
        "rocminfo": shutil.which("rocminfo.exe") or shutil.which("rocminfo"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe alternative OCR backend migration paths.")
    parser.add_argument(
        "--attempt-installs",
        action="store_true",
        help="Attempt official PaddleX install commands for hpi-cpu, hpi-gpu, and paddle2onnx.",
    )
    parser.add_argument(
        "--summary-out",
        default="output/backend_migration_probe.json",
        help="Where to write the JSON summary.",
    )
    args = parser.parse_args()

    summary: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "modules": {
            "paddle": _module_info("paddle"),
            "paddleocr": _module_info("paddleocr"),
            "paddlex": _module_info("paddlex"),
            "onnx": _module_info("onnx"),
            "onnxruntime": _module_info("onnxruntime"),
            "openvino": _module_info("openvino"),
            "ultra_infer": _module_info("ultra_infer"),
            "paddle2onnx": _module_info("paddle2onnx"),
        },
        "video_controllers": _video_controller_probe(),
        "amd_tools": _amd_probe(),
        "paddlex_hpi_links": _paddlex_link_info(),
        "paddle_hpi_probe": _paddle_hpi_probe(),
        "onnxruntime_probe": _onnxruntime_probe(),
        "openvino_probe": _openvino_probe(),
    }

    if args.attempt_installs:
        # Subprocesses inherit current environment; keep the model-source skip enabled.
        install_cmds = {
            "hpi_cpu": [sys.executable, "-m", "paddlex", "--install", "hpi-cpu", "-y"],
            "hpi_gpu": [sys.executable, "-m", "paddlex", "--install", "hpi-gpu", "-y"],
            "paddle2onnx": [sys.executable, "-m", "paddlex", "--install", "paddle2onnx", "-y"],
        }
        summary["install_attempts"] = {}
        for name, cmd in install_cmds.items():
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=240,
                check=False,
                env={**os.environ, "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"},
            )
            summary["install_attempts"][name] = {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "cmd": cmd,
                "stdout": _trim(proc.stdout),
                "stderr": _trim(proc.stderr),
            }

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
