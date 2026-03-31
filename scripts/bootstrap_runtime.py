from __future__ import annotations

import argparse
import platform
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_capture(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=8,
            check=False,
        )
    except Exception:
        return ""
    chunks = [str(proc.stdout or "").strip(), str(proc.stderr or "").strip()]
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def detect_gpu_inventory_text() -> str:
    system_name = platform.system().strip().lower()
    outputs: list[str] = []
    if system_name == "windows":
        outputs.append(
            run_capture(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join \"`n\"",
                ]
            )
        )
        outputs.append(run_capture(["wmic", "path", "win32_videocontroller", "get", "name"]))
    elif system_name == "linux":
        if shutil.which("lspci"):
            outputs.append(run_capture(["lspci"]))
        if shutil.which("rocminfo"):
            outputs.append(run_capture(["rocminfo"]))
        if shutil.which("sycl-ls"):
            outputs.append(run_capture(["sycl-ls"]))
    elif system_name == "darwin":
        outputs.append(run_capture(["system_profiler", "SPDisplaysDataType"]))
    return "\n".join(part for part in outputs if part).lower()


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def build_plan() -> dict[str, object]:
    system_name = platform.system().strip().lower()
    machine = platform.machine().strip().lower()
    gpu_inventory = detect_gpu_inventory_text()
    has_nvidia = command_exists("nvidia-smi") or "nvidia" in gpu_inventory or "geforce" in gpu_inventory
    has_intel_arc = "intel" in gpu_inventory and "arc" in gpu_inventory
    has_gpu_hint = any(token in gpu_inventory for token in ("nvidia", "geforce", "amd", "radeon", "intel", "arc", "apple"))

    classic_auto = "paddleocr"
    vl_auto = "native_paddle_vl"
    requirement_files = [REPO_ROOT / "requirements.txt"]
    manual_steps: list[str] = []
    notes: list[str] = []

    if system_name == "darwin":
        requirement_files.append(REPO_ROOT / "requirements.cpu-paddle.txt")
        requirement_files.append(REPO_ROOT / "requirements.macos-onnx.txt")
        classic_auto = "onnxruntime (if CoreMLExecutionProvider is available) else paddleocr"
        vl_auto = "local_mlx_vlm_service"
        manual_steps.append("If you want PaddleOCR-VL auto mode on macOS, start a local MLX-VLM compatible server first.")
        notes.append("macOS PaddlePaddle path is CPU-oriented. ONNX Runtime may improve classic OCR, but CoreML provider availability depends on the wheel/build on that Mac.")
    elif system_name == "windows":
        if has_nvidia:
            classic_auto = "paddleocr"
            vl_auto = "native_paddle_vl"
            manual_steps.append("Install the correct official paddlepaddle-gpu wheel for that machine's CUDA/runtime. requirements.txt does not choose it automatically.")
        else:
            requirement_files.append(REPO_ROOT / "requirements.cpu-paddle.txt")
            if has_gpu_hint:
                requirement_files.append(REPO_ROOT / "requirements.windows-directml.txt")
                classic_auto = "onnxruntime"
            else:
                classic_auto = "paddleocr"
            vl_auto = "local_vllm_service" if has_intel_arc else "native_paddle_vl"
            if has_intel_arc:
                manual_steps.append("If you want PaddleOCR-VL auto mode on Intel Arc, start a local vLLM-compatible server first.")
            notes.append("Windows non-CUDA GPU classic OCR uses DirectML when available.")
    else:
        if has_nvidia:
            classic_auto = "paddleocr"
            vl_auto = "native_paddle_vl"
            manual_steps.append("Install the matching Linux paddlepaddle-gpu wheel for the target machine if you want CUDA acceleration.")
        else:
            requirement_files.append(REPO_ROOT / "requirements.cpu-paddle.txt")
            classic_auto = "paddleocr"
            vl_auto = "native_paddle_vl"
        notes.append("Linux/WSL installs keep native Paddle as the main OCR path unless you explicitly add another backend.")

    if not command_exists("tesseract"):
        manual_steps.append("Install Tesseract separately if you will use crop OCR or classic configs that rely on it.")

    return {
        "system": system_name,
        "machine": machine,
        "gpu_inventory": gpu_inventory.strip(),
        "has_nvidia": has_nvidia,
        "has_intel_arc": has_intel_arc,
        "requirement_files": requirement_files,
        "classic_auto": classic_auto,
        "vl_auto": vl_auto,
        "manual_steps": manual_steps,
        "notes": notes,
    }


def install_requirements(requirement_files: list[Path]) -> None:
    for path in requirement_files:
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(path)]
        print(f"[install] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap this repo for the current machine.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the install plan.")
    args = parser.parse_args()

    plan = build_plan()
    requirement_files: list[Path] = list(plan["requirement_files"])  # type: ignore[assignment]

    print("Bootstrap plan")
    print(f"- system: {plan['system']}")
    print(f"- machine: {plan['machine']}")
    print(f"- classic auto engine: {plan['classic_auto']}")
    print(f"- vl auto runtime: {plan['vl_auto']}")
    print("- requirement files:")
    for path in requirement_files:
        print(f"  - {path.relative_to(REPO_ROOT)}")

    gpu_inventory = str(plan.get("gpu_inventory") or "").strip()
    if gpu_inventory:
        print("- detected gpu inventory:")
        for line in gpu_inventory.splitlines():
            if line.strip():
                print(f"  - {line.strip()}")

    notes: list[str] = list(plan["notes"])  # type: ignore[assignment]
    if notes:
        print("- notes:")
        for item in notes:
            print(f"  - {item}")

    manual_steps: list[str] = list(plan["manual_steps"])  # type: ignore[assignment]
    if manual_steps:
        print("- manual steps still required:")
        for item in manual_steps:
            print(f"  - {item}")

    if args.dry_run:
        return 0

    install_requirements(requirement_files)
    print("Bootstrap install completed.")
    if manual_steps:
        print("Remaining manual steps:")
        for item in manual_steps:
            print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
