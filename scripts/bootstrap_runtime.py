from __future__ import annotations

import argparse
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_VENV_DIR = REPO_ROOT / ".venv-mlx"


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


def iter_python_candidates() -> Iterable[str]:
    seen: set[str] = set()
    candidates = [
        sys.executable,
        shutil.which("python3.12"),
        shutil.which("python3.11"),
        shutil.which("python3.10"),
        shutil.which("python3"),
        shutil.which("python"),
        "/opt/homebrew/bin/python3.12",
        "/opt/homebrew/bin/python3.11",
        "/opt/homebrew/bin/python3.10",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        text = str(candidate).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        yield text


def python_version_of(executable: str) -> tuple[int, int, int] | None:
    raw = run_capture(
        [
            executable,
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')",
        ]
    ).splitlines()
    if not raw:
        return None
    try:
        parts = tuple(int(x) for x in raw[-1].strip().split("."))
    except Exception:
        return None
    if len(parts) != 3:
        return None
    return parts  # type: ignore[return-value]


def find_python_at_least(minimum: tuple[int, int] = (3, 10)) -> str | None:
    for executable in iter_python_candidates():
        version = python_version_of(executable)
        if version and version[:2] >= minimum:
            return executable
    return None


def venv_python_path(venv_dir: Path) -> Path:
    if platform.system().strip().lower() == "windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_mlx_server_environment(*, dry_run: bool = False) -> tuple[Path | None, str | None]:
    python_executable = find_python_at_least((3, 10))
    if not python_executable:
        return None, None

    venv_python = venv_python_path(MLX_VENV_DIR)
    if dry_run:
        return venv_python, python_executable

    if not (MLX_VENV_DIR / "pyvenv.cfg").exists():
        create_cmd = [python_executable, "-m", "venv", str(MLX_VENV_DIR)]
        print(f"[mlx-setup] {' '.join(create_cmd)}")
        subprocess.run(create_cmd, check=True)

    upgrade_cmd = [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"]
    install_cmd = [str(venv_python), "-m", "pip", "install", "-r", str(REPO_ROOT / "requirements.mlx-vlm.txt")]
    print(f"[mlx-setup] {' '.join(upgrade_cmd)}")
    subprocess.run(upgrade_cmd, check=True)
    print(f"[mlx-setup] {' '.join(install_cmd)}")
    subprocess.run(install_cmd, check=True)
    return venv_python, python_executable


def build_plan(include_glmocr: bool = False) -> dict[str, object]:
    system_name = platform.system().strip().lower()
    machine = platform.machine().strip().lower()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    gpu_inventory = detect_gpu_inventory_text()
    has_nvidia = command_exists("nvidia-smi") or "nvidia" in gpu_inventory or "geforce" in gpu_inventory
    has_intel_arc = "intel" in gpu_inventory and "arc" in gpu_inventory
    has_gpu_hint = any(token in gpu_inventory for token in ("nvidia", "geforce", "amd", "radeon", "intel", "arc", "apple"))

    classic_auto = "paddleocr"
    vl_auto = "native_paddle_vl"
    requirement_files = [REPO_ROOT / "requirements.txt"]
    manual_steps: list[str] = []
    notes: list[str] = []
    vl_runtime_setup: dict[str, object] | None = None

    if system_name == "darwin":
        requirement_files.append(REPO_ROOT / "requirements.cpu-paddle.txt")
        requirement_files.append(REPO_ROOT / "requirements.macos-onnx.txt")
        classic_auto = "onnxruntime (if CoreMLExecutionProvider is available) else paddleocr"
        vl_auto = "local_mlx_vlm_service"
        notes.append("macOS PaddlePaddle path is CPU-oriented. ONNX Runtime may improve classic OCR, but CoreML provider availability depends on the wheel/build on that Mac.")
        mlx_python = find_python_at_least((3, 10))
        if mlx_python:
            vl_runtime_setup = {
                "kind": "mlx_server_env",
                "venv_dir": MLX_VENV_DIR,
                "python": mlx_python,
                "requirements": [REPO_ROOT / "requirements.mlx-vlm.txt"],
            }
            notes.append(f"PaddleOCR-VL auto mode will use a separate MLX server environment at {MLX_VENV_DIR.name}.")
        else:
            manual_steps.append("Install Python 3.10+ to auto-bootstrap the local MLX-VLM server environment for PaddleOCR-VL on macOS.")
            manual_steps.append("If you want PaddleOCR-VL auto mode on macOS, start a local MLX-VLM compatible server first.")
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

    if include_glmocr:
        if sys.version_info >= (3, 10):
            requirement_files.append(REPO_ROOT / "requirements.glmocr.txt")
            notes.append("Optional GLM-OCR fallback will be installed from vendored source.")
        else:
            manual_steps.append("GLM-OCR optional install was skipped because it requires Python 3.10+. Use Python 3.10/3.11/3.12 if you need that fallback.")

    if sys.version_info < (3, 10):
        notes.append("This interpreter is older than Python 3.10. Base OCR may still install, but optional GLM-OCR fallback is unavailable on this interpreter.")

    return {
        "system": system_name,
        "machine": machine,
        "python_version": python_version,
        "gpu_inventory": gpu_inventory.strip(),
        "has_nvidia": has_nvidia,
        "has_intel_arc": has_intel_arc,
        "requirement_files": requirement_files,
        "classic_auto": classic_auto,
        "vl_auto": vl_auto,
        "vl_runtime_setup": vl_runtime_setup,
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
    parser.add_argument("--with-glmocr", action="store_true", help="Also install the optional vendored GLM-OCR package (requires Python 3.10+).")
    parser.add_argument("--skip-vl-runtime", action="store_true", help="Skip installing the auto-selected PaddleOCR-VL runtime helper environment.")
    args = parser.parse_args()

    plan = build_plan(include_glmocr=args.with_glmocr)
    requirement_files: list[Path] = list(plan["requirement_files"])  # type: ignore[assignment]
    vl_runtime_setup: dict[str, object] | None = plan.get("vl_runtime_setup")  # type: ignore[assignment]

    print("Bootstrap plan")
    print(f"- system: {plan['system']}")
    print(f"- machine: {plan['machine']}")
    print(f"- python: {plan['python_version']}")
    print(f"- classic auto engine: {plan['classic_auto']}")
    print(f"- vl auto runtime: {plan['vl_auto']}")
    print("- requirement files:")
    for path in requirement_files:
        print(f"  - {path.relative_to(REPO_ROOT)}")
    if vl_runtime_setup:
        print("- vl runtime setup:")
        print(f"  - kind: {vl_runtime_setup['kind']}")
        print(f"  - venv: {Path(vl_runtime_setup['venv_dir']).relative_to(REPO_ROOT)}")
        print(f"  - python: {vl_runtime_setup['python']}")
        for path in vl_runtime_setup.get("requirements", []):
            print(f"  - requirement: {Path(path).relative_to(REPO_ROOT)}")

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
    if vl_runtime_setup and not args.skip_vl_runtime and vl_runtime_setup.get("kind") == "mlx_server_env":
        ensure_mlx_server_environment(dry_run=False)
    print("Bootstrap install completed.")
    if manual_steps:
        print("Remaining manual steps:")
        for item in manual_steps:
            print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
