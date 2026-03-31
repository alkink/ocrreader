from __future__ import annotations

import argparse
from pathlib import Path
import platform
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_VENV_DIR = REPO_ROOT / ".venv-mlx"


def venv_python_path(venv_dir: Path) -> Path:
    if platform.system().strip().lower() == "windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_server_env() -> Path:
    venv_python = venv_python_path(MLX_VENV_DIR)
    if venv_python.exists():
        return venv_python

    bootstrap = REPO_ROOT / "scripts" / "bootstrap_runtime.py"
    cmd = [sys.executable, str(bootstrap)]
    print(f"[mlx-server] missing {MLX_VENV_DIR.name}, bootstrapping it first")
    subprocess.run(cmd, check=True)
    if not venv_python.exists():
        raise RuntimeError(f"Expected MLX venv python at {venv_python}, but it was not created.")
    return venv_python


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the local MLX-VLM server used by PaddleOCR-VL on macOS.")
    parser.add_argument("--port", type=int, default=8111)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Assume .venv-mlx already exists and do not try to create it.")
    args = parser.parse_args()

    if platform.system().strip().lower() != "darwin":
        raise SystemExit("This helper is intended for macOS / Apple Silicon hosts.")

    if not shutil.which("sw_vers"):
        raise SystemExit("macOS tooling was not detected on PATH.")

    venv_python = venv_python_path(MLX_VENV_DIR)
    if not args.skip_bootstrap:
        venv_python = ensure_server_env()
    elif not venv_python.exists():
        raise SystemExit(f"{venv_python} was not found. Run without --skip-bootstrap first.")

    cmd = [str(venv_python), "-m", "mlx_vlm.server", "--host", str(args.host), "--port", str(args.port)]
    print(f"[mlx-server] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
