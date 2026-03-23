from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _find_tesseract(explicit: str | None) -> str | None:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)

    which = shutil.which("tesseract")
    if which:
        return which

    candidates = [
        Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
        Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
        Path("C:/Users/alkin/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _patch_yaml_executable(config_path: Path, exe: str) -> None:
    text = config_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out: list[str] = []
    in_ocr = False
    replaced = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("ocr:"):
            in_ocr = True
            out.append(line)
            continue

        if in_ocr and line and not line.startswith(" "):
            in_ocr = False

        if in_ocr and stripped.startswith("executable:"):
            safe = exe.replace("\\", "/")
            out.append(f"  executable: '{safe}'")
            replaced = True
            continue

        out.append(line)

    if not replaced:
        # Fallback: append an ocr block if missing
        out.append("")
        out.append("ocr:")
        safe = exe.replace("\\", "/")
        out.append(f"  executable: '{safe}'")

    config_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def run() -> int:
    parser = argparse.ArgumentParser(description="Ensure Tesseract path and run benchmark")
    parser.add_argument("--config", default="config/ruhsat_schema.yaml")
    parser.add_argument("--annotations", default="dataset/generated/train_annotations.csv")
    parser.add_argument("--output-dir", default="dataset/generated/qa/benchmark_tuned_v1")
    parser.add_argument("--tesseract", default="", help="Optional explicit tesseract.exe path")
    parser.add_argument(
        "--python",
        default="",
        help="Optional python executable to run benchmark with (e.g. conda env python)",
    )
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    if not config_path.exists():
        raise RuntimeError(f"Config not found: {config_path}")

    tesseract_path = _find_tesseract(args.tesseract.strip() or None)
    if not tesseract_path:
        print("[setup] Tesseract not found.")
        print("[setup] Install Tesseract or pass --tesseract \"C:/.../tesseract.exe\".")
        return 2

    _patch_yaml_executable(config_path, tesseract_path)
    print(f"[setup] Set ocr.executable in {config_path} -> {tesseract_path}")

    py_exec = args.python.strip() or sys.executable
    cmd = [
        py_exec,
        str((PROJECT_ROOT / "scripts/benchmark_pipeline.py").resolve()),
        "--annotations",
        args.annotations,
        "--config",
        args.config,
        "--output-dir",
        args.output_dir,
    ]

    print("[setup] Running benchmark:", " ".join(cmd))
    env = os.environ.copy()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(run())

