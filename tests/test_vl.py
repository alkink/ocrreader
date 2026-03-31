import os
import site
from pathlib import Path


def _iter_site_packages() -> list[Path]:
    roots: list[Path] = []
    for raw in site.getsitepackages():
        p = Path(raw)
        if p.exists():
            roots.append(p)
    user_site = site.getusersitepackages()
    if user_site:
        p = Path(user_site)
        if p.exists():
            roots.append(p)
    return roots


def _add_nvidia_dll_dirs() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    seen: set[str] = set()
    for site_root in _iter_site_packages():
        for bin_dir in site_root.glob("nvidia/*/bin"):
            resolved = str(bin_dir.resolve())
            if resolved in seen or not bin_dir.is_dir():
                continue
            seen.add(resolved)
            os.add_dll_directory(resolved)
            print(f"[DLL] Added: {resolved}")


_add_nvidia_dll_dirs()

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCRVL
import numpy as np

def main():
    print("Loading PaddleOCRVL model...")
    vl = PaddleOCRVL(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
        use_ocr_for_image_block=False,
    )
    print("Model loaded OK!")
    img = np.ones((100, 200, 3), dtype=np.uint8) * 200
    res = vl.predict(img, prompt_label="text")
    print("Result:", res)

if __name__ == '__main__':
    main()
