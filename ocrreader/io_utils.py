from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_color(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def imwrite(path: str, image: np.ndarray) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = out.suffix if out.suffix else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Unable to encode image for: {path}")
    encoded.tofile(str(out))

