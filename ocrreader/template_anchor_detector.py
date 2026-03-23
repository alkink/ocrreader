"""
template_anchor_detector.py
===========================
OCR-bağımsız anchor detection: OpenCV template matching kullanır.
Tesseract'ın label okuyamadığı bozuk/ters/düşük-kalite görüntülerde de çalışır.

Kullanım:
    1. templates/ klasörüne label crop'larını koy:
         templates/plate_label_1.png
         templates/plate_label_2.png
         templates/brand_label_1.png
         ...
    2. anchors.py içinde detect_anchors() çağrısından önce dene,
       bulunamayan anchor'lar için template detector'a devret.

Template toplama:
    python scripts/collect_anchor_templates.py
    (aşağıda da var)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import AnchorConfig
from .types import Rect


@dataclass(frozen=True)
class TemplateHit:
    anchor_name: str
    template_file: str
    score: float
    bbox: Rect


class TemplateAnchorDetector:
    """
    Referans label patch'lerini cv2.matchTemplate ile arar.
    Tek bir scale değil, birden fazla scale dener → farklı boyutlardaki
    ruhsatlarda da tutarlı çalışır.
    """

    SCALES = (0.60, 0.75, 0.90, 1.00, 1.15, 1.30)

    def __init__(
        self,
        templates_dir: str | Path,
        default_threshold: float = 0.62,
        per_anchor_threshold: dict[str, float] | None = None,
    ) -> None:
        self.threshold = default_threshold
        self.per_anchor_threshold = per_anchor_threshold or {}
        self._templates: dict[str, list[tuple[np.ndarray, str]]] = {}
        self._load(Path(templates_dir))

    # ------------------------------------------------------------------
    def _load(self, d: Path) -> None:
        if not d.exists():
            return
        for p in sorted(d.glob("*.png")):
            # dosya adı: <anchor_name>_<N>.png  → e.g. plate_label_1.png
            parts = p.stem.rsplit("_", 1)
            anchor_name = parts[0]
            tmpl = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if tmpl is None:
                continue
            self._templates.setdefault(anchor_name, []).append((tmpl, p.name))

    # ------------------------------------------------------------------
    def detect(
        self,
        gray_img: np.ndarray,
        anchor_names: list[str] | None = None,
    ) -> dict[str, TemplateHit]:
        """
        gray_img: normalize edilmiş 8-bit grayscale görüntü
        anchor_names: sadece bu anchor'ları ara (None = hepsini ara)
        """
        results: dict[str, TemplateHit] = {}
        search_anchors = anchor_names or list(self._templates.keys())

        for anchor_name in search_anchors:
            templates = self._templates.get(anchor_name, [])
            if not templates:
                continue

            thr = self.per_anchor_threshold.get(anchor_name, self.threshold)
            best: TemplateHit | None = None

            for tmpl, tmpl_file in templates:
                for scale in self.SCALES:
                    th = max(4, int(tmpl.shape[0] * scale))
                    tw = max(4, int(tmpl.shape[1] * scale))
                    if th > gray_img.shape[0] or tw > gray_img.shape[1]:
                        continue

                    resized = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)
                    result_map = cv2.matchTemplate(
                        gray_img, resized, cv2.TM_CCOEFF_NORMED
                    )
                    _, max_val, _, max_loc = cv2.minMaxLoc(result_map)

                    if max_val >= thr:
                        x, y = max_loc
                        hit = TemplateHit(
                            anchor_name=anchor_name,
                            template_file=tmpl_file,
                            score=float(max_val),
                            bbox=Rect(x=x, y=y, w=tw, h=th),
                        )
                        if best is None or hit.score > best.score:
                            best = hit

            if best is not None:
                results[anchor_name] = best

        return results


# ======================================================================
# anchors.py ile entegrasyon — detect_anchors() için hibrit wrapper
# ======================================================================

from .anchors import AnchorMatch, detect_anchors  # noqa: E402  (circular safe)


def detect_anchors_hybrid(
    words: list[object],
    anchors: dict[str, AnchorConfig],
    gray_img: np.ndarray,
    template_detector: TemplateAnchorDetector | None,
) -> dict[str, AnchorMatch]:
    """
    1. Önce OCR tabanlı anchor detection dene (mevcut logic).
    2. Bulunamayan anchor'lar için template matching dene.
    3. İkisini birleştirip döndür.
    """
    # Adım 1: OCR-based (mevcut)
    ocr_found = detect_anchors(words, anchors)

    if template_detector is None:
        return ocr_found

    # Adım 2: sadece bulunamayan anchor'lar için template dene
    missing = [name for name in anchors if name not in ocr_found]
    if not missing:
        return ocr_found

    tmpl_hits = template_detector.detect(gray_img, anchor_names=missing)

    # Adım 3: birleştir
    combined = dict(ocr_found)
    for name, hit in tmpl_hits.items():
        combined[name] = AnchorMatch(
            name=name,
            alias=f"[template:{hit.template_file}]",
            score=hit.score,
            bbox=hit.bbox,
        )

    return combined
