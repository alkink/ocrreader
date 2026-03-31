"""
Tesseract ile anchor tespiti yapıp görüntü üzerine çizen script.
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from ocrreader.config import load_config, OCRConfig
from ocrreader.io_utils import imread_color, imwrite
from ocrreader.preprocess import preprocess_document
from ocrreader.ocr_engine import TesseractEngine
from ocrreader.anchors import detect_anchors
from ocrreader.fields import resolve_field_rois

IMAGE  = "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg"
CONFIG = "config/ruhsat_schema_paddle_v29.yaml"
OUTPUT = "output/anchor_regions_tess.png"

# 13 anchor için renk paleti (BGR)
COLORS = [
    (50,  205,  50),   # plate_label          yeşil
    (0,   140, 255),   # brand_label          turuncu
    (220,  50,  50),   # type_label           mavi
    (200,   0, 230),   # model_year_label     mor
    (0,   230, 230),   # engine_no_label      sarı
    (255,   0, 170),   # chassis_no_label     pembe
    (0,   200, 130),   # tax_no_label         açık yeşil
    (30,  160, 255),   # surname_label        açık turuncu
    (255, 200,  20),   # name_label           camgöbeği
    (80,   50, 240),   # first_registration   mor-mavi
    (255,  50, 180),   # registration_label   sıcak pembe
    (30,  190,  90),   # inspection_label     orman yeşili
    (180, 180,   0),   # serial_label         koyu sarı
]


def filled_rect(img, x1, y1, x2, y2, color, alpha=0.20):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


def put_label(img, text, x, y, color, scale=0.46, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    tx, ty = x + 4, max(y + th + 4, 16)
    cv2.rectangle(img, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), color, -1)
    cv2.putText(img, text, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    cfg = load_config(CONFIG)
    img = imread_color(IMAGE)

    print("[draw] Ön işleme...")
    prep = preprocess_document(img, cfg.pipeline)
    doc  = prep.normalized_image.copy()
    H, W = doc.shape[:2]
    print(f"[draw] Normalize boyut: {W}×{H}")

    canvas = doc.copy()

    # ── 1. Anchor arama bölgelerini yar-saydam boya ───────────────────────────
    anchor_names = list(cfg.anchors.keys())
    for idx, (name, acfg) in enumerate(cfg.anchors.items()):
        color = COLORS[idx % len(COLORS)]
        if not acfg.search_region_norm:
            continue
        rx, ry, rw, rh = acfg.search_region_norm
        x1, y1 = int(rx * W),          int(ry * H)
        x2, y2 = int((rx + rw) * W),   int((ry + rh) * H)
        filled_rect(canvas, x1, y1, x2, y2, color, alpha=0.16)
        put_label(canvas, name, x1, y1, color)
        print(f"  bölge  {name:30s}  [{rx:.2f},{ry:.2f},{rw:.2f},{rh:.2f}]  "
              f"→ ({x1},{y1})-({x2},{y2})")

    # ── 2. Tesseract ile anchor tespiti ────────────────────────────────────────
    print("\n[draw] Tesseract OCR çalıştırılıyor...")
    ocr_cfg = OCRConfig(
        engine="tesseract",
        executable="C:/Program Files/Tesseract-OCR/tesseract.exe",
        language="tur+eng",
        oem=3,
        psm=6,
    )
    engine = TesseractEngine(ocr_cfg)
    words  = engine.iter_words(doc, min_conf=0.0)
    found  = detect_anchors(words, cfg.anchors)

    print(f"\n[draw] Bulunan anchor'lar: {len(found)}/{len(anchor_names)}")
    for aname, match in found.items():
        idx   = anchor_names.index(aname) if aname in anchor_names else 0
        color = COLORS[idx % len(COLORS)]
        bx, by, bw, bh = match.bbox.x, match.bbox.y, match.bbox.w, match.bbox.h
        # Kalın beyaz dış çerçeve + renkli iç çerçeve
        cv2.rectangle(canvas, (bx - 3, by - 3), (bx + bw + 3, by + bh + 3), (255, 255, 255), 3)
        cv2.rectangle(canvas, (bx,     by),     (bx + bw,     by + bh),     color,           2)
        label = f"* {aname} ({match.score:.2f})"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.42, 1)
        ty = by - 6 if by > 22 else by + bh + th + 6
        cv2.rectangle(canvas, (bx, ty - th - 2), (bx + tw + 4, ty + 2), color, -1)
        cv2.putText(canvas, label, (bx + 2, ty), font, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        print(f"  ✓ {aname:28s}  alias='{match.alias}'  score={match.score:.3f}  "
              f"bbox=({bx},{by},{bw},{bh})")

    missing = [n for n in anchor_names if n not in found]
    if missing:
        print(f"\n  ✗ Bulunamayan: {missing}")

    # ── 3. Alan ROI'ları (sarı ince çizgi) ────────────────────────────────────
    rois = resolve_field_rois(doc.shape, cfg.fields, found)
    for fname, roi in rois.items():
        cv2.rectangle(canvas, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 255, 255), 1)
        cv2.putText(canvas, fname, (roi.x + 2, roi.y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 255), 1)

    # ── 4. Sağ alt legend ─────────────────────────────────────────────────────
    for idx, name in enumerate(anchor_names):
        color = COLORS[idx % len(COLORS)]
        ly = H - (len(anchor_names) - idx) * 17 - 4
        cv2.rectangle(canvas, (W - 250, ly - 11), (W - 237, ly), color, -1)
        cv2.putText(canvas, name, (W - 233, ly - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    imwrite(OUTPUT, canvas)
    print(f"\n[draw] Kaydedildi → {Path(OUTPUT).resolve()}")


if __name__ == "__main__":
    main()
