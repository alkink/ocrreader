"""
draw_anchors.py
---------------
Ruhsat görüntüsü üzerine tüm anchor arama bölgelerini ve
bulunan anchor konumlarını boyar.

Kullanım:
  python scripts/draw_anchors.py \
    --image "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg" \
    --config "config/ruhsat_schema_paddle_v29.yaml" \
    --output "output/anchor_regions.png"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from ocrreader.config import load_config
from ocrreader.io_utils import imread_color, imwrite
from ocrreader.preprocess import preprocess_document
from ocrreader.ocr_engine import create_ocr_engine
from ocrreader.anchors import detect_anchors
from ocrreader.fields import resolve_field_rois

# Her anchor için farklı renk (BGR)
ANCHOR_COLORS = [
    (0,   200,  50),   # plate_label         → yeşil
    (0,   120, 255),   # brand_label         → turuncu
    (255,  50,  50),   # type_label          → mavi
    (180,   0, 255),   # model_year_label    → mor
    (0,   220, 220),   # engine_no_label     → sarı
    (255,   0, 140),   # chassis_no_label    → pembe
    (0,   255, 180),   # tax_no_label        → açık yeşil
    (50,  170, 255),   # surname_label       → açık turuncu
    (255, 200,   0),   # name_label          → camgöbeği
    (100,  50, 255),   # first_registration  → koyu kırmızı
    (255,  50, 200),   # registration_label  → pembe-mor
    (40,  220, 120),   # inspection_label    → orman yeşili
    (200, 200,   0),   # serial_label        → teal
]


def draw_region(img, x, y, w, h, color, label, alpha=0.18):
    """Yarı saydam dolgu + çerçeve + etiket."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # Etiket arka planı
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.50
    th = 1
    (tw, th_), _ = cv2.getTextSize(label, font, fs, th)
    tx = x + 4
    ty = max(y + th_ + 4, 14)
    cv2.rectangle(img, (tx - 2, ty - th_ - 2), (tx + tw + 2, ty + 2), color, -1)
    cv2.putText(img, label, (tx, ty), font, fs, (255, 255, 255), th, cv2.LINE_AA)


def draw_found_anchor(img, bbox, color, name):
    """Bulunan anchor'ı kalın çerçeve + yıldız ile işaretle."""
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    cv2.rectangle(img, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 255, 255), 3)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    label = f"✓ {name}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th_), _ = cv2.getTextSize(label, font, 0.45, 1)
    ty = y - 5 if y > 20 else y + h + th_ + 5
    cv2.rectangle(img, (x, ty - th_ - 2), (x + tw + 4, ty + 2), color, -1)
    cv2.putText(img, label, (x + 2, ty), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="config/ruhsat_schema_paddle_v29.yaml")
    parser.add_argument("--output", default="output/anchor_regions.png")
    parser.add_argument("--show-fields", action="store_true",
                        help="Alan ROI'larını da göster")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"[draw_anchors] Görüntü yükleniyor: {args.image}")
    image = imread_color(args.image)

    print("[draw_anchors] Ön işleme (perspektif + deskew)...")
    prep = preprocess_document(image, cfg.pipeline)
    doc = prep.normalized_image.copy()

    H, W = doc.shape[:2]
    print(f"[draw_anchors] Normalize boyut: {W}×{H}")

    # ── 1) Anchor arama bölgelerini boya ─────────────────────────────────────
    anchor_names = list(cfg.anchors.keys())
    print(f"\n[draw_anchors] {len(anchor_names)} anchor arama bölgesi:")

    region_canvas = doc.copy()
    for idx, (name, acfg) in enumerate(cfg.anchors.items()):
        color = ANCHOR_COLORS[idx % len(ANCHOR_COLORS)]
        if acfg.search_region_norm:
            rx, ry, rw, rh = acfg.search_region_norm
            px = int(rx * W)
            py = int(ry * H)
            pw = int(rw * W)
            ph = int(rh * H)
            draw_region(region_canvas, px, py, pw, ph, color, name)
            aliases = " / ".join(acfg.aliases[:2])
            print(f"  {name:30s} → [{rx:.2f}, {ry:.2f}, {rw:.2f}, {rh:.2f}]  "
                  f"({px},{py})→({px+pw},{py+ph})  aliases: {aliases}")

    # ── 2) OCR + gerçek anchor tespiti ───────────────────────────────────────
    print("\n[draw_anchors] OCR çalıştırılıyor (anchor tespiti için)...")
    try:
        engine = create_ocr_engine(cfg.ocr)
        words = engine.iter_words(doc, psm=cfg.ocr.psm, min_conf=0.0)
        found_anchors = detect_anchors(words, cfg.anchors)

        print(f"[draw_anchors] Bulunan anchor'lar ({len(found_anchors)}/{len(anchor_names)}):")
        for aname, match in found_anchors.items():
            idx = anchor_names.index(aname) if aname in anchor_names else 0
            color = ANCHOR_COLORS[idx % len(ANCHOR_COLORS)]
            draw_found_anchor(region_canvas, match.bbox, color, f"{aname} ({match.score:.2f})")
            print(f"  ✓ {aname:28s} alias='{match.alias}'  score={match.score:.3f}  "
                  f"bbox=({match.bbox.x},{match.bbox.y},{match.bbox.w},{match.bbox.h})")

        missing = [n for n in anchor_names if n not in found_anchors]
        if missing:
            print(f"\n  ✗ Bulunamayan anchor'lar: {missing}")

        # ── 3) Alan ROI'larını da iste ────────────────────────────────────────
        if args.show_fields:
            rois = resolve_field_rois(doc.shape, cfg.fields, found_anchors)
            for fname, roi in rois.items():
                cv2.rectangle(region_canvas,
                              (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h),
                              (255, 255, 0), 1)
                cv2.putText(region_canvas, fname,
                            (roi.x + 2, roi.y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

    except Exception as e:
        print(f"[draw_anchors] OCR çalışamadı ({e}), sadece bölgeler çiziliyor.")

    # ── Açıklama kutusu ───────────────────────────────────────────────────────
    legend_y = 10
    for idx, (name, acfg) in enumerate(cfg.anchors.items()):
        color = ANCHOR_COLORS[idx % len(ANCHOR_COLORS)]
        ly = H - (len(anchor_names) - idx) * 18 - 5
        cv2.rectangle(region_canvas, (W - 240, ly - 12), (W - 225, ly), color, -1)
        cv2.putText(region_canvas, name,
                    (W - 220, ly - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(out_path), region_canvas)
    print(f"\n[draw_anchors] Kaydedildi → {out_path.resolve()}")


if __name__ == "__main__":
    run()
