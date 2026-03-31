"""
draw_detection_flow.py
-----------------------
Bir anchor için tüm tespit akışını adım adım görselleştirir:

  Adım 1  → Arama bölgesi (search_region_norm)
  Adım 2  → OCR ile okunan tüm kelimeler (bölge içinde)
  Adım 3  → Normalize + fuzzy match sonucu (anchor kelimesi)
  Adım 4  → value_margin_norm penceresi (değerin arandığı alan)
  Adım 5  → Değer kelimesi (bulunan sonuç)

Her adım ayrı bir PNG olarak kaydedilir.
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
from ocrreader.anchors import detect_anchors, _split_alias, _match_alias
from ocrreader.text_utils import normalize_for_match
from ocrreader.field_value_locator import locate_value_from_anchor, _make_window
from ocrreader.fields import cleanup_text, _post_field_filters, _post_cleanup

IMAGE  = "testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg"
CONFIG = "config/ruhsat_schema_paddle_v29.yaml"
OUTDIR = Path("output/detection_flow")

FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img, text, x, y, color, scale=0.46, pad=3):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + pad), color, -1)
    cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), 1, cv2.LINE_AA)


def filled(img, x1, y1, x2, y2, color, alpha=0.20):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config(CONFIG)
    img = imread_color(IMAGE)
    prep = preprocess_document(img, cfg.pipeline)
    doc  = prep.normalized_image
    H, W = doc.shape[:2]

    ocr_cfg = OCRConfig(
        engine="tesseract",
        executable="C:/Program Files/Tesseract-OCR/tesseract.exe",
        language="tur+eng", oem=3, psm=6,
    )
    engine = TesseractEngine(ocr_cfg)
    print("[flow] OCR çalışıyor...")
    words = engine.iter_words(doc, min_conf=0.0)
    found_anchors = detect_anchors(words, cfg.anchors)

    print(f"[flow] Toplam {len(words)} kelime tespit edildi.\n")

    # ── Her anchor için akış görselleştir ─────────────────────────────────────
    for anchor_name, acfg in cfg.anchors.items():
        canvas = doc.copy()

        # ── ADIM 1: Arama bölgesi ─────────────────────────────────────────────
        STEP1_COLOR = (50, 200, 50)  # yeşil
        if acfg.search_region_norm:
            rx, ry, rw, rh = acfg.search_region_norm
            sx1, sy1 = int(rx * W), int(ry * H)
            sx2, sy2 = int((rx + rw) * W), int((ry + rh) * H)
            filled(canvas, sx1, sy1, sx2, sy2, STEP1_COLOR, alpha=0.12)
            label(canvas, f"[1] Arama Bolgesi: {acfg.search_region_norm}", sx1 + 2, sy1 + 16,
                  STEP1_COLOR)
        else:
            sx1, sy1, sx2, sy2 = 0, 0, W, H

        # ── ADIM 2: Bölge içindeki OCR kelimeleri ────────────────────────────
        STEP2_COLOR = (200, 200, 50)  # sarı
        max_x = max((w.bbox.x + w.bbox.w for w in words), default=1)
        max_y = max((w.bbox.y + w.bbox.h for w in words), default=1)

        scoped_words = []
        for w in words:
            if acfg.search_region_norm:
                rx, ry, rw, rh = acfg.search_region_norm
                region_x = int(rx * max_x)
                region_y = int(ry * max_y)
                region_w = max(1, int(rw * max_x))
                region_h = max(1, int(rh * max_y))
                cx = w.bbox.x + w.bbox.w / 2
                cy = w.bbox.y + w.bbox.h / 2
                if region_x <= cx <= region_x + region_w and region_y <= cy <= region_y + region_h:
                    scoped_words.append(w)
            else:
                scoped_words.append(w)

        for w in scoped_words:
            bx, by, bw, bh = w.bbox.x, w.bbox.y, w.bbox.w, w.bbox.h
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), STEP2_COLOR, 1)
            # Normalize metin göster
            norm = normalize_for_match(w.text)
            cv2.putText(canvas, norm, (bx, by - 2), FONT, 0.30, STEP2_COLOR, 1)

        # ── ADIM 3: Fuzzy match — hangi kelime anchor olarak seçildi ─────────
        STEP3_COLOR = (0, 50, 255)  # kırmızı

        best_alias = None
        best_score = 0.0
        best_rect  = None

        for alias in acfg.aliases:
            alias_tokens = _split_alias(alias)
            result = _match_alias(scoped_words, alias_tokens, acfg.min_score)
            if result and result[0] > best_score:
                best_score = result[0]
                best_rect  = result[1]
                best_alias = alias

        if best_rect:
            bx, by = best_rect.x, best_rect.y
            bw, bh = best_rect.w, best_rect.h
            # Kalın kırmızı çerçeve
            cv2.rectangle(canvas, (bx - 3, by - 3), (bx + bw + 3, by + bh + 3), (255,255,255), 3)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), STEP3_COLOR, 2)
            label(canvas, f"[3] ANCHOR BULUNDU: '{best_alias}' (score={best_score:.3f})",
                  bx, max(by - 8, 14), STEP3_COLOR)
        else:
            # Bulunamadı — üst sol köşeye yaz
            label(canvas, f"[3] ANCHOR BULUNAMADI (min_score={acfg.min_score})",
                  sx1 + 2, sy1 + 36, (0, 0, 220))

        # ── ADIM 4 & 5: Değer penceresi ve bulunan değer ─────────────────────
        field_cfg  = cfg.fields.get(anchor_name.replace("_label", ""))

        # anchor_name → field_name eşlemesi (ör. engine_no_label → engine_no)
        field_map = {
            "plate_label":              "plate",
            "brand_label":              "brand",
            "type_label":               "type",
            "model_year_label":         "model_year",
            "engine_no_label":          "engine_no",
            "chassis_no_label":         "chassis_no",
            "tax_no_label":             "tax_or_id_no",
            "surname_label":            "owner_surname",
            "name_label":               "owner_name",
            "first_registration_label": "first_registration_date",
            "registration_label":       "registration_date",
            "inspection_label":         "inspection_date",
            "serial_label":             "serial_no",
        }
        field_name = field_map.get(anchor_name)
        field_cfg  = cfg.fields.get(field_name) if field_name else None

        STEP4_COLOR = (255, 140, 0)   # turuncu
        STEP5_COLOR = (0,   255, 180) # açık yeşil

        if field_cfg and best_rect and anchor_name in found_anchors:
            anchor_match = found_anchors[anchor_name]
            margin = field_cfg.value_margin_norm or (0.0, 0.0, 0.22, 0.10)
            direction = field_cfg.value_from_anchor

            win = _make_window(anchor_match.bbox, W, H, direction, margin)
            filled(canvas, win.x, win.y, win.x + win.w, win.y + win.h, STEP4_COLOR, alpha=0.22)
            label(canvas,
                  f"[4] Deger penceresi ({direction}) margin={margin}",
                  win.x + 2, win.y + 14, STEP4_COLOR)

            # Pencere içindeki kelimeler
            in_win = [w for w in words
                      if win.x <= (w.bbox.x + w.bbox.w/2) <= win.x + win.w
                      and win.y <= (w.bbox.y + w.bbox.h/2) <= win.y + win.h]
            for w in in_win:
                cv2.rectangle(canvas, (w.bbox.x, w.bbox.y),
                              (w.bbox.x + w.bbox.w, w.bbox.y + w.bbox.h),
                              STEP4_COLOR, 1)

            # Değer bul
            candidate = locate_value_from_anchor(field_cfg, anchor_match, words, doc.shape)
            if candidate:
                vb = candidate.bbox
                cleaned = cleanup_text(candidate.text, field_cfg.cleanup)
                cleaned = _post_cleanup(cleaned, field_cfg)
                cleaned = _post_field_filters(field_name, cleaned)
                cv2.rectangle(canvas, (vb.x - 2, vb.y - 2), (vb.x + vb.w + 2, vb.y + vb.h + 2),
                              (255, 255, 255), 3)
                cv2.rectangle(canvas, (vb.x, vb.y), (vb.x + vb.w, vb.y + vb.h), STEP5_COLOR, 2)
                label(canvas,
                      f"[5] DEGER: '{cleaned}' (raw='{candidate.text}')",
                      vb.x, max(vb.y + vb.h + 14, 14), STEP5_COLOR, scale=0.50)
                print(f"  {anchor_name:30s} → field={field_name:25s} → '{cleaned}'")
            else:
                print(f"  {anchor_name:30s} → field={field_name:25s} → (değer penceresi boş)")

        elif anchor_name not in found_anchors:
            print(f"  {anchor_name:30s} → ANCHOR BULUNAMADI")

        # ── Adım açıklamaları sol üst köşe ────────────────────────────────────
        legend = [
            ("[1] Arama bolgesi",  STEP1_COLOR),
            ("[2] OCR kelimeleri", STEP2_COLOR),
            ("[3] Anchor eslesme", STEP3_COLOR),
            ("[4] Deger penceresi",STEP4_COLOR),
            ("[5] Bulunan deger",  STEP5_COLOR),
        ]
        for i, (txt, col) in enumerate(legend):
            ly = 18 + i * 20
            cv2.rectangle(canvas, (4, ly - 12), (18, ly), col, -1)
            cv2.putText(canvas, txt, (22, ly - 1), FONT, 0.40, col, 1, cv2.LINE_AA)

        # Başlık
        title = f"{anchor_name}  |  aliases: {', '.join(acfg.aliases[:3])}"
        cv2.rectangle(canvas, (0, 0), (W, 14), (30, 30, 30), -1)
        cv2.putText(canvas, title, (4, 11), FONT, 0.38, (255, 255, 255), 1)

        out = OUTDIR / f"{anchor_name}.png"
        imwrite(str(out), canvas)

    # ── Tüm anchor'lar tek birleşik görüntüde ──────────────────────────────────
    print("\n[flow] Birleşik özet görüntü oluşturuluyor...")
    summary = doc.copy()
    PALETTE = [
        (50,205,50),(0,140,255),(220,50,50),(200,0,230),(0,230,230),
        (255,0,170),(0,200,130),(30,160,255),(255,200,20),(80,50,240),
        (255,50,180),(30,190,90),(180,180,0),
    ]
    anchor_names = list(cfg.anchors.keys())
    for idx, (anchor_name, acfg) in enumerate(cfg.anchors.items()):
        col = PALETTE[idx % len(PALETTE)]
        if acfg.search_region_norm:
            rx, ry, rw, rh = acfg.search_region_norm
            x1, y1 = int(rx*W), int(ry*H)
            x2, y2 = int((rx+rw)*W), int((ry+rh)*H)
            filled(summary, x1, y1, x2, y2, col, alpha=0.13)
            label(summary, anchor_name, x1+2, y1+14, col, scale=0.38)

        if anchor_name in found_anchors:
            m = found_anchors[anchor_name]
            bx, by, bw, bh = m.bbox.x, m.bbox.y, m.bbox.w, m.bbox.h
            cv2.rectangle(summary, (bx-2,by-2),(bx+bw+2,by+bh+2),(255,255,255),2)
            cv2.rectangle(summary, (bx,by),(bx+bw,by+bh),col,2)

    imwrite(str(OUTDIR / "_summary.png"), summary)
    print(f"[flow] Tüm dosyalar → {OUTDIR.resolve()}")
    print(f"  Bireysel: {anchor_name}.png (her anchor için)")
    print(f"  Özet:     _summary.png")


if __name__ == "__main__":
    main()
