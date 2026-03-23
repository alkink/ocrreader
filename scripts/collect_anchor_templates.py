"""
collect_anchor_templates.py
============================
Anchor template'lerini toplar.

İki mod:
  --mode auto   : anchor_label_probe_*_words.csv'den yüksek-confidence
                  kelime bounding box'larını kullanarak crop alır.
  --mode manual : interaktif, görüntü üzerinde crop seçersin.

Kullanım:
  # Otomatik — mevcut probe CSV'lerden:
  python scripts/collect_anchor_templates.py \
      --mode auto \
      --probe-dir dataset/generated/qa \
      --image-dir dataset/photo \
      --out-dir config/anchor_templates

  # Manuel — tek görüntü:
  python scripts/collect_anchor_templates.py \
      --mode manual \
      --image dataset/photo/06CFZ624.png \
      --out-dir config/anchor_templates
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np


# anchor_name → ruhsatta aranacak label'ın normalize metni
ANCHOR_LABEL_KEYWORDS = {
    "plate_label":              ["PLAKA"],
    "brand_label":              ["MARKASI", "MARKA"],
    "type_label":               ["TIPI", "TIP"],
    "model_year_label":         ["MODEL", "YILI"],
    "engine_no_label":          ["MOTOR", "NO"],
    "chassis_no_label":         ["SASE", "SASE NO", "ŞASE"],
    "tax_no_label":             ["VERGI", "NO"],
    "surname_label":            ["SOYADI"],
    "name_label":               ["ADI"],
    "first_registration_label": ["ILK", "TESCIL"],
    "registration_label":       ["TESCIL", "TARIHI"],
    "inspection_label":         ["MUAYENE", "DIGER"],
    "serial_label":             ["BELGE", "SERI"],
}

PADDING = 4  # crop etrafında pixel boşluk


def _crop_and_save(img: np.ndarray, x: int, y: int, w: int, h: int, out_path: Path) -> None:
    H, W = img.shape[:2]
    x1 = max(0, x - PADDING)
    y1 = max(0, y - PADDING)
    x2 = min(W, x + w + PADDING)
    y2 = min(H, y + h + PADDING)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    cv2.imwrite(str(out_path), gray)
    print(f"  saved: {out_path}")


def _next_idx(out_dir: Path, anchor_name: str) -> int:
    existing = list(out_dir.glob(f"{anchor_name}_*.png"))
    return len(existing) + 1


# ──────────────────────────────────────────────────────────────────────
# AUTO MODE
# ──────────────────────────────────────────────────────────────────────

def run_auto(probe_dir: Path, image_dir: Path, out_dir: Path, min_conf: float = 55.0) -> None:
    """
    Her *_words.csv dosyasındaki kelimeleri anchor keyword'leriyle eşleştir,
    yüksek confidence'lı eşleşmeleri crop olarak kaydet.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_per_anchor: dict[str, int] = {k: 0 for k in ANCHOR_LABEL_KEYWORDS}

    words_csvs = sorted(probe_dir.rglob("*_words.csv"))
    print(f"Found {len(words_csvs)} words CSV files")

    for wcsv in words_csvs:
        # ilgili görüntüyü bul
        # dosya adından: anchor_label_probe_<stem>_words.csv → <stem>
        stem = wcsv.stem  # e.g. anchor_label_probe_arora_osd_words
        # görüntü adını extract et
        img_candidates = list(image_dir.rglob("*.jpg")) + list(image_dir.rglob("*.png"))
        matched_img = None
        # probe dosyası adından görüntü adını çıkarmak zor — tüm görüntüleri dene
        # Bu basit heuristik: probe stem'inde geçen kelimeleri görüntü adında ara
        probe_hint = stem.replace("anchor_label_probe_", "").replace("_words", "").replace("_osd", "")
        for cand in img_candidates:
            if probe_hint.lower() in cand.stem.lower() or cand.stem.lower() in probe_hint.lower():
                matched_img = cand
                break

        if matched_img is None:
            print(f"  [skip] no image for {wcsv.name}")
            continue

        img = cv2.imread(str(matched_img))
        if img is None:
            continue

        # Normalized boyuta scale et (preprocess_document çıktısıyla uyumlu)
        # Probe words normalized koordinatlarda — gerçek px'i yeniden hesapla
        with open(wcsv, encoding="utf-8") as f:
            words = list(csv.DictReader(f))

        print(f"\nProcessing: {matched_img.name}  ({len(words)} words)")

        # Görüntü boyutunu probe'daki max x,y'den inferla
        max_x = max((int(w['x']) + int(w['w']) for w in words), default=img.shape[1])
        max_y = max((int(w['y']) + int(w['h']) for w in words), default=img.shape[0])
        scale_x = img.shape[1] / max_x
        scale_y = img.shape[0] / max_y

        for anchor_name, keywords in ANCHOR_LABEL_KEYWORDS.items():
            if saved_per_anchor.get(anchor_name, 0) >= 5:
                continue  # anchor başına max 5 template yeterli

            for w in words:
                if float(w['conf']) < min_conf:
                    continue
                text_up = w['text'].upper().strip()
                if any(kw in text_up for kw in keywords):
                    x = int(int(w['x']) * scale_x)
                    y = int(int(w['y']) * scale_y)
                    bw = int(int(w['w']) * scale_x)
                    bh = int(int(w['h']) * scale_y)
                    if bw < 5 or bh < 5:
                        continue
                    idx = _next_idx(out_dir, anchor_name)
                    out_path = out_dir / f"{anchor_name}_{idx}.png"
                    _crop_and_save(img, x, y, bw, bh, out_path)
                    saved_per_anchor[anchor_name] = saved_per_anchor.get(anchor_name, 0) + 1

    print("\n=== TEMPLATE SUMMARY ===")
    for name, cnt in saved_per_anchor.items():
        status = "✓" if cnt >= 3 else ("△" if cnt > 0 else "✗")
        print(f"  {status} {name:30} {cnt} templates")


# ──────────────────────────────────────────────────────────────────────
# MANUAL MODE
# ──────────────────────────────────────────────────────────────────────

def run_manual(image_path: Path, out_dir: Path) -> None:
    """
    OpenCV penceresi açar. Her anchor için:
      - Mouse ile dikdörtgen çiz (sol tık → sürükle → bırak)
      - 's' → kaydet ve sonraki anchor'a geç
      - 'n' → bu anchor'u atla
      - 'q' → çık
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    anchor_names = list(ANCHOR_LABEL_KEYWORDS.keys())

    for anchor_name in anchor_names:
        print(f"\nDraw ROI for: {anchor_name}  (s=save, n=skip, q=quit)")
        display = img.copy()
        cv2.putText(display, f"Anchor: {anchor_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
        roi = cv2.selectROI(f"Select: {anchor_name}", display, fromCenter=False)
        cv2.destroyAllWindows()

        if roi == (0, 0, 0, 0):
            print(f"  [skip] {anchor_name}")
            continue

        x, y, w, h = [int(v) for v in roi]
        idx = _next_idx(out_dir, anchor_name)
        out_path = out_dir / f"{anchor_name}_{idx}.png"
        _crop_and_save(img, x, y, w, h, out_path)


# ──────────────────────────────────────────────────────────────────────

def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "manual"], default="auto")
    parser.add_argument("--probe-dir", default="dataset/generated/qa")
    parser.add_argument("--image-dir", default="dataset/photo")
    parser.add_argument("--image", default="", help="Single image (manual mode)")
    parser.add_argument("--out-dir", default="config/anchor_templates")
    parser.add_argument("--min-conf", type=float, default=55.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.mode == "manual":
        if not args.image:
            print("--image required for manual mode")
            return 1
        run_manual(Path(args.image), out_dir)
    else:
        run_auto(
            probe_dir=Path(args.probe_dir),
            image_dir=Path(args.image_dir),
            out_dir=out_dir,
            min_conf=args.min_conf,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
