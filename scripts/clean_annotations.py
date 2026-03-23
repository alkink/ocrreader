"""
clean_annotations.py (v2)
--------------------------
Doğru çalışma mantığı:
  1. train_annotations.csv'yi yükle (65 görüntünün tamamı = ground truth base)
  2. latest_dataset_manual_review.csv'yi yükle (düzeltmeler)
  3. Çakışan change satırlarını çöz (en yüksek score kazanır)
  4. Düzeltmeleri base annotation'a uygula
  5. ambiguous_skip / unmatched görüntüleri dışarıda bırak
  6. gold_annotations.csv olarak yaz

Kullanım:
    python clean_annotations.py
    python clean_annotations.py \\
        --base    dataset/generated/train_annotations.csv \\
        --review  dataset/generated/qa/latest_dataset_manual_review.csv \\
        --output  dataset/generated/qa/gold_annotations.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


SKIP_STATUSES = {"ambiguous_skip", "skip", "unmatched"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def resolve_change_conflicts(
    review_rows: list[dict[str, str]],
) -> tuple[dict[tuple[str, str], str], list[dict[str, str]], set[str]]:
    """
    Review CSV'den:
    - corrections: {(image, field): new_value}  — uygulanacak düzeltmeler
    - conflicts:   çakışma raporu
    - skip_images: ambiguous_skip/unmatched görüntüler (tamamen dışlanır)
    """
    skip_images: set[str] = set()
    for row in review_rows:
        if row.get("status", "").lower() in SKIP_STATUSES:
            img = row.get("image", "")
            if img:
                skip_images.add(img)

    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in review_rows:
        if row.get("type", "").lower() != "change":
            continue
        img = row.get("image", "")
        field = row.get("field", "")
        if not img or not field:
            continue
        buckets[(img, field)].append(row)

    corrections: dict[tuple[str, str], str] = {}
    conflicts: list[dict[str, str]] = []

    for (img, field), candidates in buckets.items():
        if img in skip_images:
            continue

        candidates.sort(key=lambda r: int(r.get("score", 0) or 0), reverse=True)
        winner = candidates[0]
        corrections[(img, field)] = winner.get("new", "") or winner.get("old", "")

        for loser in candidates[1:]:
            conflicts.append({
                "image": img,
                "field": field,
                "kept_value": winner.get("new", ""),
                "kept_score": winner.get("score", ""),
                "dropped_value": loser.get("new", ""),
                "dropped_score": loser.get("score", ""),
                "reason": "higher_score_wins",
            })

    return corrections, conflicts, skip_images


def apply_corrections(
    base_rows: list[dict[str, str]],
    corrections: dict[tuple[str, str], str],
    skip_images: set[str],
) -> list[dict[str, str]]:
    result = []
    for row in base_rows:
        img = row.get("image", "")
        if img in skip_images:
            continue

        new_row = dict(row)
        for (r_img, field), value in corrections.items():
            if r_img == img and field in new_row:
                new_row[field] = value

        result.append(new_row)

    return result


def write_csv(rows: list[dict[str, str]], fieldnames: list[str], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run() -> int:
    parser = argparse.ArgumentParser(description="Annotation temizleme ve gold subset üretimi")
    parser.add_argument(
        "--base",
        default="dataset/generated/train_annotations.csv",
        help="Tam annotation CSV'si (65 görüntü)",
    )
    parser.add_argument(
        "--review",
        default="dataset/generated/qa/latest_dataset_manual_review.csv",
        help="Review / düzeltme CSV'si",
    )
    parser.add_argument(
        "--output",
        default="dataset/generated/qa/gold_annotations.csv",
        help="Çıktı gold CSV'si",
    )
    parser.add_argument(
        "--conflict-report",
        default="dataset/generated/qa/conflict_report.csv",
        help="Çakışma raporu",
    )
    args = parser.parse_args()

    base_path     = Path(args.base)
    review_path   = Path(args.review)
    out_path      = Path(args.output)
    conflict_path = Path(args.conflict_report)

    for p in [base_path, review_path]:
        if not p.exists():
            print(f"[HATA] Dosya bulunamadı: {p}", file=sys.stderr)
            return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    conflict_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Base annotation yükleniyor: {base_path}")
    base_rows = load_csv(base_path)
    print(f"      {len(base_rows)} satır (görüntü) bulundu.")

    print(f"[2/5] Review CSV yükleniyor: {review_path}")
    review_rows = load_csv(review_path)
    print(f"      {len(review_rows)} satır bulundu.")

    print("[3/5] Çakışmalar çözümleniyor...")
    corrections, conflicts, skip_images = resolve_change_conflicts(review_rows)

    if conflicts:
        print(f"      {len(conflicts)} çakışma çözüldü:")
        for c in conflicts:
            print(f"      ⚠  {Path(c['image']).name} | {c['field']}: "
                  f"'{c['kept_value']}' (score {c['kept_score']}) seçildi, "
                  f"'{c['dropped_value']}' (score {c['dropped_score']}) atıldı.")
    else:
        print("      Çakışma yok.")

    if skip_images:
        print(f"      Dışlanan görüntüler ({len(skip_images)}):")
        for img in sorted(skip_images):
            print(f"        - {Path(img).name}")

    write_csv(
        conflicts,
        ["image", "field", "kept_value", "kept_score", "dropped_value", "dropped_score", "reason"],
        conflict_path,
    )
    print(f"      Çakışma raporu → {conflict_path}")

    print("[4/5] Düzeltmeler base annotation'a uygulanıyor...")
    gold_rows = apply_corrections(base_rows, corrections, skip_images)
    applied = sum(1 for k in corrections if k[0] not in skip_images)
    print(f"      {applied} alan düzeltmesi uygulandı.")
    print(f"      {len(base_rows) - len(gold_rows)} görüntü dışlandı.")
    print(f"      Kalan: {len(gold_rows)} görüntü.")

    print("[5/5] Gold CSV yazılıyor...")
    fieldnames = list(base_rows[0].keys()) if base_rows else ["image"]
    write_csv(gold_rows, fieldnames, out_path)
    print(f"      → {out_path}")
    print()
    print("✅ Tamamlandı.")
    print(f"   Gold CSV     : {out_path}")
    print(f"   Conflict CSV : {conflict_path}")
    print()
    print("Sonraki adım — benchmark'ı gold üzerinde çalıştır:")
    print(f"   python benchmark_pipeline.py --annotations {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
