from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def run() -> int:
    parser = argparse.ArgumentParser(description="Prepare train/val splits from OCR annotation dataset")
    parser.add_argument("--annotations", default="dataset/generated/train_annotations.csv", help="Input CSV annotations")
    parser.add_argument("--out-dir", default="dataset/generated/splits", help="Output split directory")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    src = Path(args.annotations)
    if not src.exists():
        raise RuntimeError(f"Annotations not found: {src}")

    df = pd.read_csv(src)
    if "image" not in df.columns:
        raise RuntimeError("CSV must contain 'image' column")

    # Keep only samples with at least one non-empty target field
    target_cols = [
        c
        for c in df.columns
        if c
        not in {
            "image",
            "source_markdown",
        }
    ]
    mask = df[target_cols].fillna("").astype(str).apply(lambda row: any(v.strip() for v in row), axis=1)
    df = df[mask].reset_index(drop=True)

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_val = max(1, int(len(df) * args.val_ratio)) if len(df) > 1 else 0

    val_df = df.iloc[:n_val].copy()
    train_df = df.iloc[n_val:].copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = out_dir / "train.csv"
    val_csv = out_dir / "val.csv"
    train_df.to_csv(train_csv, index=False, encoding="utf-8")
    val_df.to_csv(val_csv, index=False, encoding="utf-8")

    summary = {
        "input_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "target_columns": target_cols,
        "train_csv": train_csv.as_posix(),
        "val_csv": val_csv.as_posix(),
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

