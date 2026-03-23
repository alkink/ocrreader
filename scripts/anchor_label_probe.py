from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocrreader.config import load_config
from ocrreader.io_utils import imread_color
from ocrreader.ocr_engine import create_ocr_engine
from ocrreader.preprocess import preprocess_document
from ocrreader.text_utils import normalize_for_match


def run() -> int:
    parser = argparse.ArgumentParser(description="Probe label hits for anchor aliases on one image")
    parser.add_argument("--image", required=True, help="Image path")
    parser.add_argument("--config", default="config/ruhsat_schema.yaml", help="YAML config")
    parser.add_argument(
        "--out-csv",
        default="output/anchor_label_probe.csv",
        help="Output CSV with alias hit candidates",
    )
    parser.add_argument(
        "--words-csv",
        default="",
        help="Optional output CSV for all OCR words (default: <out-csv-stem>_words.csv)",
    )
    parser.add_argument("--psm", type=int, default=6, help="PSM for full-page OCR words")
    args = parser.parse_args()

    cfg = load_config(args.config)
    engine = create_ocr_engine(cfg.ocr)

    img = imread_color(args.image)
    prep = preprocess_document(img, cfg.pipeline)
    words = engine.iter_words(prep.normalized_image, psm=args.psm, min_conf=0.0)

    word_rows: list[dict[str, object]] = []
    for w in words:
        nx = w.bbox.x / max(1, prep.normalized_image.shape[1])
        ny = w.bbox.y / max(1, prep.normalized_image.shape[0])
        word_rows.append(
            {
                "text": w.text,
                "norm_text": normalize_for_match(w.text),
                "conf": round(float(w.conf), 2),
                "x": w.bbox.x,
                "y": w.bbox.y,
                "w": w.bbox.w,
                "h": w.bbox.h,
                "norm_x": round(nx, 4),
                "norm_y": round(ny, 4),
                "block": w.block_num,
                "par": w.par_num,
                "line": w.line_num,
            }
        )

    hits: list[dict[str, object]] = []
    by_line: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    for r in word_rows:
        key = (int(r["block"]), int(r["par"]), int(r["line"]))
        by_line.setdefault(key, []).append(r)

    for anchor_name, ac in cfg.anchors.items():
        for alias in ac.aliases:
            alias_norm = normalize_for_match(alias)
            alias_tokens = [t for t in alias_norm.split(" ") if t]
            if not alias_tokens:
                continue

            for key, line_words in by_line.items():
                toks = [str(w["norm_text"]) for w in line_words]
                raw = " ".join(str(w["text"]) for w in line_words)
                joined = " ".join(toks)

                # cheap containment score
                overlap = sum(1 for t in alias_tokens if t in toks)
                if overlap == 0:
                    continue

                best_x = min(int(w["x"]) for w in line_words)
                best_y = min(int(w["y"]) for w in line_words)
                hits.append(
                    {
                        "anchor": anchor_name,
                        "alias": alias,
                        "alias_norm": alias_norm,
                        "line_block": key[0],
                        "line_par": key[1],
                        "line_no": key[2],
                        "token_overlap": overlap,
                        "alias_token_count": len(alias_tokens),
                        "line_norm_text": joined,
                        "line_raw_text": raw,
                        "x": best_x,
                        "y": best_y,
                        "norm_x": round(best_x / max(1, prep.normalized_image.shape[1]), 4),
                        "norm_y": round(best_y / max(1, prep.normalized_image.shape[0]), 4),
                    }
                )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    words_csv = Path(args.words_csv) if args.words_csv.strip() else out_csv.with_name(f"{out_csv.stem}_words.csv")
    with words_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "text",
            "norm_text",
            "conf",
            "x",
            "y",
            "w",
            "h",
            "norm_x",
            "norm_y",
            "block",
            "par",
            "line",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in word_rows:
            w.writerow(r)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "anchor",
            "alias",
            "alias_norm",
            "line_block",
            "line_par",
            "line_no",
            "token_overlap",
            "alias_token_count",
            "line_norm_text",
            "line_raw_text",
            "x",
            "y",
            "norm_x",
            "norm_y",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(hits, key=lambda x: (x["anchor"], -int(x["token_overlap"]))):
            w.writerow(r)

    print(f"wrote {len(word_rows)} OCR words -> {words_csv.as_posix()}")
    print(f"wrote {len(hits)} alias-hit rows -> {out_csv.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

