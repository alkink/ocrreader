from __future__ import annotations

import argparse
import csv
from pathlib import Path


def run() -> int:
    parser = argparse.ArgumentParser(description="Build manual review queue from latest dataset matching outputs")
    parser.add_argument("--qa-dir", default="dataset/generated/qa", help="QA directory containing match reports")
    parser.add_argument(
        "--out",
        default="dataset/generated/qa/latest_dataset_manual_review.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--low-score-threshold",
        type=int,
        default=160,
        help="Include applied rows with score below this threshold for manual review",
    )
    args = parser.parse_args()

    qa_dir = Path(args.qa_dir)
    match_path = qa_dir / "latest_dataset_match_report.csv"
    change_path = qa_dir / "latest_dataset_applied_changes.csv"

    if not match_path.exists():
        raise RuntimeError(f"Missing match report: {match_path}")
    if not change_path.exists():
        raise RuntimeError(f"Missing applied changes report: {change_path}")

    with match_path.open("r", encoding="utf-8", newline="") as f:
        match_rows = list(csv.DictReader(f))
    with change_path.open("r", encoding="utf-8", newline="") as f:
        change_rows = list(csv.DictReader(f))

    review_rows: list[dict[str, str]] = []
    low_score_lines = {
        r.get("matched_line", "")
        for r in match_rows
        if (r.get("status") == "applied" and int(r.get("score") or 0) < args.low_score_threshold)
    }

    for r in match_rows:
        status = r.get("status", "")
        if status in {"ambiguous_skip", "unmatched", "low_score_skip"}:
            review_rows.append(
                {
                    "type": "match",
                    "latest_line": r.get("latest_line", ""),
                    "matched_line": r.get("matched_line", ""),
                    "image": r.get("matched_image", ""),
                    "field": "",
                    "old": "",
                    "new": "",
                    "score": r.get("score", ""),
                    "status": status,
                    "reason": r.get("reasons", ""),
                }
            )

    for r in change_rows:
        if r.get("matched_line", "") in low_score_lines:
            review_rows.append(
                {
                    "type": "change",
                    "latest_line": r.get("latest_line", ""),
                    "matched_line": r.get("matched_line", ""),
                    "image": r.get("image", ""),
                    "field": r.get("field", ""),
                    "old": r.get("old", ""),
                    "new": r.get("new", ""),
                    "score": r.get("score", ""),
                    "status": "low_score_applied",
                    "reason": "",
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "type",
                "latest_line",
                "matched_line",
                "image",
                "field",
                "old",
                "new",
                "score",
                "status",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(review_rows)

    print(f"manual_review_rows={len(review_rows)} out={out_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

