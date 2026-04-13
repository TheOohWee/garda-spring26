#!/usr/bin/env python3
"""
Earnings call theme scoring: demand, hiring, pricing, capex, AI, efficiency.
Produces per-company theme scores, a composite earnings signal, and regional means.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scoring_utils import (
    EARNINGS_THEMES,
    load_companies_rows,
    mean_by_region,
    read_transcript,
    repo_root,
    theme_adjusted_score,
    word_count,
)


def compute_company_earnings_row(
    symbol: str, name: str, region: str, sector: str, text: str
) -> dict[str, float | str | int]:
    w = word_count(text)
    row: dict = {
        "symbol": symbol,
        "name": name,
        "region": region,
        "sector": sector,
        "word_count": w,
    }
    theme_vals: list[float] = []
    for theme, kws in EARNINGS_THEMES.items():
        s = theme_adjusted_score(text, kws, w)
        row[f"theme_{theme}"] = round(s, 4)
        theme_vals.append(s)
    row["earnings_composite"] = round(sum(theme_vals) / len(theme_vals), 4)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Earnings call theme scores by region.")
    parser.add_argument(
        "--companies",
        type=Path,
        default=Path("data/companies.csv"),
        help="CSV with symbol, name, region, sector, transcript_path",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output"),
        help="Directory for CSV outputs",
    )
    args = parser.parse_args()
    root = repo_root()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    companies = load_companies_rows(args.companies)
    rows: list[dict] = []
    for r in companies:
        text = read_transcript(Path(r["transcript_path"]))
        rows.append(
            compute_company_earnings_row(
                str(r["symbol"]),
                str(r["name"]),
                str(r["region"]),
                str(r["sector"]),
                text,
            )
        )

    summary_path = out_dir / "transcript_theme_summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    regional_rows = mean_by_region(rows, value_key="earnings_composite")
    regional_path = out_dir / "regional_earnings_scores.csv"
    with regional_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["region", "earnings_composite", "n_companies"])
        wr.writeheader()
        for rr in regional_rows:
            wr.writerow(
                {
                    "region": rr["region"],
                    "earnings_composite": rr["earnings_composite"],
                    "n_companies": rr["n_companies"],
                }
            )

    print(f"Wrote {summary_path}")
    print(f"Wrote {regional_path}")


if __name__ == "__main__":
    main()
