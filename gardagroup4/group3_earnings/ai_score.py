#!/usr/bin/env python3
"""
AI labor / productivity proxy from earnings transcripts.
Combines AI-technology language, productivity/automation, and labor-downside mentions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scoring_utils import (
    AI_LABOR,
    load_companies_rows,
    mean_by_region,
    read_transcript,
    repo_root,
    theme_adjusted_score,
    theme_density_score,
    theme_sentiment,
    word_count,
)


def ai_labor_company_score(text: str) -> float:
    w = word_count(text)
    ai_tech = theme_adjusted_score(text, AI_LABOR["ai_tech"], w)
    prod = theme_adjusted_score(text, AI_LABOR["productivity"], w)
    labor_up = theme_adjusted_score(text, AI_LABOR["labor_up"], w)

    down_density = theme_density_score(text, AI_LABOR["labor_down"], w)
    down_sent = theme_sentiment(text, AI_LABOR["labor_down"])
    penalty = down_density * (0.45 + 0.55 * max(0.0, -down_sent))

    raw = 0.45 * ai_tech + 0.35 * prod + 0.25 * labor_up - 0.35 * penalty
    return round(raw, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI labor / productivity proxy scores by region.")
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
        score = ai_labor_company_score(text)
        rows.append(
            {
                "symbol": r["symbol"],
                "name": r["name"],
                "region": r["region"],
                "sector": r["sector"],
                "ai_labor_score": score,
            }
        )

    per_company = out_dir / "company_ai_scores.csv"
    with per_company.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(
            f, fieldnames=["symbol", "name", "region", "sector", "ai_labor_score"]
        )
        wr.writeheader()
        wr.writerows(rows)

    regional_rows = mean_by_region(rows, value_key="ai_labor_score")
    out_path = out_dir / "regional_ai_scores.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["region", "ai_labor_score", "n_companies"])
        wr.writeheader()
        for rr in regional_rows:
            wr.writerow(
                {
                    "region": rr["region"],
                    "ai_labor_score": rr["ai_labor_score"],
                    "n_companies": rr["n_companies"],
                }
            )

    print(f"Wrote {per_company}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
