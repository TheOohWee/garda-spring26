from __future__ import annotations

import pandas as pd

from common import OUTPUTS_DIR, SCORED_CSV_PATH, configure_logging, ensure_directories, parse_args


def main(argv: list[str] | None = None) -> None:
    parser = parse_args("Generate downstream reporting datasets.")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    ensure_directories()

    if not SCORED_CSV_PATH.exists():
        raise FileNotFoundError(f"Scored transcript dataset not found: {SCORED_CSV_PATH}")

    frame = pd.read_csv(SCORED_CSV_PATH)

    transcript_theme_summary = frame[
        [
            "transcript_id",
            "company",
            "ticker",
            "call_date",
            "quarter",
            "region",
            "word_count",
            "theme_demand",
            "theme_hiring",
            "theme_pricing",
            "theme_capex",
            "theme_ai",
            "theme_efficiency",
            "earnings_composite",
        ]
    ].copy()
    transcript_theme_summary.to_csv(OUTPUTS_DIR / "transcript_theme_summary.csv", index=False)

    regional_earnings = (
        frame.groupby("region", dropna=False)
        .agg(
            earnings_composite=("earnings_composite", "mean"),
            n_companies=("ticker", "nunique"),
            n_transcripts=("transcript_id", "count"),
        )
        .reset_index()
    )
    regional_earnings.to_csv(OUTPUTS_DIR / "regional_earnings_scores.csv", index=False)

    company_ai = (
        frame.groupby(["company", "ticker", "region"], dropna=False)
        .agg(ai_labor_score=("ai_labor_score", "mean"), n_transcripts=("transcript_id", "count"))
        .reset_index()
    )
    company_ai.to_csv(OUTPUTS_DIR / "company_ai_scores.csv", index=False)

    regional_ai = (
        company_ai.groupby("region", dropna=False)
        .agg(ai_labor_score=("ai_labor_score", "mean"), n_companies=("ticker", "count"))
        .reset_index()
    )
    regional_ai.to_csv(OUTPUTS_DIR / "regional_ai_scores.csv", index=False)

    company_summary = (
        frame.groupby(["company", "ticker", "region", "sector"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    company_summary.to_csv(OUTPUTS_DIR / "company_summary.csv", index=False)

    numeric_cols = frame.select_dtypes(include="number").columns.tolist()
    score_distributions = frame[numeric_cols].describe().transpose().reset_index().rename(columns={"index": "metric"})
    score_distributions.to_csv(OUTPUTS_DIR / "score_distributions.csv", index=False)

    top_bottom = pd.concat(
        [
            frame.nlargest(20, "sentiment_score").assign(segment="top"),
            frame.nsmallest(20, "sentiment_score").assign(segment="bottom"),
        ],
        ignore_index=True,
    )
    top_bottom.to_csv(OUTPUTS_DIR / "top_bottom.csv", index=False)

    print("Report complete")
    print(f"Transcripts scored: {len(frame):,}")
    print(f"Regions covered: {frame['region'].nunique(dropna=True)}")
    print(f"Companies covered: {frame['ticker'].nunique(dropna=True)}")
    print(f"Mean earnings composite: {frame['earnings_composite'].mean():.4f}")
    print(f"Mean AI labor score: {frame['ai_labor_score'].mean():.4f}")


if __name__ == "__main__":
    main()
