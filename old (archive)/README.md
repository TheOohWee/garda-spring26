# Group 4 Integration + Portfolio Output

Integrates Group 1 (Macro), Group 2 (Policy), and Group 3 (Earnings/AI) into a unified regional ranking and trade idea pipeline.

## How to run

    python3 main_model

## Pipeline Logic (Step-by-Step)

1. **Extraction:** The script targets specific regional CSVs across three distinct research groups.
2. **Metric Inversion:** For Macro data (Group 1), "Negative" indicators like CPI and Unemployment are inverted so that a higher value always equals a "better" economic outlook.
3. **Pillar Scaling (0 to 1):** Each pillar (Macro, Policy, Earnings, AI) is Min-Max scaled. This ensures that a "0.8" in AI is mathematically comparable to a "0.8" in Central Bank Policy.
4. **Neutral Padding:** If a region is missing a specific data file, it is assigned a **0.5 (Neutral)** score for that pillar.
5. **Weighted Summation:** Scores are multiplied by their strategic weights (30/30/20/20) and summed to create the **Adjusted Composite Score**.
6. **Relative Ranking:** Regions are ranked by score. The highest becomes the "Long" candidate; the lowest becomes the "Short" candidate.

## Generated outputs (in `outputs/`)

- `final_composite_score_table.csv`
- `regional_rankings.csv`
- `trade_ideas.csv`
- `summary_report.txt`
- `composite_scores_chart.png`
- `pillar_contributions_chart.png`

## Data Usage Notes

- **Group 1 (Macro):**
    - **Used:** `US.csv`, `Europe.csv`, and `Japan.csv` are utilized for GDP, CPI, and Unemployment metrics.
- **Group 2 (Policy):**
    - **Used:** `2_daily_macro_scores_by_bank.csv` is used for aggregate central bank sentiment.
    - **Not Used:** `1_detailed_sentences.csv` and `3_daily_topic_scores_by_bank.csv` (too granular).
- **Group 3 (Earnings & AI):**
    - **Used:** `regional_earnings_scores.csv` and `regional_ai_scores.csv` for regional composites.
    - **Not Used:** `company_ai_scores.csv` and `transcript_theme_summary.csv` (bottom-up focus).

## Methodology Notes

- **Standardization:** All pillars are forced into a **0-1 scale** for alignment.
- **Trade Safeguards:** The script will issue a hard-stop warning if fewer than two regions have valid data, as relative value pair trades require a comparison.
- **Synchronized Charts:** The stacked pillar chart is built using the weighted contributions, meaning it sums exactly to the value shown in the ranking chart.