import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from tabulate import tabulate


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

REGION_ORDER = ["US", "Europe", "Japan", "UK", "LatAm"]

GROUP1_MACRO_PATHS = {
    "US": SCRIPT_DIR / "group1_macro" / "US.csv",
    "Europe": SCRIPT_DIR / "group1_macro" / "Europe.csv",
    "Japan": SCRIPT_DIR / "group1_macro" / "Japan.csv",
}

POLICY_REGION_MAP = {
    "US Federal Reserve": "US",
    "European Central Bank": "Europe",
    "Bank of Japan": "Japan",
    "Bank of England": "UK",
}

EARNINGS_REGION_MAP = {
    "North_America": "US",
    "Asia_Pacific": "Japan",
    "Latin_America": "LatAm",
}

PILLARS = [
    ("Macro_Score", "Macro"),
    ("Policy_Score", "Policy"),
    ("Earnings_Score", "Earnings"),
    ("AI_Score", "AI"),
]

# Updated to custom weighting
PILLAR_WEIGHTS = {
    "Macro_Score": 0.30,
    "Policy_Score": 0.30,
    "Earnings_Score": 0.20,
    "AI_Score": 0.20,
}
NEUTRAL_SCORE = 0.5

WARNINGS: list[str] = []

def append_warning(message: str) -> None:
    WARNINGS.append(message)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(value):
    try:
        if isinstance(value, str):
            return float(value.replace(",", "").strip())
        return float(value)
    except Exception:
        return np.nan


def load_group1_macro() -> pd.DataFrame:
    records = []
    for region, path in GROUP1_MACRO_PATHS.items():
        if not path.exists():
            append_warning(f"Missing macro source for {region}: {path}")
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        latest = df.iloc[-1]
        if region == "US":
            values = {
                "GDP_Growth": safe_float(latest.get("GDP YoY % Change")),
                "Core_CPI_Inv": -safe_float(latest.get("Core CPI")),
                "Unemp_Inv": -safe_float(latest.get("Unemployment Levels %")),
            }
        elif region == "Europe":
            values = {
                "GDP_Growth": safe_float(latest.get("GDP Growth")),
                "Core_CPI_Inv": -safe_float(latest.get("Core HICP")),
                "Unemp_Inv": -safe_float(latest.get("Unemployment")),
            }
        elif region == "Japan":
            values = {
                "GDP_Growth": safe_float(latest.get("GDP YoY % Change")),
                "Core_CPI_Inv": -safe_float(latest.get("Core CPI")),
                "Unemp_Inv": -safe_float(latest.get("Unemployment Levels %")),
            }
        else:
            values = {}

        records.append({"Region": region, **values})

    df_macro = pd.DataFrame(records)
    if df_macro.empty:
        return pd.DataFrame(columns=["Region", "Macro_Score"])

    for metric in ["GDP_Growth", "Core_CPI_Inv", "Unemp_Inv"]:
        df_macro[f"{metric}_Z"] = (df_macro[metric] - df_macro[metric].mean()) / df_macro[metric].std(ddof=0)

    df_macro["Macro_Score"] = df_macro[["GDP_Growth_Z", "Core_CPI_Inv_Z", "Unemp_Inv_Z"]].mean(axis=1)
    df_macro = df_macro[["Region", "Macro_Score"]]
    return df_macro


def load_group2_policy() -> pd.DataFrame:
    path = SCRIPT_DIR / "group2_policy" / "outputs" / "2_daily_macro_scores_by_bank.csv"
    if not path.exists():
        append_warning(f"Missing policy file: {path}")
        return pd.DataFrame(columns=["Region", "Policy_Score"])
    df = pd.read_csv(path)
    df["Region"] = df["Source_Bank"].map(POLICY_REGION_MAP)
    df = df.dropna(subset=["Region"])
    df["Daily_Macro_Score"] = pd.to_numeric(df["Daily_Macro_Score"], errors="coerce")
    df_policy = df.groupby("Region")["Daily_Macro_Score"].mean().reset_index()
    df_policy.rename(columns={"Daily_Macro_Score": "Policy_Score"}, inplace=True)
    return df_policy


def load_group3_earnings_ai() -> pd.DataFrame:
    earnings_path = SCRIPT_DIR / "group3_earnings" / "output" / "regional_earnings_scores.csv"
    ai_path = SCRIPT_DIR / "group3_earnings" / "output" / "regional_ai_scores.csv"
    if not earnings_path.exists() or not ai_path.exists():
        append_warning(f"Missing Group 3 outputs: {earnings_path} or {ai_path}")
        return pd.DataFrame(columns=["Region", "Earnings_Score", "AI_Score"])

    df_earn = pd.read_csv(earnings_path)
    df_ai = pd.read_csv(ai_path)

    df_earn["Region"] = df_earn["region"].map(EARNINGS_REGION_MAP).fillna(df_earn["region"])
    df_ai["Region"] = df_ai["region"].map(EARNINGS_REGION_MAP).fillna(df_ai["region"])

    df_g3 = pd.merge(
        df_earn[["Region", "earnings_composite"]],
        df_ai[["Region", "ai_labor_score"]],
        on="Region",
        how="outer",
    )
    df_g3.rename(columns={"earnings_composite": "Earnings_Score", "ai_labor_score": "AI_Score"}, inplace=True)
    return df_g3


def scale_to_range(series: pd.Series, minimum: float = 0.0, maximum: float = 1.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series([NEUTRAL_SCORE] * len(series), index=series.index)
    min_val = numeric.min()
    max_val = numeric.max()
    if min_val == max_val:
        return pd.Series([NEUTRAL_SCORE] * len(series), index=series.index)
    scaled = (numeric - min_val) / (max_val - min_val) * (maximum - minimum) + minimum
    return scaled.clip(lower=minimum, upper=maximum)


def build_composite_table(macro_df: pd.DataFrame, policy_df: pd.DataFrame, g3_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.DataFrame({"Region": REGION_ORDER})
    merged = merged.merge(macro_df, on="Region", how="left")
    merged = merged.merge(policy_df, on="Region", how="left")
    merged = merged.merge(g3_df, on="Region", how="left")

    for col, _ in PILLARS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged[f"{col}_Scaled"] = scale_to_range(merged[col])
        merged[f"{col}_Present"] = merged[col].notna().astype(int)
        
        # Fill missing values with neutral score
        merged[f"{col}_Scaled"] = merged[f"{col}_Scaled"].fillna(NEUTRAL_SCORE)
        
        # Calculate Weighted Contribution
        weight = PILLAR_WEIGHTS.get(col, 0.0)
        merged[f"{col}_Contribution"] = merged[f"{col}_Scaled"] * weight

    merged["Coverage"] = merged[[f"{col}_Present" for col, _ in PILLARS]].sum(axis=1)
    merged["Composite_Score_Adjusted"] = merged[[f"{col}_Contribution" for col, _ in PILLARS]].sum(axis=1)

    # Sorting and Ranking
    merged = merged.sort_values(["Composite_Score_Adjusted", "Coverage"], ascending=[False, False]).reset_index(drop=True)
    merged["Rank"] = merged.index + 1

    # Trade Action logic
    merged["Trade_Action"] = "Neutral"
    if not merged.empty:
        merged.loc[0, "Trade_Action"] = "Long"
        merged.loc[len(merged) - 1, "Trade_Action"] = "Short"

    # Define Regimes based on score thresholds (Matches your Proposal Graph)
    def get_regime(score):
        if score >= 0.60: return "Expansion"
        if score <= 0.40: return "Contraction"
        return "Recovery/Neutral"

    # CRITICAL: This line must be indented inside the function!
    merged["Regime"] = merged["Composite_Score_Adjusted"].apply(get_regime)

    return merged


def build_trade_ideas(df: pd.DataFrame) -> pd.DataFrame:
    top = df.iloc[0]
    bottom = df.iloc[-1]
    
    # Identify the highest scoring component for the top ranked region
    top_pillar = df[[f"{col}_Scaled" for col, _ in PILLARS]].loc[top.name].idxmax().replace("_Scaled", "")
    top_pillar_name = next(label for col, label in PILLARS if col == top_pillar)

    ideas = [
        {
            "Idea_ID": 1,
            "Trade_Type": "Relative Value Pair",
            "Long_Leg": top["Region"],
            "Short_Leg": bottom["Region"],
            "Primary_Driver": top_pillar_name,
        },
        {
            "Idea_ID": 2,
            "Trade_Type": "Outright Beta / Growth",
            "Long_Leg": top["Region"],
            "Short_Leg": "Market Benchmark",
            "Primary_Driver": top_pillar_name,
        }
    ]

    return pd.DataFrame(ideas)


def render_charts(df: pd.DataFrame) -> None:
    if df.empty:
        return

    # Chart 1: Composite Score Ranking
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ca02c" if action == "Long" else "#d62728" if action == "Short" else "#7f7f7f" for action in df["Trade_Action"]]
    ax.barh(df["Region"], df["Composite_Score_Adjusted"], color=colors)
    ax.set_xlabel("Adjusted Composite Score")
    ax.set_title("Regional Composite Score Ranking")
    for i, score in enumerate(df["Composite_Score_Adjusted"]):
        ax.text(score + 0.01, i, f"{score:.2f}", va="center")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "composite_scores_chart.png", dpi=300)
    plt.close(fig)

    # Chart 2: Stacked Contributions (now perfectly aligns with Chart 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(df))
    for col, label in PILLARS:
        ax.barh(df["Region"], df[f"{col}_Contribution"], left=bottom, label=label)
        bottom += df[f"{col}_Contribution"].values
    
    ax.set_xlabel("Weighted Pillar Contribution (Adds exactly to Composite Score)")
    ax.set_title("Pillar Contribution by Region")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pillar_contributions_chart.png", dpi=300)
    plt.close(fig)


def build_report(df: pd.DataFrame, trades: pd.DataFrame) -> str:
    lines = [
        "GLOBAL COMPOSITE SCORE REPORT",
        "===============================================",
        "",
    ]

    if WARNINGS:
        lines.append("WARNINGS:")
        for warning in WARNINGS:
            lines.append(f"   - {warning}")
        lines.append("")

    lines.extend([
        "1. DATA SOURCES:",
        "   - Group 1 Macro: US, Europe, Japan latest macro metrics",
        "   - Group 2 Policy: central bank NLP daily macro sentiment",
        "   - Group 3 Earnings/AI: regional earnings and AI labor signals",
        "",
        "2. APPROACH:",
        "   - Standardize each pillar to a 0-1 comparable scale",
        "   - Replace missing pillar values with neutral 0.5 scores",
        "   - Apply strategic weights (Macro: 30%, Policy: 30%, Earnings: 20%, AI: 20%)",
        "   - Sum weighted scores into the final adjusted composite",
        "   - Rank regions and generate trade ideas from the top and bottom regions",
        "",
        "3. REGIONAL RANKINGS:",
    ])
    lines.append(df[["Rank", "Region", "Composite_Score_Adjusted", "Coverage", "Trade_Action"]].to_string(index=False))
    lines.extend([
        "",
        "4. GENERATED TRADE SIGNALS:",
    ])
    
    if trades.empty:
        lines.append("   [Insufficient data to generate trade signals]")
    else:
        lines.append(trades.to_string(index=False))
        
    lines.append("")
    lines.append("Charts saved to outputs/composite_scores_chart.png and outputs/pillar_contributions_chart.png")
    return "\n".join(lines)


def render_regime_map(df: pd.DataFrame) -> None:
    if df.empty:
        return

    print("Generating Regime Map...")
    
    try:
        import geodatasets
        # Use ADMIN_0 (Countries) instead of LAND (Physical)
        path = geodatasets.get_path('naturalearth.admin_0_countries')
        world = gpd.read_file(path)
    except Exception as e:
        print(f"Primary map source failed, trying fallback... ({e})")
        url = "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson"
        world = gpd.read_file(url)
    
    # Standardize column names to lowercase to avoid 'Name' vs 'name' issues
    world.columns = world.columns.str.lower()
    
    # Check if 'name' exists, if not, try 'name_en' or 'admin'
    name_col = 'name'
    if 'name' not in world.columns:
        for col in ['name_en', 'admin', 'name_long']:
            if col in world.columns:
                name_col = col
                break

    region_map = {
    "US": ["United States of America"],
    "Europe": ["France", "Germany", "Italy", "Spain", "Poland", "Netherlands", "Belgium", 
               "Austria", "Portugal", "Greece", "Sweden", "Norway", "Finland", "Denmark", "Switzerland"],
    "LatAm": ["Brazil", "Mexico", "Argentina", "Chile", "Colombia", "Peru", "Ecuador", "Bolivia", "Paraguay", "Uruguay"],
    "UK": ["United Kingdom"],
    "Japan": ["Japan"]
}
    
    world['regime_score'] = np.nan
    for _, row in df.iterrows():
        countries = region_map.get(row['Region'], [])
        # Use the name_col we identified
        world.loc[world[name_col].isin(countries), 'regime_score'] = row['Composite_Score_Adjusted']

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    # Plot with 'RdYlGn'
    world.plot(
        column='regime_score', 
        ax=ax, 
        cmap='RdYlGn', 
        legend=True, 
        missing_kwds={'color': '#f0f0f0'}, 
        legend_kwds={'label': "Regime Strength", 'orientation': "horizontal", 'pad': 0.05}
    )
    
    ax.set_title("Global Investment Regime Classification", fontsize=16, pad=20)
    ax.set_axis_off()
    
    plt.savefig(OUTPUT_DIR / "regime_map.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"SUCCESS: Map saved to {OUTPUT_DIR}/regime_map.png")


def main() -> None:
    ensure_output_dir()
    
    print("Loading Group 1: Macro Data...")
    macro_df = load_group1_macro()
    if not macro_df.empty:
        print(tabulate(macro_df.round(3), headers='keys', tablefmt='psql', showindex=False))
        
    print("\nLoading Group 2: Policy Data...")
    policy_df = load_group2_policy()
    if not policy_df.empty:
        print(tabulate(policy_df.round(3), headers='keys', tablefmt='psql', showindex=False))
        
    print("\nLoading Group 3: Earnings & AI Data...")
    g3_df = load_group3_earnings_ai()
    if not g3_df.empty:
        print(tabulate(g3_df.round(3), headers='keys', tablefmt='psql', showindex=False))

    df = build_composite_table(macro_df, policy_df, g3_df)
    
    print("\n--- Final Merged Composite Data ---")
    display_cols = ["Rank", "Region", "Composite_Score_Adjusted", "Regime", "Coverage"] 
    print(tabulate(df[display_cols].round(3), headers='keys', tablefmt='psql', showindex=False))

    # Defensive check: Ensure at least two regions have actual data
    valid_regions_count = len(df[df["Coverage"] > 0])
    
    if valid_regions_count < 2:
        print("\nWARNING: Less than two regions have valid data.")
        print("Composite ranking and cross-region trade ideas are not reliable until more data is added.")
        trades = pd.DataFrame(columns=["Idea_ID", "Trade_Type", "Long_Leg", "Short_Leg", "Primary_Driver"])
    else:
        trades = build_trade_ideas(df)
        print("\n--- Generated Trade Signals ---")
        print(tabulate(trades, headers='keys', tablefmt='psql', showindex=False))

    # Output generation
    df.to_csv(OUTPUT_DIR / "final_composite_score_table.csv", index=False)
    df[["Rank", "Region", "Composite_Score_Adjusted", "Coverage", "Trade_Action"]].to_csv(
        OUTPUT_DIR / "regional_rankings.csv", index=False
    )
    trades.to_csv(OUTPUT_DIR / "trade_ideas.csv", index=False)

    render_charts(df)
    render_regime_map(df)

    report_text = build_report(df, trades)
    with open(OUTPUT_DIR / "summary_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)


if __name__ == "__main__":
    main()