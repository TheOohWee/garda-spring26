
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# -------------------------
# FILE PATHS
# -------------------------
macro_file = 'Garda_Macro_Scoring_and_Data.xlsx'
policy_file = '2_daily_macro_scores_by_bank.csv'
earnings_file = 'regional_earnings_scores.csv'
ai_file = 'regional_ai_scores.csv'

# -------------------------
# Z-SCORE FUNCTION
# -------------------------

def zscore(series):
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([0] * len(series), index=series.index)
    return (series - series.mean()) / std

# -------------------------
# GROUP 1: MACRO
# -------------------------

xls = pd.ExcelFile(macro_file)
print("Available macro sheets:", xls.sheet_names)

# NOTE: keep exact sheet name as it appears in workbook
us_macro = pd.read_excel(macro_file, sheet_name='US Macro Scoring ', header=None)

# Only US available for now
us_value = us_macro.iloc[51, 2]   # verify this cell is correct

macro = pd.DataFrame({
    'Region': ['US'],
    'Macro': [us_value]
})

print("\nMacro data:")
print(tabulate(macro, headers='keys', tablefmt='psql', showindex=False))

# -------------------------
# GROUP 2: POLICY
# -------------------------
policy = pd.read_csv(policy_file)

policy = policy.rename(columns={
    'Source_Bank': 'Bank',
    'Daily_Macro_Score': 'Score'
})

bank_to_region = {
    'US Federal Reserve': 'US',
    'European Central Bank': 'EU',
    'Bank of Japan': 'Japan',
    'Bank of England': 'UK'
}

policy['Region'] = policy['Bank'].map(bank_to_region)
policy = policy.dropna(subset=['Region'])
policy = policy.groupby('Region', as_index=False)['Score'].mean()
policy = policy.rename(columns={'Score': 'Policy'})

print("\nPolicy data:")
print(tabulate(policy, headers='keys', tablefmt='psql', showindex=False))

# -------------------------
# GROUP 3: EARNINGS + AI
# -------------------------
earnings = pd.read_csv(earnings_file)
ai = pd.read_csv(ai_file)

earnings = earnings.rename(columns={
    'region': 'Region',
    'earnings_composite': 'Earnings'
})

ai = ai.rename(columns={
    'region': 'Region',
    'ai_labor_score': 'AI'
})

earnings = earnings[['Region', 'Earnings']]
ai = ai[['Region', 'AI']]

group3 = earnings.merge(ai, on='Region', how='inner')

# Temporary region alignment assumptions
region_map_group3 = {
    'North_America': 'US',
    'Asia_Pacific': 'Japan'
    # Latin_America excluded for now
}

group3['Region'] = group3['Region'].map(region_map_group3)
group3 = group3.dropna(subset=['Region'])
group3 = group3.groupby('Region', as_index=False)[['Earnings', 'AI']].mean()

print("\nGroup 3 data:")
print(tabulate(group3, headers='keys', tablefmt='psql', showindex=False))

# -------------------------
# MERGE EVERYTHING
# -------------------------
df = macro.merge(policy, on='Region', how='inner') \
          .merge(group3, on='Region', how='inner')

print("\nMerged data:")
print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

# -------------------------
# CHECK WHETHER RANKING IS VALID
# -------------------------
if len(df) < 2:
    print("\nWARNING: Only one region has complete data right now.")
    print("Composite ranking and cross-region trade ideas are not valid until ECB/Japan macro scores are added.")
    print("\nCurrent merged table:")
    print(tabulate(df.round(3), headers='keys', tablefmt='psql', showindex=False))

else:
    for col in ['Macro', 'Policy', 'Earnings', 'AI']:
        df[col + '_z'] = zscore(df[col])

    df['Composite'] = (
        0.3 * df['Macro_z'] +
        0.3 * df['Policy_z'] +
        0.2 * df['Earnings_z'] +
        0.2 * df['AI_z']
    )

    df = df.sort_values(by='Composite', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1

    final_table = df[['Rank', 'Region', 'Macro', 'Policy', 'Earnings', 'AI', 'Composite']].round(3)

    print("\nFinal composite table:")
    print(tabulate(final_table, headers='keys', tablefmt='psql', showindex=False))

    final_table.to_csv('composite_scores.csv', index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(df['Region'], df['Composite'])
    plt.title('Composite Scores by Region')
    plt.xlabel('Region')
    plt.ylabel('Composite Score')
    plt.tight_layout()
    plt.savefig('composite_scores_chart.png')
    plt.show()