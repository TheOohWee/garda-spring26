from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from common import COMPANIES_CSV_PATH, RAW_HTML_DIR, configure_logging, ensure_directories, parse_args, read_json


def normalize_company_name(name: str) -> str:
    cleaned = " ".join((name or "").replace("Earnings Transcript", "").replace("Earnings Call Transcript", "").split())
    return cleaned.strip(" -:")


def build_company_seed_frame() -> pd.DataFrame:
    names_by_ticker: dict[str, Counter[str]] = defaultdict(Counter)
    for metadata_path in sorted(RAW_HTML_DIR.glob("*.json")):
        metadata = read_json(metadata_path)
        ticker = (metadata.get("ticker") or "").strip().upper()
        company = normalize_company_name(metadata.get("company") or ticker)
        if not ticker:
            continue
        if company:
            names_by_ticker[ticker][company] += 1

    rows: list[dict[str, str]] = []
    for ticker, counts in sorted(names_by_ticker.items()):
        company = counts.most_common(1)[0][0] if counts else ticker
        rows.append(
            {
                "company": company,
                "ticker": ticker,
                "country": "",
                "region": "",
                "sector": "",
            }
        )
    return pd.DataFrame(rows, columns=["company", "ticker", "country", "region", "sector"])


def merge_companies(existing: pd.DataFrame, discovered: pd.DataFrame) -> pd.DataFrame:
    existing = existing.copy() if not existing.empty else pd.DataFrame(columns=["company", "ticker", "country", "region", "sector"])
    if not existing.empty:
        existing.columns = [column.lower() for column in existing.columns]
        for column in ["company", "ticker", "country", "region", "sector"]:
            if column not in existing.columns:
                existing[column] = ""
        existing = existing[["company", "ticker", "country", "region", "sector"]].fillna("")
        existing["ticker"] = existing["ticker"].astype(str).str.upper().str.strip()

    discovered = discovered.copy()
    discovered["ticker"] = discovered["ticker"].astype(str).str.upper().str.strip()

    existing_by_ticker = {row["ticker"]: row for _, row in existing.iterrows()}
    merged_rows: list[dict[str, str]] = []
    for _, row in discovered.iterrows():
        ticker = row["ticker"]
        if ticker in existing_by_ticker:
            current = existing_by_ticker[ticker]
            merged_rows.append(
                {
                    "company": current["company"] or row["company"],
                    "ticker": ticker,
                    "country": current["country"],
                    "region": current["region"],
                    "sector": current["sector"],
                }
            )
        else:
            merged_rows.append(row.to_dict())

    extra_existing = existing.loc[~existing["ticker"].isin(discovered["ticker"])]
    if not extra_existing.empty:
        merged_rows.extend(extra_existing.to_dict(orient="records"))

    merged = pd.DataFrame(merged_rows, columns=["company", "ticker", "country", "region", "sector"])
    merged = merged.drop_duplicates(subset=["ticker"], keep="first")
    return merged.sort_values(["ticker", "company"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = parse_args("Update data/companies.csv from discovered transcript metadata.")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    ensure_directories()

    existing = pd.read_csv(COMPANIES_CSV_PATH).fillna("") if COMPANIES_CSV_PATH.exists() else pd.DataFrame()
    discovered = build_company_seed_frame()
    merged = merge_companies(existing, discovered)
    merged.to_csv(COMPANIES_CSV_PATH, index=False)
    print(f"Wrote {len(merged):,} company rows to {COMPANIES_CSV_PATH}")


if __name__ == "__main__":
    main()
