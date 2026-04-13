from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

import pandas as pd

from common import load_companies_override


@dataclass(frozen=True, slots=True)
class CompanyMapping:
    company: str
    ticker: str
    country: str
    region: str
    sector: str


DEFAULT_COMPANY_MAPPINGS = [
    CompanyMapping("Apple", "AAPL", "US", "North America", "Technology"),
    CompanyMapping("Microsoft", "MSFT", "US", "North America", "Technology"),
    CompanyMapping("Amazon", "AMZN", "US", "North America", "Consumer Discretionary"),
    CompanyMapping("Alphabet", "GOOGL", "US", "North America", "Communication Services"),
    CompanyMapping("NVIDIA", "NVDA", "US", "North America", "Technology"),
    CompanyMapping("Meta Platforms", "META", "US", "North America", "Communication Services"),
    CompanyMapping("Tesla", "TSLA", "US", "North America", "Consumer Discretionary"),
    CompanyMapping("ASML", "ASML", "NL", "Europe", "Technology"),
    CompanyMapping("SAP", "SAP", "DE", "Europe", "Technology"),
    CompanyMapping("Toyota", "TM", "JP", "Asia Pacific", "Consumer Discretionary"),
    CompanyMapping("Sony", "SONY", "JP", "Asia Pacific", "Communication Services"),
    CompanyMapping("Alibaba", "BABA", "CN", "Asia Pacific", "Consumer Discretionary"),
]


def build_company_mapping_frame() -> pd.DataFrame:
    defaults = pd.DataFrame([asdict(mapping) for mapping in DEFAULT_COMPANY_MAPPINGS])
    overrides = load_companies_override()
    if overrides.empty:
        return defaults

    overrides = overrides.rename(columns=str.lower)
    merged = defaults.set_index("ticker")
    for _, row in overrides.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        existing = merged.loc[ticker].to_dict() if ticker in merged.index else {"company": ticker, "country": "Unknown", "region": "Unknown", "sector": "Unknown"}
        merged.loc[ticker, "company"] = row.get("company", "") or existing["company"]
        merged.loc[ticker, "country"] = row.get("country", "") or existing["country"]
        merged.loc[ticker, "region"] = row.get("region", "") or existing["region"]
        merged.loc[ticker, "sector"] = row.get("sector", "") or existing["sector"]

    return merged.reset_index(names="ticker")


def lookup_company_fields(company: str | None, ticker: str | None) -> tuple[str, str, str]:
    frame = build_company_mapping_frame()
    ticker_norm = (ticker or "").upper()
    company_norm = (company or "").strip().lower()

    if ticker_norm:
        match = frame.loc[frame["ticker"].str.upper() == ticker_norm]
        if not match.empty:
            row = match.iloc[0]
            return str(row.get("country", "Unknown")), str(row.get("region", "Unknown")), str(row.get("sector", "Unknown"))

    if company_norm:
        match = frame.loc[frame["company"].str.lower() == company_norm]
        if not match.empty:
            row = match.iloc[0]
            return str(row.get("country", "Unknown")), str(row.get("region", "Unknown")), str(row.get("sector", "Unknown"))

    return "Unknown", "Unknown", "Unknown"
