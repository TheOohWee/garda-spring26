import time
import requests
import pandas as pd

# Replace with your NEW FRED API key
API_KEY = "f76929c05337c63998256783557f5de0"

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
START_DATE = "2006-01-01"

SERIES = {
    "A191RO1Q156NBEA": "GDP_YoY_Pct",
    "CPIAUCSL": "CPI",
    "CPILFESL": "Core_CPI",
    "UNRATE": "Unemployment_Rate",
}


def fetch_series(series_id: str) -> pd.DataFrame:
    """Fetch one FRED series from 2006 onward."""
    params = {
        "series_id": series_id,
        "api_key": API_KEY,
        "file_type": "json",
        "observation_start": START_DATE,
    }

    # simple retry logic for temporary errors / rate limits
    for attempt in range(5):
        resp = requests.get(BASE_URL, params=params, timeout=30)

        if resp.status_code == 429:
            wait = 2 ** attempt
            print(f"Rate limited on {series_id}. Waiting {wait} seconds...")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        payload = resp.json()
        observations = payload.get("observations", [])

        rows = []
        for obs in observations:
            value = obs.get("value")
            if value == "." or value is None:
                value = None
            else:
                value = float(value)

            rows.append({
                "date": obs.get("date"),
                "series_id": series_id,
                "series_name": SERIES[series_id],
                "value": value,
            })

        return pd.DataFrame(rows)

    raise RuntimeError(f"Failed to fetch {series_id} after multiple retries.")


def main():
    dfs = []

    for series_id, series_name in SERIES.items():
        print(f"Fetching {series_name} ({series_id})...")
        df = fetch_series(series_id)
        dfs.append(df)
        time.sleep(0.6)  # stay under FRED rate limits

    out = pd.concat(dfs, ignore_index=True)

    if out.empty:
        print("No data found.")
        return

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out = out[out["date"] >= pd.Timestamp(START_DATE)]
    out = out.sort_values(["series_name", "date"]).reset_index(drop=True)

    # save long format
    out.to_csv("fred_macro_long.csv", index=False)

    # save wide format
    wide = out.pivot(index="date", columns="series_name", values="value").reset_index()
    wide.to_csv("fred_macro_wide.csv", index=False)

    print("\nSaved:")
    print(" - fred_macro_long.csv")
    print(" - fred_macro_wide.csv")

    print("\nLatest values:")
    latest = (
        out.dropna(subset=["value"])
           .sort_values("date")
           .groupby("series_name", as_index=False)
           .tail(1)[["series_name", "date", "value"]]
           .sort_values("series_name")
    )
    print(latest.to_string(index=False))


if __name__ == "__main__":
    main()