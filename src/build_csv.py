from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import METADATA_CSV_PATH, RAW_HTML_DIR, configure_logging, ensure_directories, parse_args, read_json, transcript_uuid, word_count
from mappings import lookup_company_fields


def build_metadata_frame(us_only: bool = False) -> pd.DataFrame:
    rows: list[dict] = []
    for metadata_path in sorted(RAW_HTML_DIR.glob("*.json")):
        metadata = read_json(metadata_path)
        transcript_path = Path(metadata["transcript_path"])
        if not transcript_path.exists():
            continue

        ticker = (metadata.get("ticker") or "UNKNOWN").upper()
        call_date = metadata.get("call_date") or "1900-01-01"
        quarter = metadata.get("quarter") or "UNKNOWN"
        company = metadata.get("company") or ticker
        country, region, sector = lookup_company_fields(company, ticker)
        if us_only and country != "US":
            continue
        text = transcript_path.read_text(encoding="utf-8")

        rows.append(
            {
                "transcript_id": transcript_uuid(ticker, call_date, quarter),
                "company": company,
                "ticker": ticker,
                "country": country,
                "call_date": call_date,
                "quarter": quarter,
                "fiscal_year": metadata.get("fiscal_year") or "",
                "source_url": metadata.get("source_url") or "",
                "transcript_path": str(transcript_path),
                "word_count": word_count(text),
                "region": region,
                "sector": sector,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["call_date", "ticker", "quarter"], ascending=[False, True, True]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = parse_args("Build transcript metadata CSV.")
    parser.add_argument("--us-only", action="store_true", help="Keep only US companies based on country mapping.")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    ensure_directories()

    frame = build_metadata_frame(us_only=args.us_only)
    frame.to_csv(METADATA_CSV_PATH, index=False)
    print(f"Wrote {len(frame):,} metadata rows to {METADATA_CSV_PATH}")


if __name__ == "__main__":
    main()
