# Garda Spring 26

Project scaffold for building dated regional signals from macro data, central bank commentary, and earnings / AI-labor commentary.

## Repository Layout

- `src/regional_signals/`: Python package for signal building, integration, and backtesting
- `configs/`: shared schema, region, and target-asset configuration
- `data/raw/`: source inputs by domain
- `data/interim/`: cleaned and normalized inputs
- `data/processed/`: exported scores, composites, and backtest outputs
- `outputs/`: CSVs, charts, and reports
- `notebooks/`: exploration and analysis notebooks
- `tests/`: smoke tests and validation checks

## Setup

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the pipeline modules from `src/regional_signals/` as they are implemented.

## Common Output Contract

Every exported score table should include:

- `as_of_date`
- `effective_date`
- `region`
- `pillar`
- `subpillar`
- `score_raw`
- `score_std`
- `coverage`
- `source_count`

## Motley Fool Transcript Scraper

Use Selenium to search and scrape earnings call transcript pages from Motley Fool.

1. Install dependencies: `pip install -r requirements.txt`
2. Run default batch (10 tickers, FY2020-FY2025):
   `python scrape_motley_fool_transcripts.py`
3. Custom run example:
   `python scrape_motley_fool_transcripts.py --tickers MSFT,AAPL,NVDA --fy-start 2020 --fy-end 2025`
4. If search engines block headless traffic, run visible Chrome and optionally wait on search pages:
   `python scrape_motley_fool_transcripts.py --headful --manual-pause-seconds 6`
5. Run five concurrent browser workers:
   `python scrape_motley_fool_transcripts.py --workers 5 --headful`

Output is written under:
- `data/raw/earnings/motley_fool_transcripts/<TICKER>/FY<YEAR>.txt`
- `data/raw/earnings/motley_fool_transcripts/manifest.csv`
