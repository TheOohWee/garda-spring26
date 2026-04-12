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
