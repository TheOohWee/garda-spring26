# garda1

`garda1` is Group 3's earnings call analysis application. It is designed to:

- scrape Motley Fool earnings call transcript archive pages at scale
- persist raw HTML and resumable scrape state
- clean transcript text into one file per call
- build a metadata index for downstream use
- score transcripts with FinBERT, Loughran-McDonald lexicon ratios, and custom finance themes
- generate company and regional summary outputs for Group 4

## Layout

```text
src/
  run_pipeline.py
  scrape.py
  clean.py
  build_csv.py
  score.py
  report.py
  common.py
  mappings.py
  nlp_utils.py
  garda1/
    cli.py
    __init__.py
data/
  raw_html/
  speakers/
  transcripts/
  lexicons/
  companies.csv
outputs/
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp garda1.example.toml garda1.toml
garda1 run-all --config garda1.toml --seed-archive --write-speakers
```

## CLI

```bash
garda1 scrape --config garda1.toml --seed-archive
garda1 clean --config garda1.toml --write-speakers
garda1 build-csv --us-only
garda1 score --config garda1.toml
garda1 report
```

## Config

The application reads optional settings from `garda1.toml`.

- `scrape.archive_url`: archive landing page to crawl
- `scrape.delay_seconds`: default delay between requests
- `scrape.max_retries`: retry count for `429`/`5xx` responses
- `clean.write_speakers`: whether speaker JSON sidecars are written by default
- `score.finbert_model_name`: transformer model name for sentiment scoring

## Notes

- The scraper respects `robots.txt`, rate limits requests, retries on transient failures, rotates user agents, and persists state in SQLite.
- `src/score.py` caches FinBERT and the Loughran-McDonald lexicon locally under `data/`.
- `data/companies.csv` overrides built-in region and sector mappings when provided.
- `data/companies.csv` can also include a `country` column; use `US` for companies you want kept when building a US-only metadata file.
- Re-running each stage is safe; completed work is skipped unless `--force` is used.
- You can still run stages individually if you want finer control over long scrape jobs.
- `make test` runs the offline validation suite.
- Cleaned transcripts are organized as `data/transcripts/<company>/<year>/<quarter>/<ticker>_<date>_<quarter>.txt`.
