# Earnings + AI Labor Signals (Group 3)

This repo implements **Group 3** of the course project: a small **bottom-up** pipeline that scores **earnings call transcripts** for macro-relevant themes and for an **AI / labor / productivity** proxy, then **aggregates scores by region** for handoff to integration (Group 4).

---


### I just found out there is some problem with transcripts here (some .txt look strange), but scores were calculated fine.

## What this is for

- **Earnings themes**: demand, hiring, pricing, capex, AI, efficiency — keyword density plus sentiment (VADER) on sentences that mention each theme.
- **AI labor signal**: separate from the generic “AI” theme — combines AI/tech language, productivity/automation language, positive labor/hiring language, and a **penalty** when “labor down” keywords (layoffs, hiring freeze, etc.) appear, weighted by how negative those passages sound.

The broader project goal (see `task.md`) is to combine regional scores from multiple groups into one composite view; this repo produces **two regional CSVs** plus detailed company-level outputs.

---

## What we did (step by step)

1. **Defined the assignment scope**  
   Per `task.md`: ~10–20 companies, transcripts, theme tracking, simple keyword/sentiment scoring, regional earnings and AI scores, plus a transcript/theme summary file.

2. **Chose a cross-sector sample**  
   Companies span consumer, tech, energy, healthcare, REITs, etc., with **regions** such as `North_America`, `Asia_Pacific`, and `Latin_America` (see `data/companies.csv`).

3. **Collected transcript text**  
   Plain-text files under `data/transcripts/` (paths listed in `companies.csv`).

4. **Centralized scoring logic** (`scoring_utils.py`)  
   - Word counts, keyword hit counts (word-boundary aware for single tokens).  
   - **Theme density** (hits per 1k words).  
   - **VADER** sentiment on keyword-touching sentences (or whole doc fallback).  
   - **Adjusted theme score** = density × a sentiment multiplier.  
   - Separate **AI_LABOR** keyword groups for the AI-labor proxy.

5. **Built `earnings_score.py`**  
   For each company: per-theme scores, then **earnings_composite** = average of the six theme scores. Writes:
   - `output/transcript_theme_summary.csv` — full per-company breakdown (themes + composite + word count).
   - `output/regional_earnings_scores.csv` — mean `earnings_composite` by region and `n_companies`.

6. **Built `ai_score.py`**  
   For each company: **ai_labor_score** from the weighted formula in `ai_labor_company_score()` (AI tech, productivity, labor-up, minus penalized labor-down). Writes:
   - `output/company_ai_scores.csv`
   - `output/regional_ai_scores.csv`

7. **Dependencies**  
   `requirements.txt` lists `vaderSentiment` for sentiment.

---

## Repository layout

| Path | Role |
|------|------|
| `data/companies.csv` | Master list: symbol, name, region, sector, `transcript_path` |
| `data/transcripts/*.txt` | Earnings call text (UTF-8) |
| `scoring_utils.py` | Shared keywords, reading transcripts, scoring helpers |
| `earnings_score.py` | Theme + composite + regional earnings |
| `ai_score.py` | AI-labor proxy + regional AI |
| `output/` | Generated CSVs (gitignored contents except `.gitkeep` if used) |

Supporting docs in the repo: `task.md` (full project group breakdown), `ppt.md` (framing / sectors).

---

## How to run

From the repo root (Python 3):

```bash
pip install -r requirements.txt
python earnings_score.py
python ai_score.py
```

Optional:

```bash
python earnings_score.py --companies data/companies.csv --out-dir output
python ai_score.py --companies data/companies.csv --out-dir output
```

---

## Outputs (deliverables)

| File | Description |
|------|-------------|
| `output/transcript_theme_summary.csv` | Per company: `theme_*`, `earnings_composite`, `word_count` |
| `output/regional_earnings_scores.csv` | Regional mean of `earnings_composite` |
| `output/company_ai_scores.csv` | Per company `ai_labor_score` |
| `output/regional_ai_scores.csv` | Regional mean of `ai_labor_score` |

---

## Next steps (outside this repo)

Group 4 can **standardize scales**, merge with macro/policy scores, rank regions, and produce composite tables and trade ideas (`task.md` Group 4 deliverables).
