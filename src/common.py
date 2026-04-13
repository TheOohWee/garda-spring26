from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import tomllib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
RAW_HTML_DIR = DATA_DIR / "raw_html"
SPEAKERS_DIR = DATA_DIR / "speakers"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
LEXICONS_DIR = DATA_DIR / "lexicons"
SCRAPE_DB_PATH = DATA_DIR / "scrape_state.db"
CLEAN_DB_PATH = DATA_DIR / "clean_state.db"
SCORE_DB_PATH = DATA_DIR / "score_cache.db"
COMPANIES_CSV_PATH = DATA_DIR / "companies.csv"
METADATA_CSV_PATH = DATA_DIR / "transcripts_metadata.csv"
SCORED_CSV_PATH = OUTPUTS_DIR / "scored_transcripts.csv"
DEFAULT_CONFIG_PATH = ROOT_DIR / "garda1.toml"

DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

THEME_KEYWORDS = {
    "demand": [
        "demand",
        "backlog",
        "orders",
        "bookings",
        "pipeline",
        "consumption",
        "traffic",
        "volume",
    ],
    "hiring": [
        "hiring",
        "headcount",
        "recruiting",
        "talent",
        "workforce",
        "attrition",
        "layoff",
        "staffing",
    ],
    "pricing": [
        "pricing",
        "price increase",
        "price realization",
        "discounting",
        "promotional",
        "margin pressure",
    ],
    "capex": [
        "capex",
        "capital expenditure",
        "data center",
        "plant",
        "facility",
        "infrastructure",
        "expansion",
    ],
    "ai": [
        "artificial intelligence",
        "generative ai",
        "machine learning",
        "copilot",
        "llm",
        "automation",
        "foundation model",
    ],
    "efficiency": [
        "efficiency",
        "productivity",
        "optimization",
        "cost savings",
        "streamline",
        "lean",
        "operating leverage",
    ],
}

AI_LABOR_KEYWORDS = {
    "ai_tech": [
        "artificial intelligence",
        "generative ai",
        "automation",
        "machine learning",
        "copilot",
        "agentic",
    ],
    "labor_down": [
        "reduce headcount",
        "fewer employees",
        "labor savings",
        "workforce reduction",
        "automation savings",
        "restructuring",
    ],
    "labor_up": [
        "increase headcount",
        "hire more",
        "expand workforce",
        "recruiting",
        "talent investment",
    ],
    "productivity": [
        "productivity",
        "efficiency",
        "throughput",
        "cycle time",
        "faster development",
        "operating leverage",
    ],
}

GUIDANCE_RAISED_PATTERNS = [
    r"\brais(?:e|ed|ing)\s+guidance\b",
    r"\brevis(?:e|ed|ing)\s+upward\b",
    r"\babove expectations\b",
    r"\bbeat(?:ing)? estimates\b",
    r"\boutperform(?:ed|ing)?\b",
]

GUIDANCE_LOWERED_PATTERNS = [
    r"\blower(?:ed|ing)?\s+guidance\b",
    r"\brevis(?:e|ed|ing)\s+downward\b",
    r"\bbelow expectations\b",
    r"\bmiss(?:ed|ing)? estimates\b",
    r"\bsoften(?:ed|ing)? outlook\b",
]

CONFIDENCE_TERMS = [
    "committed",
    "confident",
    "strong execution",
    "on track",
    "disciplined",
    "solid demand",
    "well positioned",
]

HEDGING_TERMS = [
    "may",
    "might",
    "uncertain",
    "challenging environment",
    "headwinds",
    "volatility",
    "cautious",
]

RISK_PATTERNS = [
    r"\brisk\b",
    r"\bheadwinds?\b",
    r"\bmacro(?:economic)? uncertainty\b",
    r"\bsupply chain\b",
    r"\bregulatory\b",
    r"\blitigation\b",
    r"\bcyber(?:security)?\b",
]


@dataclass(slots=True)
class TranscriptRecord:
    url: str
    raw_html_path: str
    metadata_path: str
    transcript_path: str
    slug: str
    company: str | None = None
    ticker: str | None = None
    call_date: str | None = None
    quarter: str | None = None
    fiscal_year: str | None = None


@dataclass(slots=True)
class AppConfig:
    archive_url: str = "https://www.fool.com/earnings-call-transcripts/"
    scrape_delay_seconds: float = 3.0
    scrape_max_retries: int = 5
    write_speakers: bool = True
    finbert_model_name: str = "ProsusAI/finbert"


def load_app_config(path: Path | None = None) -> AppConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return AppConfig()

    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    scrape = payload.get("scrape", {})
    clean = payload.get("clean", {})
    score = payload.get("score", {})
    return AppConfig(
        archive_url=str(scrape.get("archive_url", AppConfig.archive_url)),
        scrape_delay_seconds=float(scrape.get("delay_seconds", AppConfig.scrape_delay_seconds)),
        scrape_max_retries=int(scrape.get("max_retries", AppConfig.scrape_max_retries)),
        write_speakers=bool(clean.get("write_speakers", AppConfig.write_speakers)),
        finbert_model_name=str(score.get("finbert_model_name", AppConfig.finbert_model_name)),
    )


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_directories() -> None:
    for path in [DATA_DIR, OUTPUTS_DIR, RAW_HTML_DIR, SPEAKERS_DIR, TRANSCRIPTS_DIR, LEXICONS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return cleaned or "unknown"


def transcript_filename(ticker: str | None, call_date: str | None, quarter: str | None) -> str:
    safe_ticker = slugify((ticker or "unknown").upper()).upper()
    safe_date = call_date or "unknown-date"
    safe_quarter = slugify(quarter or "unknown-quarter").upper()
    return f"{safe_ticker}_{safe_date}_{safe_quarter}.txt"


def transcript_storage_path(
    company: str | None,
    fiscal_year: str | None,
    call_date: str | None,
    quarter: str | None,
    ticker: str | None,
) -> Path:
    company_dir = slugify(company or ticker or "unknown-company")
    year_dir = str(fiscal_year or (call_date or "unknown-date")[:4] or "unknown-year")
    quarter_dir = slugify(quarter or "unknown-quarter").upper()
    filename = transcript_filename(ticker, call_date, quarter)
    return TRANSCRIPTS_DIR / company_dir / year_dir / quarter_dir / filename


def speakers_storage_path(
    company: str | None,
    fiscal_year: str | None,
    call_date: str | None,
    quarter: str | None,
    ticker: str | None,
) -> Path:
    company_dir = slugify(company or ticker or "unknown-company")
    year_dir = str(fiscal_year or (call_date or "unknown-date")[:4] or "unknown-year")
    quarter_dir = slugify(quarter or "unknown-quarter").upper()
    filename = transcript_filename(ticker, call_date, quarter).replace(".txt", ".json")
    return SPEAKERS_DIR / company_dir / year_dir / quarter_dir / filename


def transcript_uuid(ticker: str, call_date: str, quarter: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{ticker}|{call_date}|{quarter}"))


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def connect_sqlite(path: Path) -> sqlite3.Connection:
    ensure_directories()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_scrape_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS listing_pages (
            url TEXT PRIMARY KEY,
            scraped_at TEXT,
            discovered_urls INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS transcript_urls (
            url TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'pending',
            discovered_at TEXT,
            scraped_at TEXT,
            retries INTEGER DEFAULT 0,
            error TEXT,
            raw_html_path TEXT,
            metadata_path TEXT,
            transcript_path TEXT
        );
        """
    )
    conn.commit()


def init_clean_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS cleaned_transcripts (
            slug TEXT PRIMARY KEY,
            transcript_path TEXT NOT NULL,
            speakers_path TEXT,
            cleaned_at TEXT,
            status TEXT NOT NULL DEFAULT 'cleaned',
            language TEXT,
            notes TEXT
        );
        """
    )
    conn.commit()


def init_score_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS transcript_scores (
            transcript_id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            scored_at TEXT NOT NULL
        );
        """
    )
    conn.commit()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def clean_text_basic(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\[\*\]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def count_keyword_hits(text: str, keywords: Iterable[str]) -> int:
    lowered = text.lower()
    total = 0
    for keyword in keywords:
        total += len(re.findall(rf"(?<!\w){re.escape(keyword.lower())}(?!\w)", lowered))
    return total


def parse_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def load_companies_override() -> pd.DataFrame:
    if COMPANIES_CSV_PATH.exists():
        return pd.read_csv(COMPANIES_CSV_PATH).fillna("")
    return pd.DataFrame(columns=["company", "ticker", "region", "sector"])
