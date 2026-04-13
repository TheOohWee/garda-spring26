from __future__ import annotations

import json
import logging
import random
import re
import time
import urllib.parse
import urllib.robotparser
from datetime import datetime
from itertools import cycle
from pathlib import Path

import certifi
import requests
from bs4 import BeautifulSoup

from common import (
    DEFAULT_USER_AGENTS,
    RAW_HTML_DIR,
    SCRAPE_DB_PATH,
    TRANSCRIPTS_DIR,
    TranscriptRecord,
    clean_text_basic,
    configure_logging,
    connect_sqlite,
    ensure_directories,
    init_scrape_db,
    load_app_config,
    parse_args,
    sha1_text,
    transcript_filename,
    utc_now_iso,
    write_json,
)

LOGGER = logging.getLogger(__name__)
ARCHIVE_URL = "https://www.fool.com/earnings-call-transcripts/"
TRANSCRIPT_URL_PATTERNS = [
    re.compile(r"^https://www\.fool\.com/earnings/call-transcripts/\d{4}/\d{2}/\d{2}/"),
    re.compile(r"^https://www\.fool\.com/earnings-call-transcripts/\d{4}/\d{2}/\d{2}/"),
]
MONTH_NAME_DATE_PATTERNS = [
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%b %d %Y",
]


class RateLimitedSession:
    def __init__(self, delay_seconds: float, user_agents: list[str], max_retries: int = 5) -> None:
        self.delay_seconds = delay_seconds
        self.user_agents = cycle(user_agents)
        self.max_retries = max_retries
        self.session = requests.Session()
        self._next_allowed_time = 0.0

    def get(self, url: str, timeout: int = 60) -> requests.Response:
        backoff = self.delay_seconds
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            now = time.time()
            if now < self._next_allowed_time:
                time.sleep(self._next_allowed_time - now)

            headers = {"User-Agent": next(self.user_agents)}
            try:
                response = self.session.get(url, timeout=timeout, headers=headers)
                self._next_allowed_time = time.time() + self.delay_seconds + random.uniform(0, 0.3)
                if response.status_code in {429, 500, 502, 503, 504}:
                    LOGGER.warning("Retryable status %s for %s (attempt %s/%s)", response.status_code, url, attempt, self.max_retries)
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_exc = exc
                LOGGER.warning("Request failed for %s (attempt %s/%s): %s", url, attempt, self.max_retries, exc)
                time.sleep(backoff)
                backoff *= 2

        raise RuntimeError(f"Failed to fetch {url}") from last_exc


def parse_robots(base_url: str) -> urllib.robotparser.RobotFileParser:
    parsed = urllib.parse.urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    response = requests.get(
        robots_url,
        timeout=60,
        headers={"User-Agent": DEFAULT_USER_AGENTS[0]},
        verify=certifi.where(),
    )
    response.raise_for_status()
    parser.parse(response.text.splitlines())
    return parser


def can_fetch(robots: urllib.robotparser.RobotFileParser, user_agent: str, url: str) -> bool:
    return robots.can_fetch(user_agent, url)


def listing_page_urls(soup: BeautifulSoup, current_url: str) -> list[str]:
    urls = []
    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "").strip()
        if not href:
            continue
        absolute = urllib.parse.urljoin(current_url, href)
        rel = " ".join(anchor.get("rel", []))
        is_archive_page = absolute.startswith(ARCHIVE_URL) and absolute != current_url
        looks_paginated = any(
            token in absolute.lower()
            for token in ["?page=", "&page=", "/page/"]
        )
        if is_archive_page and ("next" in rel.lower() or looks_paginated):
            urls.append(absolute)
    return sorted(set(urls))


def transcript_urls_from_listing(soup: BeautifulSoup, current_url: str) -> list[str]:
    transcript_urls: list[str] = []
    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "").strip()
        if not href:
            continue
        absolute = urllib.parse.urljoin(current_url, href)
        if any(pattern.search(absolute) for pattern in TRANSCRIPT_URL_PATTERNS):
            transcript_urls.append(absolute)
    return sorted(set(transcript_urls))


def extract_json_ld(soup: BeautifulSoup) -> dict:
    for node in soup.select('script[type="application/ld+json"]'):
        try:
            payload = json.loads(node.get_text(strip=True))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and item.get("@type") in {"NewsArticle", "Article"}:
                    return item
        if isinstance(payload, dict) and payload.get("@type") in {"NewsArticle", "Article"}:
            return payload
    return {}


def extract_text_blocks(soup: BeautifulSoup) -> str:
    for node in soup(["script", "style", "noscript", "svg"]):
        node.decompose()

    selectors = [
        'article',
        '[data-test="article-content"]',
        '.article-content',
        '.tailwind-article-body',
        '.article-body',
        'main',
        'body',
    ]
    candidates: list[str] = []
    for selector in selectors:
        for node in soup.select(selector):
            text = clean_text_basic(_trim_transcript_body(node.get_text("\n", strip=True)))
            if len(text.split()) >= 50:
                candidates.append(text)

    if candidates:
        return max(candidates, key=lambda value: len(value.split()))
    return clean_text_basic(_trim_transcript_body(soup.get_text("\n", strip=True)))


def _trim_transcript_body(text: str) -> str:
    cutoff_patterns = [
        r"Find out why .*? investors",
        r"Motley Fool Returns",
        r"This article is a transcript of",
        r"The Motley Fool has positions in and recommends",
        r"You're reading a free article",
    ]
    trimmed = text
    for pattern in cutoff_patterns:
        match = re.search(pattern, trimmed, flags=re.IGNORECASE | re.DOTALL)
        if match:
            trimmed = trimmed[: match.start()]
    return trimmed


def normalize_date(date_text: str | None) -> str | None:
    if not date_text:
        return None

    match = re.search(r"(\d{4}-\d{2}-\d{2})", date_text)
    if match:
        return match.group(1)

    compact = re.sub(r"\s+", " ", date_text.replace(".", "")).strip()
    for pattern in MONTH_NAME_DATE_PATTERNS:
        try:
            return datetime.strptime(compact, pattern).strftime("%Y-%m-%d")
        except ValueError:
            continue

    month_match = re.search(
        r"\b([A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4})\b",
        compact,
    )
    if month_match:
        return normalize_date(month_match.group(1))
    return None


def infer_company_from_title(title: str, ticker: str | None) -> str | None:
    if not title:
        return ticker
    company = re.sub(r"\bQ[1-4]\b.*", "", title, flags=re.IGNORECASE)
    company = re.sub(r"\s*\([A-Z.\-]{1,8}\)", "", company)
    company = re.sub(r"\bearnings call transcript\b", "", company, flags=re.IGNORECASE)
    company = re.sub(r"\bfiscal\b.*", "", company, flags=re.IGNORECASE)
    company = company.strip(" :-|")
    return company or ticker


def parse_transcript_metadata(html: str, url: str) -> tuple[dict, str]:
    soup = BeautifulSoup(html, "lxml")
    json_ld = extract_json_ld(soup)
    title = (json_ld.get("headline") or soup.title.get_text(strip=True) if soup.title else "").strip()
    text = extract_text_blocks(soup)

    company = None
    ticker = None
    quarter = None
    fiscal_year = None
    call_date = None

    header_text = " ".join(filter(None, [title, soup.get_text(" ", strip=True)[:4000]]))

    ticker_match = re.search(r"\(([A-Z.\-]{1,8})\)", header_text)
    if ticker_match:
        ticker = ticker_match.group(1).replace(".", "-")

    quarter_match = re.search(r"\b(Q[1-4])\b", header_text, flags=re.IGNORECASE)
    if quarter_match:
        quarter = quarter_match.group(1).upper()

    year_match = re.search(r"\b(FY\s*)?(20\d{2})\b", header_text, flags=re.IGNORECASE)
    if year_match:
        fiscal_year = year_match.group(2)

    if title:
        company = infer_company_from_title(title, ticker)

    date_candidates = [
        json_ld.get("datePublished"),
        json_ld.get("dateModified"),
        soup.find("time").get("datetime") if soup.find("time") else None,
        title,
        header_text,
    ]
    for candidate in date_candidates:
        call_date = normalize_date(str(candidate))
        if call_date:
            break

    if not quarter:
        quarter_long_match = re.search(r"\b(first|second|third|fourth)\s+quarter\b", header_text, flags=re.IGNORECASE)
        if quarter_long_match:
            quarter_map = {"first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4"}
            quarter = quarter_map[quarter_long_match.group(1).lower()]

    if not company and ticker:
        company = ticker

    metadata = {
        "company": company,
        "ticker": ticker,
        "call_date": call_date,
        "quarter": quarter,
        "fiscal_year": fiscal_year,
        "source_url": url,
        "title": title,
        "scraped_at": utc_now_iso(),
        "text_preview": text[:500],
    }
    return metadata, text


def persist_transcript(url: str, html: str, metadata: dict, text: str) -> TranscriptRecord:
    slug = sha1_text(url)
    raw_html_path = RAW_HTML_DIR / f"{slug}.html"
    metadata_path = RAW_HTML_DIR / f"{slug}.json"
    transcript_path = TRANSCRIPTS_DIR / transcript_filename(metadata.get("ticker"), metadata.get("call_date"), metadata.get("quarter"))

    raw_html_path.write_text(html, encoding="utf-8")
    if not transcript_path.exists():
        transcript_path.write_text(text + "\n", encoding="utf-8")
    write_json(
        metadata_path,
        {
            **metadata,
            "slug": slug,
            "raw_html_path": str(raw_html_path),
            "transcript_path": str(transcript_path),
        },
    )
    return TranscriptRecord(
        url=url,
        raw_html_path=str(raw_html_path),
        metadata_path=str(metadata_path),
        transcript_path=str(transcript_path),
        slug=slug,
        company=metadata.get("company"),
        ticker=metadata.get("ticker"),
        call_date=metadata.get("call_date"),
        quarter=metadata.get("quarter"),
        fiscal_year=metadata.get("fiscal_year"),
    )


def seed_archive_urls(conn, session: RateLimitedSession, robots) -> None:
    pending_listing_pages = [ARCHIVE_URL]
    seen = set()
    while pending_listing_pages:
        listing_url = pending_listing_pages.pop(0)
        if listing_url in seen:
            continue
        seen.add(listing_url)
        if not can_fetch(robots, DEFAULT_USER_AGENTS[0], listing_url):
            LOGGER.warning("Skipping disallowed listing page %s", listing_url)
            continue

        response = session.get(listing_url)
        soup = BeautifulSoup(response.text, "lxml")
        transcript_urls = transcript_urls_from_listing(soup, listing_url)
        for transcript_url in transcript_urls:
            conn.execute(
                """
                INSERT INTO transcript_urls (url, discovered_at)
                VALUES (?, ?)
                ON CONFLICT(url) DO NOTHING
                """,
                (transcript_url, utc_now_iso()),
            )

        conn.execute(
            """
            INSERT INTO listing_pages (url, scraped_at, discovered_urls)
            VALUES (?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET scraped_at=excluded.scraped_at, discovered_urls=excluded.discovered_urls
            """,
            (listing_url, utc_now_iso(), len(transcript_urls)),
        )
        conn.commit()

        for next_url in listing_page_urls(soup, listing_url):
            if next_url not in seen:
                pending_listing_pages.append(next_url)

        LOGGER.info("Discovered %s transcript URLs so far.", conn.execute("SELECT COUNT(*) FROM transcript_urls").fetchone()[0])


def scrape_pending(conn, session: RateLimitedSession, robots) -> None:
    total = conn.execute("SELECT COUNT(*) FROM transcript_urls").fetchone()[0]
    rows = conn.execute(
        """
        SELECT url, retries FROM transcript_urls
        WHERE status NOT IN ('done', 'skipped')
        ORDER BY discovered_at ASC, url ASC
        """
    ).fetchall()

    for index, row in enumerate(rows, start=1):
        url = row["url"]
        if not can_fetch(robots, DEFAULT_USER_AGENTS[0], url):
            LOGGER.warning("Skipping disallowed transcript URL %s", url)
            conn.execute("UPDATE transcript_urls SET status='skipped', error=? WHERE url=?", ("disallowed by robots.txt", url))
            conn.commit()
            continue

        try:
            response = session.get(url)
            metadata, text = parse_transcript_metadata(response.text, url)
            record = persist_transcript(url, response.text, metadata, text)
            conn.execute(
                """
                UPDATE transcript_urls
                SET status='done', scraped_at=?, error=NULL, raw_html_path=?, metadata_path=?, transcript_path=?
                WHERE url=?
                """,
                (utc_now_iso(), record.raw_html_path, record.metadata_path, record.transcript_path, url),
            )
            conn.commit()
            completed = conn.execute("SELECT COUNT(*) FROM transcript_urls WHERE status='done'").fetchone()[0]
            LOGGER.info("Scraped %s / ~%s transcripts...", completed, total)
        except Exception as exc:  # noqa: BLE001
            retries = int(row["retries"]) + 1
            status = "failed" if retries >= session.max_retries else "pending"
            conn.execute(
                """
                UPDATE transcript_urls
                SET status=?, retries=?, error=?
                WHERE url=?
                """,
                (status, retries, str(exc), url),
            )
            conn.commit()
            LOGGER.exception("Failed to scrape %s", url)


def main(argv: list[str] | None = None) -> None:
    parser = parse_args("Scrape Motley Fool earnings call transcripts.")
    parser.add_argument("--config", type=Path, default=None, help="Path to a TOML config file.")
    parser.add_argument("--seed-archive", action="store_true", help="Crawl archive pages to discover transcript URLs before scraping.")
    parser.add_argument("--delay-seconds", type=float, default=None, help="Seconds to wait between requests.")
    parser.add_argument("--max-retries", type=int, default=None, help="Maximum retries for retryable failures.")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    ensure_directories()
    config = load_app_config(args.config)
    conn = connect_sqlite(SCRAPE_DB_PATH)
    init_scrape_db(conn)

    delay_seconds = args.delay_seconds if args.delay_seconds is not None else config.scrape_delay_seconds
    max_retries = args.max_retries if args.max_retries is not None else config.scrape_max_retries
    session = RateLimitedSession(delay_seconds, DEFAULT_USER_AGENTS, max_retries=max_retries)
    robots = parse_robots(config.archive_url)

    existing_transcript_count = conn.execute("SELECT COUNT(*) FROM transcript_urls").fetchone()[0]
    pending_transcript_count = conn.execute(
        "SELECT COUNT(*) FROM transcript_urls WHERE status NOT IN ('done', 'skipped')"
    ).fetchone()[0]

    if args.seed_archive and existing_transcript_count == 0:
        LOGGER.info("No transcript URL queue found yet, running archive discovery.")
        seed_archive_urls(conn, session, robots)
    elif args.seed_archive and existing_transcript_count > 0:
        LOGGER.info(
            "Transcript URL queue already exists with %s URLs and %s pending; skipping archive rediscovery.",
            existing_transcript_count,
            pending_transcript_count,
        )
    elif existing_transcript_count == 0:
        LOGGER.info("No transcript URL queue found yet, running archive discovery.")
        seed_archive_urls(conn, session, robots)

    scrape_pending(conn, session, robots)


if __name__ == "__main__":
    main()
