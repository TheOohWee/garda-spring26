from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, quote_plus, urlparse

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

DEFAULT_TICKERS = [
    "MSFT",
    "AAPL",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "TSLA",
    "JPM",
    "UNH",
    "XOM",
]


@dataclass
class ScrapeResult:
    ticker: str
    fiscal_year: int
    query_used: str
    url: str
    status: str
    text_path: str
    char_count: int
    error: str
    fetched_at_utc: str


def parse_tickers(raw: str) -> list[str]:
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            deduped.append(ticker)
    return deduped


def parse_tickers_file(path: str) -> list[str]:
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8")
    normalized = raw.replace("\n", ",").replace("\r", ",")
    return parse_tickers(normalized)


def build_queries(ticker: str, fiscal_year: int) -> list[str]:
    short_fy = str(fiscal_year)[-2:]
    return [
        f"{ticker.lower()} fy{short_fy} earnings call transcript motley fool",
        f"{ticker.lower()} q4 {fiscal_year} earnings call transcript motley fool",
    ]


def normalize_search_result_url(url: str) -> str:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    if "duckduckgo.com/l/?" in url:
        uddg = qs.get("uddg", [])
        if uddg:
            return uddg[0]

    if "google." in parsed.netloc and parsed.path == "/url":
        target = qs.get("q", [])
        if target:
            return target[0]

    return url


def is_fool_host(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host == "fool.com" or host.endswith(".fool.com")


def is_motley_fool_transcript_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    return (
        is_fool_host(url)
        and "/call-transcripts/" in path
        and "/search/" not in path
    )


def is_q4_target_year_url(url: str, fiscal_year: int) -> bool:
    path = urlparse(url).path.lower()
    return f"q4-{fiscal_year}" in path


def pick_best_fool_link(urls: Iterable[str], fiscal_year: int) -> str | None:
    for url in urls:
        if is_motley_fool_transcript_url(url) and is_q4_target_year_url(url, fiscal_year):
            return url
    return None


def create_driver(headless: bool, wait_seconds: int) -> tuple[webdriver.Chrome, WebDriverWait]:
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--window-size=1600,1200")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--log-level=3")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    wait = WebDriverWait(driver, wait_seconds)
    return driver, wait


def collect_duckduckgo_links(driver: webdriver.Chrome, wait: WebDriverWait, query: str) -> list[str]:
    search_url = f"https://duckduckgo.com/?q={quote_plus(query)}"
    driver.get(search_url)
    try:
        wait.until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "a[data-testid='result-title-a'], h2 a, a[href]")
            )
        )
    except TimeoutException:
        return []

    anchors = driver.find_elements(By.CSS_SELECTOR, "a[data-testid='result-title-a'], h2 a, a[href]")
    urls: list[str] = []
    for anchor in anchors:
        href = anchor.get_attribute("href") or ""
        href = normalize_search_result_url(href)
        if href.startswith("http"):
            urls.append(href)
    return urls


def collect_google_links(
    driver: webdriver.Chrome,
    wait: WebDriverWait,
    query: str,
    pause_for_robot_check: bool,
) -> list[str]:
    search_url = f"https://www.google.com/search?hl=en&num=20&q={quote_plus(query)}"
    driver.get(search_url)
    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    except TimeoutException:
        return []
    maybe_pause_for_google_robot_check(driver, query, enable_pause=pause_for_robot_check)

    selectors = [
        "div#search a[href]",
        "a[href]",
    ]
    urls: list[str] = []
    for selector in selectors:
        anchors = driver.find_elements(By.CSS_SELECTOR, selector)
        for anchor in anchors:
            href = anchor.get_attribute("href") or ""
            href = normalize_search_result_url(href)
            if href.startswith("http"):
                urls.append(href)
        if urls:
            break
    return urls


def google_robot_check_detected(driver: webdriver.Chrome) -> bool:
    current_url = (driver.current_url or "").lower()
    if "/sorry/" in current_url:
        return True

    try:
        body_text = (driver.find_element(By.TAG_NAME, "body").text or "").lower()
    except Exception:  # noqa: BLE001
        body_text = ""

    signals = [
        "unusual traffic",
        "not a robot",
        "i'm not a robot",
        "captcha",
        "verify you are human",
    ]
    return any(signal in body_text for signal in signals)


def maybe_pause_for_google_robot_check(driver: webdriver.Chrome, query: str, enable_pause: bool) -> None:
    if not enable_pause:
        return
    if not google_robot_check_detected(driver):
        return

    print(
        f"Google robot-check detected for query '{query}'. "
        "Solve it in the open browser; script will auto-continue once cleared."
    )
    max_wait_seconds = 180
    waited = 0
    while waited < max_wait_seconds and google_robot_check_detected(driver):
        time.sleep(2)
        waited += 2


def collect_bing_links(driver: webdriver.Chrome, wait: WebDriverWait, query: str) -> list[str]:
    search_url = f"https://www.bing.com/search?q={quote_plus(query)}"
    driver.get(search_url)
    try:
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li.b_algo h2 a, h2 a, a[href]")))
    except TimeoutException:
        return []

    anchors = driver.find_elements(By.CSS_SELECTOR, "li.b_algo h2 a, h2 a, a[href]")
    urls: list[str] = []
    for anchor in anchors:
        href = anchor.get_attribute("href") or ""
        if href.startswith("http"):
            urls.append(href)
    return urls


def find_first_motley_fool_link(
    driver: webdriver.Chrome,
    wait: WebDriverWait,
    query: str,
    fiscal_year: int,
    manual_pause_seconds: float = 0.0,
    pause_for_robot_check: bool = True,
) -> str | None:
    google_best = pick_best_fool_link(
        collect_google_links(driver, wait, query, pause_for_robot_check=pause_for_robot_check),
        fiscal_year=fiscal_year,
    )
    if google_best:
        return google_best

    if manual_pause_seconds > 0:
        search_url = f"https://www.google.com/search?hl=en&num=20&q={quote_plus(query)}"
        driver.get(search_url)
        maybe_pause_for_google_robot_check(driver, query, enable_pause=pause_for_robot_check)
        time.sleep(manual_pause_seconds)
        google_best_after_pause = pick_best_fool_link(
            collect_google_links(driver, wait, query, pause_for_robot_check=pause_for_robot_check),
            fiscal_year=fiscal_year,
        )
        if google_best_after_pause:
            return google_best_after_pause

    ddg_best = pick_best_fool_link(
        collect_duckduckgo_links(driver, wait, query),
        fiscal_year=fiscal_year,
    )
    if ddg_best:
        return ddg_best

    bing_best = pick_best_fool_link(
        collect_bing_links(driver, wait, query),
        fiscal_year=fiscal_year,
    )
    if bing_best:
        return bing_best

    return None


def extract_text_with_selenium(driver: webdriver.Chrome) -> tuple[str, str]:
    title = ""
    try:
        title = (driver.find_element(By.TAG_NAME, "h1").text or "").strip()
    except WebDriverException:
        title = (driver.title or "").strip()

    selectors = [
        "div.article-body p",
        "[data-test='article-content'] p",
        "article p",
        "div.article-content p",
        "main p",
    ]

    best_text = ""
    for selector in selectors:
        nodes = driver.find_elements(By.CSS_SELECTOR, selector)
        paragraphs = [n.text.strip() for n in nodes if n.text and n.text.strip()]
        joined = "\n\n".join(paragraphs)
        if len(joined) > len(best_text):
            best_text = joined

    return title, best_text


def extract_text_with_bs4(page_source: str) -> str:
    soup = BeautifulSoup(page_source, "lxml")
    candidates = soup.select("div.article-body, [data-test='article-content'], article, main")

    best_text = ""
    for candidate in candidates:
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in candidate.select("p")
            if p.get_text(" ", strip=True)
        ]
        joined = "\n\n".join(paragraphs)
        if len(joined) > len(best_text):
            best_text = joined

    return best_text


def write_transcript_file(
    output_root: Path,
    ticker: str,
    fiscal_year: int,
    url: str,
    query_used: str,
    title: str,
    body_text: str,
) -> Path:
    ticker_dir = output_root / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    path = ticker_dir / f"FY{fiscal_year}.txt"
    fetched_at = datetime.now(UTC).isoformat()
    content = (
        f"Ticker: {ticker}\n"
        f"Fiscal Year: FY{fiscal_year}\n"
        f"Query Used: {query_used}\n"
        f"Source URL: {url}\n"
        f"Title: {title}\n"
        f"Fetched At (UTC): {fetched_at}\n\n"
        f"{body_text}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def write_debug_snapshot(
    output_root: Path,
    ticker: str,
    fiscal_year: int,
    query_used: str,
    url: str,
    driver: webdriver.Chrome,
) -> tuple[str, str]:
    debug_dir = output_root / "_debug" / ticker
    debug_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"FY{fiscal_year}_{stamp}"
    html_path = debug_dir / f"{base_name}.html"
    png_path = debug_dir / f"{base_name}.png"

    html_content = (
        f"<!-- query: {query_used} -->\n"
        f"<!-- url: {url} -->\n"
        f"{driver.page_source}"
    )
    html_path.write_text(html_content, encoding="utf-8")
    try:
        driver.save_screenshot(str(png_path))
    except Exception:  # noqa: BLE001
        pass
    return str(html_path), str(png_path)


def write_manifest(manifest_path: Path, rows: Iterable[ScrapeResult]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ticker",
                "fiscal_year",
                "query_used",
                "url",
                "status",
                "text_path",
                "char_count",
                "error",
                "fetched_at_utc",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def scrape_single_transcript(
    driver: webdriver.Chrome,
    wait: WebDriverWait,
    output_root: Path,
    ticker: str,
    fiscal_year: int,
    retries: int,
    sleep_seconds: float,
    manual_pause_seconds: float,
    pause_for_robot_check: bool,
) -> ScrapeResult:
    fetched_at = datetime.now(UTC).isoformat()
    last_no_content: ScrapeResult | None = None
    last_error: ScrapeResult | None = None

    for query in build_queries(ticker, fiscal_year):
        for attempt in range(1, retries + 1):
            try:
                url = find_first_motley_fool_link(
                    driver,
                    wait,
                    query,
                    fiscal_year=fiscal_year,
                    pause_for_robot_check=pause_for_robot_check,
                )
                if not url and manual_pause_seconds > 0:
                    url = find_first_motley_fool_link(
                        driver,
                        wait,
                        query,
                        fiscal_year=fiscal_year,
                        manual_pause_seconds=manual_pause_seconds,
                        pause_for_robot_check=pause_for_robot_check,
                    )
                if not url:
                    break

                driver.get(url)
                time.sleep(sleep_seconds)

                title, text = extract_text_with_selenium(driver)
                render_deadline = time.time() + 12
                while len(text) < 300 and time.time() < render_deadline:
                    time.sleep(1)
                    title, text = extract_text_with_selenium(driver)
                if len(text) < 500:
                    text = extract_text_with_bs4(driver.page_source)

                if len(text) < 300:
                    debug_html, debug_png = write_debug_snapshot(
                        output_root=output_root,
                        ticker=ticker,
                        fiscal_year=fiscal_year,
                        query_used=query,
                        url=url,
                        driver=driver,
                    )
                    last_no_content = ScrapeResult(
                        ticker=ticker,
                        fiscal_year=fiscal_year,
                        query_used=query,
                        url=url,
                        status="no_content",
                        text_path="",
                        char_count=0,
                        error=(
                            "Motley Fool page found but transcript extraction was too short. "
                            f"Debug HTML: {debug_html} | Debug Screenshot: {debug_png}"
                        ),
                        fetched_at_utc=fetched_at,
                    )
                    continue

                text_path = write_transcript_file(
                    output_root=output_root,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    url=url,
                    query_used=query,
                    title=title,
                    body_text=text,
                )
                return ScrapeResult(
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    query_used=query,
                    url=url,
                    status="ok",
                    text_path=str(text_path),
                    char_count=len(text),
                    error="",
                    fetched_at_utc=fetched_at,
                )
            except Exception as exc:  # noqa: BLE001
                if attempt == retries:
                    last_error = ScrapeResult(
                        ticker=ticker,
                        fiscal_year=fiscal_year,
                        query_used=query,
                        url="",
                        status="error",
                        text_path="",
                        char_count=0,
                        error=str(exc),
                        fetched_at_utc=fetched_at,
                    )
                    break
                time.sleep(sleep_seconds)

    if last_no_content is not None:
        return last_no_content
    if last_error is not None:
        return last_error

    return ScrapeResult(
        ticker=ticker,
        fiscal_year=fiscal_year,
        query_used="; ".join(build_queries(ticker, fiscal_year)),
        url="",
        status="no_fool_link",
        text_path="",
        char_count=0,
        error=f"No Motley Fool Q4 FY{fiscal_year} transcript URL found in search results.",
        fetched_at_utc=fetched_at,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Motley Fool earnings call transcripts with Selenium via search queries."
    )
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated stock tickers (default includes 10 tickers).",
    )
    parser.add_argument(
        "--tickers-file",
        default="",
        help="Optional text file of tickers (comma or newline separated). Overrides --tickers.",
    )
    parser.add_argument("--fy-start", type=int, default=2020, help="Start fiscal year (inclusive).")
    parser.add_argument("--fy-end", type=int, default=2025, help="End fiscal year (inclusive).")
    parser.add_argument(
        "--output-dir",
        default="data/raw/earnings/motley_fool_transcripts",
        help="Directory where transcript text files and manifest CSV are written.",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.csv",
        help="Manifest CSV filename inside output directory.",
    )
    parser.add_argument("--wait-seconds", type=int, default=15, help="Selenium explicit wait timeout.")
    parser.add_argument("--sleep-seconds", type=float, default=1.2, help="Pause after page load.")
    parser.add_argument(
        "--manual-pause-seconds",
        type=float,
        default=0.0,
        help="Optional extra pause on Google results before parsing links (useful if prompts appear).",
    )
    parser.add_argument(
        "--pause-for-robot-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pause and wait for Enter when Google robot-check/CAPTCHA is detected.",
    )
    parser.add_argument("--retries", type=int, default=2, help="Retry attempts per query.")
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run Chrome with UI visible. Default is headless.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fy_end < args.fy_start:
        print("Error: --fy-end must be >= --fy-start", file=sys.stderr)
        return 2

    tickers = parse_tickers_file(args.tickers_file) if args.tickers_file else parse_tickers(args.tickers)
    if not tickers:
        print("Error: no valid tickers were provided.", file=sys.stderr)
        return 2

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / args.manifest_name
    fiscal_years = list(range(args.fy_start, args.fy_end + 1))

    print(f"Tickers: {tickers}")
    print(f"Fiscal years: {fiscal_years}")
    print(f"Output directory: {output_root.resolve()}")

    try:
        driver, wait = create_driver(headless=not args.headful, wait_seconds=args.wait_seconds)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to start Chrome WebDriver: {exc}", file=sys.stderr)
        return 1

    rows: list[ScrapeResult] = []
    total = len(tickers) * len(fiscal_years)
    counter = 0
    try:
        for ticker in tickers:
            for fy in fiscal_years:
                counter += 1
                print(f"[{counter}/{total}] Scraping {ticker} FY{fy}...")
                result = scrape_single_transcript(
                    driver=driver,
                    wait=wait,
                    output_root=output_root,
                    ticker=ticker,
                    fiscal_year=fy,
                    retries=args.retries,
                    sleep_seconds=args.sleep_seconds,
                    manual_pause_seconds=args.manual_pause_seconds,
                    pause_for_robot_check=args.pause_for_robot_check,
                )
                rows.append(result)
                print(f"  -> {result.status} ({result.char_count} chars)")
    finally:
        driver.quit()

    write_manifest(manifest_path, rows)

    ok_count = sum(1 for r in rows if r.status == "ok")
    print(f"\nDone. Successful transcripts: {ok_count}/{len(rows)}")
    print(f"Manifest written to: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
