from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

try:
    from langdetect import DetectorFactory, LangDetectException, detect
except ModuleNotFoundError:  # pragma: no cover - lightweight fallback for pre-install validation
    class LangDetectException(Exception):
        pass

    class DetectorFactory:
        seed = 0

    def detect(text: str) -> str:
        ascii_ratio = sum(char.isascii() for char in text) / max(len(text), 1)
        return "en" if ascii_ratio > 0.9 else "unknown"

from common import (
    CLEAN_DB_PATH,
    RAW_HTML_DIR,
    SPEAKERS_DIR,
    TRANSCRIPTS_DIR,
    clean_text_basic,
    configure_logging,
    connect_sqlite,
    ensure_directories,
    init_clean_db,
    load_app_config,
    parse_args,
    read_json,
    speakers_storage_path,
    transcript_storage_path,
    write_json,
    utc_now_iso,
    word_count,
)

DetectorFactory.seed = 0
LOGGER = logging.getLogger(__name__)

BOILERPLATE_PATTERNS = [
    r"Find out why .*? investors",
    r"You're reading a free article with opinions that may differ from The Motley Fool's Premium Investing Services",
    r"The Motley Fool has positions in and recommends",
    r"Motley Fool Returns",
    r"This transcript was created by",
    r"This article is a transcript of .*? conference call",
    r"Call participants:",
    r"Prepared Remarks:",
    r"\bfreestar\b",
]


def remove_boilerplate(text: str) -> str:
    cleaned = text
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"\bAdvertisement\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[[0-9]+\]", " ", cleaned)
    cleaned = re.sub(r"\(\s*TMF[a-z0-9_-]*\s*\)", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^\S\r\n]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_clean_text(html_text: str) -> str:
    transcript_html = extract_transcript_html_segment(html_text) or html_text
    soup = BeautifulSoup(transcript_html, "lxml")
    for node in soup(["script", "style", "noscript", "svg"]):
        node.decompose()

    text = soup.get_text("\n", strip=True)
    text = clean_text_basic(text)
    text = remove_boilerplate(text)
    text = normalize_transcript_lines(text)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\u024F]", " ", text)
    text = re.sub(r"\n +", "\n", text)
    cleaned = clean_text_basic(text)
    if len(cleaned.split()) < 100:
        return extract_clean_text_fallback(html_text)
    return cleaned


def extract_transcript_html_segment(html_text: str) -> str | None:
    start_markers = [
        "Full Conference Call Transcript",
        "Prepared Remarks:</h2>",
        "Prepared Remarks:",
    ]
    end_markers = [
        "Read Next",
        "More ",
        "All earnings call transcripts",
    ]

    start_idx = -1
    for marker in start_markers:
        start_idx = html_text.find(marker)
        if start_idx != -1:
            break
    if start_idx == -1:
        return None

    start_h2_idx = html_text.rfind("<h2", 0, start_idx)
    if start_h2_idx == -1:
        start_h2_idx = start_idx

    end_idx = -1
    for marker in end_markers:
        candidate = html_text.find(marker, start_idx)
        if candidate != -1 and (end_idx == -1 or candidate < end_idx):
            end_idx = candidate
    if end_idx == -1:
        return html_text[start_h2_idx:]

    end_h2_idx = html_text.rfind("<h2", start_idx, end_idx)
    if end_h2_idx == -1:
        end_h2_idx = end_idx

    return html_text[start_h2_idx:end_h2_idx]


def extract_clean_text_fallback(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "lxml")
    for node in soup(["script", "style", "noscript", "svg"]):
        node.decompose()

    selectors = [
        "article",
        '[data-test="article-content"]',
        ".article-content",
        ".tailwind-article-body",
        ".article-body",
        "main",
        "body",
    ]
    candidate_texts: list[str] = []
    for selector in selectors:
        for source in soup.select(selector):
            text = source.get_text("\n", strip=True)
            normalized = clean_text_basic(text)
            if len(normalized.split()) >= 50:
                candidate_texts.append(normalized)

    text = max(candidate_texts, key=lambda value: len(value.split())) if candidate_texts else soup.get_text("\n", strip=True)
    text = clean_text_basic(text)
    text = remove_boilerplate(text)
    text = normalize_transcript_lines(text)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\u024F]", " ", text)
    text = re.sub(r"\n +", "\n", text)
    return clean_text_basic(text)


def normalize_transcript_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    kept: list[str] = []
    for line in lines:
        if not line:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if len(line) < 2:
            continue
        if re.fullmatch(r"Page \d+", line, flags=re.IGNORECASE):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def detect_language_or_unknown(text: str) -> str:
    snippet = " ".join(text.split()[:400])
    if not snippet:
        return "unknown"
    try:
        return detect(snippet)
    except LangDetectException:
        return "unknown"


def extract_speakers(text: str) -> list[dict[str, str]]:
    speaker_blocks: list[dict[str, str]] = []
    current = None
    speaker_regex = re.compile(
        r"^(?P<speaker>[A-Z][A-Za-z .']{1,80}?)\s*(?:--|-|:)\s*(?P<role>[A-Za-z,&/ .'-]{2,120})?$"
    )
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = speaker_regex.match(stripped)
        if match and len(stripped.split()) <= 10:
            if current:
                speaker_blocks.append(current)
            current = {
                "speaker": match.group("speaker").strip(),
                "role": (match.group("role") or "").strip(),
                "text": "",
            }
            continue
        if stripped in {"Question-and-Answer Session", "Questions and Answers"}:
            if current:
                speaker_blocks.append(current)
            current = {"speaker": "Operator", "role": "Moderator", "text": stripped}
            continue
        if current is None:
            current = {"speaker": "Unknown", "role": "", "text": stripped}
        else:
            current["text"] = f"{current['text']} {stripped}".strip()
    if current:
        speaker_blocks.append(current)
    return [block for block in speaker_blocks if word_count(block["text"]) > 3]


def main(argv: list[str] | None = None) -> None:
    parser = parse_args("Clean scraped transcript HTML into plain text files.")
    parser.add_argument("--config", type=Path, default=None, help="Path to a TOML config file.")
    parser.add_argument("--write-speakers", action="store_true", help="Write speaker JSON sidecars.")
    parser.add_argument("--force", action="store_true", help="Reprocess already-cleaned files.")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    ensure_directories()
    config = load_app_config(args.config)
    conn = connect_sqlite(CLEAN_DB_PATH)
    init_clean_db(conn)

    metadata_files = sorted(RAW_HTML_DIR.glob("*.json"))
    for metadata_path in metadata_files:
        metadata = read_json(metadata_path)
        slug = metadata["slug"]
        html_path = Path(metadata["raw_html_path"])
        transcript_path = Path(metadata["transcript_path"])

        existing = conn.execute("SELECT status FROM cleaned_transcripts WHERE slug=?", (slug,)).fetchone()
        if existing and existing["status"] in {"cleaned", "skipped_non_english"} and not args.force:
            continue

        if not html_path.exists():
            LOGGER.warning("Missing raw HTML for %s", slug)
            continue

        html_text = html_path.read_text(encoding="utf-8")
        cleaned_text = extract_clean_text(html_text)
        language = detect_language_or_unknown(cleaned_text)
        organized_transcript_path = transcript_storage_path(
            metadata.get("company"),
            metadata.get("fiscal_year"),
            metadata.get("call_date"),
            metadata.get("quarter"),
            metadata.get("ticker"),
        )
        if language != "en":
            conn.execute(
                """
                INSERT INTO cleaned_transcripts (slug, transcript_path, cleaned_at, status, language, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET cleaned_at=excluded.cleaned_at, status=excluded.status, language=excluded.language, notes=excluded.notes
                """,
                (slug, str(organized_transcript_path), utc_now_iso(), "skipped_non_english", language, "non-English transcript"),
            )
            conn.commit()
            LOGGER.info("Skipping non-English transcript %s (%s)", slug, language)
            continue

        organized_transcript_path.parent.mkdir(parents=True, exist_ok=True)
        organized_transcript_path.write_text(cleaned_text + "\n", encoding="utf-8")
        if transcript_path != organized_transcript_path and transcript_path.exists():
            transcript_path.unlink()

        speakers_path = None
        if args.write_speakers or config.write_speakers:
            speaker_rows = extract_speakers(cleaned_text)
            if speaker_rows:
                speakers_path = speakers_storage_path(
                    metadata.get("company"),
                    metadata.get("fiscal_year"),
                    metadata.get("call_date"),
                    metadata.get("quarter"),
                    metadata.get("ticker"),
                )
                speakers_path.parent.mkdir(parents=True, exist_ok=True)
                speakers_path.write_text(json.dumps(speaker_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        legacy_speakers_path = Path(metadata.get("speakers_path", "")) if metadata.get("speakers_path") else None
        if legacy_speakers_path and speakers_path and legacy_speakers_path != speakers_path and legacy_speakers_path.exists():
            legacy_speakers_path.unlink()

        metadata["transcript_path"] = str(organized_transcript_path)
        if speakers_path:
            metadata["speakers_path"] = str(speakers_path)
        write_json(metadata_path, metadata)

        conn.execute(
            """
            INSERT INTO cleaned_transcripts (slug, transcript_path, speakers_path, cleaned_at, status, language, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                transcript_path=excluded.transcript_path,
                speakers_path=excluded.speakers_path,
                cleaned_at=excluded.cleaned_at,
                status=excluded.status,
                language=excluded.language,
                notes=excluded.notes
            """,
            (slug, str(organized_transcript_path), str(speakers_path) if speakers_path else None, utc_now_iso(), "cleaned", language, ""),
        )
        conn.commit()
        LOGGER.info("Cleaned %s", organized_transcript_path)


if __name__ == "__main__":
    main()
