#!/usr/bin/env python3
"""
Transcript sentiment and theme extraction for earnings calls.

This reuses the core idea from finbert.py, but makes it practical for a
directory of transcripts:
1. Read all .txt files under an input folder.
2. Strip common Motley Fool / transcript boilerplate.
3. Split into substantive sentences and tag macro-relevant themes.
4. Score each sentence with FinBERT.
5. Save sentence-level and document-level outputs plus a simple time-series plot.

The default theme set is tuned for earnings calls, but it is intentionally broad
enough to reuse on policy texts later by pointing --input-dir at a different
folder and editing THEME_KEYWORDS if needed.
"""

from __future__ import annotations

import argparse
import logging
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


THEME_KEYWORDS: dict[str, list[str]] = {
    "demand_growth": [
        "demand",
        "orders",
        "bookings",
        "backlog",
        "volume",
        "traffic",
        "consumption",
        "spending",
        "customer demand",
        "sales growth",
        "slowdown",
        "softness",
    ],
    "pricing_inflation": [
        "pricing",
        "price increase",
        "price decreases",
        "inflation",
        "deflation",
        "discount",
        "promotion",
        "tariff",
        "surcharge",
        "mix",
        "asp",
    ],
    "costs_margin": [
        "margin",
        "gross margin",
        "operating margin",
        "cost",
        "costs",
        "efficiency",
        "productivity",
        "savings",
        "restructuring",
        "utilization",
    ],
    "labor": [
        "hiring",
        "headcount",
        "workforce",
        "labor",
        "staffing",
        "wage",
        "wages",
        "attrition",
        "recruiting",
        "overtime",
    ],
    "capex_investment": [
        "capex",
        "capital expenditure",
        "investment",
        "investing",
        "capacity",
        "expansion",
        "buildout",
        "plant",
        "facility",
        "data center",
        "datacenter",
        "infrastructure",
    ],
    "guidance_outlook": [
        "guidance",
        "outlook",
        "expect",
        "expects",
        "expected",
        "anticipate",
        "forecast",
        "confidence",
        "cautious",
        "next quarter",
        "full year",
        "second half",
    ],
    "supply_chain_inventory": [
        "supply chain",
        "inventory",
        "lead time",
        "lead times",
        "logistics",
        "shortage",
        "bottleneck",
        "stocking",
        "destocking",
        "channel inventory",
        "procurement",
        "sourcing",
    ],
    "ai_technology": [
        "ai",
        "artificial intelligence",
        "machine learning",
        "generative ai",
        "automation",
        "inference",
        "cloud",
        "silicon",
        "chip",
        "chips",
        "compute",
        "agent",
    ],
    "macro_policy": [
        "interest rates",
        "rate environment",
        "foreign exchange",
        "fx",
        "currency",
        "recession",
        "macro environment",
        "geopolitical",
        "trade policy",
        "central bank",
        "fed",
        "regulatory",
    ],
}

HEADER_PREFIXES = (
    "Ticker:",
    "Fiscal Year:",
    "Query Used:",
    "Source URL:",
    "Title:",
    "Fetched At (UTC):",
)

DROP_LINE_PATTERNS = [
    re.compile(r"^Image source:", re.IGNORECASE),
    re.compile(r"^Need a quote from a Motley Fool analyst\?", re.IGNORECASE),
    re.compile(r"^[()+-]?\d+(?:\.\d+)?%$"),
    re.compile(r"^[A-Z]{1,6}$"),
]

DROP_SENTENCE_PATTERNS = [
    re.compile(r"\bforward-looking statements?\b", re.IGNORECASE),
    re.compile(r"\bprivate securities litigation reform act\b", re.IGNORECASE),
    re.compile(r"\bfilings with the sec\b", re.IGNORECASE),
    re.compile(r"\bnon-gaap\b", re.IGNORECASE),
    re.compile(r"\breconciliation to\b.*\bgaap\b", re.IGNORECASE),
    re.compile(r"\binvestor relations section\b", re.IGNORECASE),
    re.compile(r"\bpress star(?: then)? 1\b", re.IGNORECASE),
    re.compile(r"\bplease limit yourself to one question\b", re.IGNORECASE),
    re.compile(r"\btelephone keypad\b", re.IGNORECASE),
    re.compile(r"\bpick up your handset\b", re.IGNORECASE),
]

QA_MARKERS = (
    "let's move on to your questions",
    "we will now open the call up for questions",
    "we'll now open the line for questions",
    "question-and-answer session",
    "question and answer session",
    "our first question comes from",
)

DATE_PATTERN = re.compile(
    r"\b(?:Mon|Tue|Tues|Wed|Thu|Thur|Thurs|Fri|Sat|Sun)"
    r"(?:day)?"
    r",?\s+"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\.?"
    r"\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)

SPEAKER_PATTERN = re.compile(r"^([A-Z][A-Za-z&.,'()/\- ]{1,80}?):\s*(.+)$")
logger = logging.getLogger(__name__)


@dataclass
class SentenceUnit:
    company: str
    ticker: str
    fiscal_year: str
    source_file: str
    title: str
    source_url: str
    call_date: str | None
    analysis_date: str | None
    section: str
    speaker: str
    sentence: str
    themes: list[str]


class FinBERTScorer:
    def __init__(self, model_name: str = "ProsusAI/finbert", max_length: int = 256) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise SystemExit(
                "FinBERT dependencies are missing. Install them with "
                "`pip install torch transformers` and rerun."
            ) from exc

        self.torch = torch
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.labels = {
            int(idx): str(label).lower()
            for idx, label in self.model.config.id2label.items()
        }

    def score_texts(self, texts: list[str], batch_size: int = 24) -> list[dict[str, float | str]]:
        outputs: list[dict[str, float | str]] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info(
            "Starting FinBERT scoring for %s sentences in %s batches (batch_size=%s).",
            len(texts),
            total_batches,
            batch_size,
        )
        score_start = perf_counter()
        for batch_index, start in enumerate(range(0, len(texts), batch_size), start=1):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.no_grad():
                logits = self.model(**encoded).logits
                probs = self.torch.softmax(logits, dim=1).cpu().numpy()

            for prob in probs:
                row = {self.labels[i]: float(prob[i]) for i in range(len(prob))}
                label = max(row, key=row.get)
                outputs.append(
                    {
                        "sentiment_label": label,
                        "sentiment_confidence": float(row[label]),
                        "positive_prob": float(row.get("positive", 0.0)),
                        "neutral_prob": float(row.get("neutral", 0.0)),
                        "negative_prob": float(row.get("negative", 0.0)),
                        "sentiment_score": float(
                            row.get("positive", 0.0) - row.get("negative", 0.0)
                        ),
                    }
                )
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
                elapsed = perf_counter() - score_start
                done = min(start + len(batch), len(texts))
                rate = done / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "Scored batch %s/%s (%s/%s sentences, %.1f sentences/sec).",
                    batch_index,
                    total_batches,
                    done,
                    len(texts),
                    rate,
                )
        return outputs


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_company_and_ticker(path: Path) -> tuple[str, str]:
    folder = path.parent.name.strip()
    match = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", folder)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return folder, ""


def extract_header_value(text: str, prefix: str) -> str:
    pattern = re.compile(rf"^{re.escape(prefix)}\s*(.+)$", re.MULTILINE)
    match = pattern.search(text)
    return normalize_space(match.group(1)) if match else ""


def extract_call_date(text: str, fiscal_year: str) -> tuple[str | None, str | None]:
    match = DATE_PATTERN.search(text)
    if match:
        parsed = pd.to_datetime(match.group(0).replace(".", ""), errors="coerce")
        if pd.notna(parsed):
            date_str = parsed.strftime("%Y-%m-%d")
            return date_str, date_str

    year_match = re.search(r"(\d{4})", fiscal_year)
    if year_match:
        fallback = f"{year_match.group(1)}-12-31"
        return None, fallback
    return None, None


def should_drop_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith(HEADER_PREFIXES):
        return True
    return any(pattern.search(stripped) for pattern in DROP_LINE_PATTERNS)


def split_into_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", normalize_space(text))
    return [part.strip() for part in parts if part.strip()]


def is_substantive_sentence(sentence: str, speaker: str) -> bool:
    if speaker.lower() == "operator":
        return False
    if len(re.findall(r"[A-Za-z]", sentence)) < 25:
        return False
    if len(sentence.split()) < 6:
        return False
    return not any(pattern.search(sentence) for pattern in DROP_SENTENCE_PATTERNS)


def contains_keyword(text: str, keyword: str) -> bool:
    low = text.lower()
    key = keyword.lower().strip()
    if not key:
        return False
    if " " in key:
        return key in low
    return bool(re.search(rf"\b{re.escape(key)}\b", low))


def tag_themes(sentence: str) -> list[str]:
    tags: list[str] = []
    for theme, keywords in THEME_KEYWORDS.items():
        if any(contains_keyword(sentence, keyword) for keyword in keywords):
            tags.append(theme)
    return tags


def collect_sentence_units(path: Path) -> list[SentenceUnit]:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    company, ticker = parse_company_and_ticker(path)
    fiscal_year = extract_header_value(raw_text, "Fiscal Year:") or path.stem
    title = extract_header_value(raw_text, "Title:")
    source_url = extract_header_value(raw_text, "Source URL:")
    call_date, analysis_date = extract_call_date(raw_text, fiscal_year)

    units: list[SentenceUnit] = []
    section = "prepared_remarks"
    current_speaker = ""
    started = False

    for raw_line in raw_text.splitlines():
        line = normalize_space(raw_line)
        if should_drop_line(line):
            continue

        if any(marker in line.lower() for marker in QA_MARKERS):
            section = "qa"

        speaker = current_speaker
        content = line
        match = SPEAKER_PATTERN.match(line)
        if match:
            started = True
            speaker = normalize_space(match.group(1))
            current_speaker = speaker
            content = normalize_space(match.group(2))
        elif not started:
            continue

        for sentence in split_into_sentences(content):
            if not is_substantive_sentence(sentence, speaker):
                continue
            units.append(
                SentenceUnit(
                    company=company,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    source_file=str(path),
                    title=title,
                    source_url=source_url,
                    call_date=call_date,
                    analysis_date=analysis_date,
                    section=section,
                    speaker=speaker,
                    sentence=sentence,
                    themes=tag_themes(sentence),
                )
            )

    return units


def units_to_frame(units: list[SentenceUnit]) -> pd.DataFrame:
    frame = pd.DataFrame([unit.__dict__ for unit in units])
    if frame.empty:
        return frame
    frame["themes"] = frame["themes"].apply(list)
    frame["theme_count"] = frame["themes"].apply(len)
    return frame


def shorten_text(text: str, width: int = 180) -> str:
    return textwrap.shorten(text, width=width, placeholder="...")


def summarize_documents(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "company",
        "ticker",
        "fiscal_year",
        "source_file",
        "title",
        "source_url",
        "call_date",
        "analysis_date",
    ]

    for keys, doc in frame.groupby(group_cols, dropna=False):
        relevant = doc[doc["theme_count"] > 0].copy()
        prepared = doc[doc["section"] == "prepared_remarks"]
        qa = doc[doc["section"] == "qa"]

        row = {col: value for col, value in zip(group_cols, keys)}
        row["sentence_count"] = int(len(doc))
        row["relevant_sentence_count"] = int(len(relevant))
        row["theme_coverage"] = float(len(relevant) / len(doc)) if len(doc) else 0.0
        row["overall_sentiment"] = float(doc["sentiment_score"].mean())
        row["prepared_sentiment"] = float(prepared["sentiment_score"].mean()) if len(prepared) else None
        row["qa_sentiment"] = float(qa["sentiment_score"].mean()) if len(qa) else None
        row["positive_share"] = float((doc["sentiment_label"] == "positive").mean())
        row["negative_share"] = float((doc["sentiment_label"] == "negative").mean())

        if len(relevant):
            top_positive = relevant.sort_values("sentiment_score", ascending=False).iloc[0]["sentence"]
            top_negative = relevant.sort_values("sentiment_score", ascending=True).iloc[0]["sentence"]
            row["top_positive_excerpt"] = shorten_text(str(top_positive))
            row["top_negative_excerpt"] = shorten_text(str(top_negative))
        else:
            row["top_positive_excerpt"] = ""
            row["top_negative_excerpt"] = ""

        for theme in THEME_KEYWORDS:
            theme_slice = doc[doc["themes"].apply(lambda tags: theme in tags)]
            row[f"{theme}_count"] = int(len(theme_slice))
            row[f"{theme}_sentiment"] = (
                float(theme_slice["sentiment_score"].mean()) if len(theme_slice) else None
            )

        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["analysis_date", "company", "fiscal_year"])
    return summary


def build_theme_time_series(frame: pd.DataFrame) -> pd.DataFrame:
    relevant = frame[frame["theme_count"] > 0].copy()
    if relevant.empty:
        return pd.DataFrame()

    exploded = relevant.explode("themes").rename(columns={"themes": "theme"})
    series = (
        exploded.groupby(["analysis_date", "theme"], dropna=False)
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            sentence_count=("sentence", "count"),
            doc_count=("source_file", "nunique"),
        )
        .reset_index()
        .sort_values(["analysis_date", "theme"])
    )
    return series


def plot_sentiment_over_time(
    summary: pd.DataFrame, theme_series: pd.DataFrame, output_path: Path
) -> None:
    if summary.empty:
        return

    plot_summary = summary.copy()
    plot_summary["analysis_date"] = pd.to_datetime(plot_summary["analysis_date"], errors="coerce")
    plot_summary = plot_summary.dropna(subset=["analysis_date"])
    if plot_summary.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for _, company_frame in plot_summary.groupby("company"):
        company_frame = company_frame.sort_values("analysis_date")
        if len(company_frame) > 1:
            axes[0].plot(
                company_frame["analysis_date"],
                company_frame["overall_sentiment"],
                alpha=0.25,
                linewidth=1.0,
            )

    universe = (
        plot_summary.groupby("analysis_date", as_index=False)["overall_sentiment"]
        .mean()
        .sort_values("analysis_date")
    )
    axes[0].plot(
        universe["analysis_date"],
        universe["overall_sentiment"],
        color="black",
        linewidth=2.5,
        label="Equal-weight avg",
    )
    axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Overall FinBERT Sentiment Over Time")
    axes[0].set_ylabel("Positive - Negative")
    axes[0].legend(loc="best")

    if not theme_series.empty:
        plot_theme = theme_series.copy()
        plot_theme["analysis_date"] = pd.to_datetime(plot_theme["analysis_date"], errors="coerce")
        plot_theme = plot_theme.dropna(subset=["analysis_date"])
        theme_priority = [
            "demand_growth",
            "pricing_inflation",
            "costs_margin",
            "capex_investment",
            "guidance_outlook",
        ]
        for theme in theme_priority:
            theme_frame = plot_theme[plot_theme["theme"] == theme].sort_values("analysis_date")
            if theme_frame.empty:
                continue
            axes[1].plot(
                theme_frame["analysis_date"],
                theme_frame["avg_sentiment"],
                linewidth=2,
                label=theme,
            )
        axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
        axes[1].set_title("Macro-Relevant Theme Sentiment")
        axes[1].set_ylabel("Positive - Negative")
        axes[1].legend(loc="best", ncol=2)

    axes[1].set_xlabel("Date")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FinBERT over earnings-call transcripts and build simple macro signal outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/earnings"),
        help="Folder containing transcript .txt files (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/earnings_signals"),
        help="Directory to write CSVs and plot.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="FinBERT batch size for sentence scoring.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    run_start = perf_counter()
    transcript_paths = sorted(args.input_dir.rglob("*.txt"))
    if not transcript_paths:
        raise SystemExit(f"No .txt transcripts found under {args.input_dir}")
    logger.info("Found %s transcript files under %s.", len(transcript_paths), args.input_dir)

    all_units: list[SentenceUnit] = []
    parse_start = perf_counter()
    for file_index, path in enumerate(transcript_paths, start=1):
        file_units = collect_sentence_units(path)
        all_units.extend(file_units)
        if file_index == 1 or file_index % 10 == 0 or file_index == len(transcript_paths):
            elapsed = perf_counter() - parse_start
            logger.info(
                "Parsed %s/%s files; extracted %s cumulative sentences (latest file: %s, %s sentences).",
                file_index,
                len(transcript_paths),
                len(all_units),
                path.name,
                len(file_units),
            )
            if elapsed > 0:
                logger.info("Current parse rate: %.2f files/sec.", file_index / elapsed)

    analysis_frame = units_to_frame(all_units)
    if analysis_frame.empty:
        raise SystemExit("No substantive transcript sentences were found after cleaning.")
    logger.info(
        "Prepared %s sentences for scoring; %s have at least one theme tag.",
        len(analysis_frame),
        int((analysis_frame["theme_count"] > 0).sum()),
    )

    model_load_start = perf_counter()
    scorer = FinBERTScorer()
    logger.info("Loaded FinBERT in %.1f seconds.", perf_counter() - model_load_start)

    scored = scorer.score_texts(analysis_frame["sentence"].tolist(), batch_size=args.batch_size)
    score_frame = pd.DataFrame(scored)
    analysis_frame = pd.concat([analysis_frame.reset_index(drop=True), score_frame], axis=1)

    logger.info("Building summary tables.")
    summary = summarize_documents(analysis_frame)
    theme_series = build_theme_time_series(analysis_frame)
    output_frame = analysis_frame.copy()
    output_frame["themes"] = output_frame["themes"].apply(lambda tags: "|".join(tags))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sentence_path = args.output_dir / "sentence_signals.csv"
    summary_path = args.output_dir / "document_summary.csv"
    theme_path = args.output_dir / "theme_time_series.csv"
    plot_path = args.output_dir / "sentiment_over_time.png"

    logger.info("Writing outputs to %s.", args.output_dir)
    output_frame.sort_values(["analysis_date", "company", "fiscal_year"]).to_csv(
        sentence_path, index=False
    )
    summary.to_csv(summary_path, index=False)
    theme_series.to_csv(theme_path, index=False)
    plot_sentiment_over_time(summary, theme_series, plot_path)
    logger.info("Finished in %.1f seconds.", perf_counter() - run_start)

    print(f"Wrote {sentence_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {theme_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
