"""
Shared helpers for earnings and AI-labor keyword scoring (Group 3).
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_ANALYZER = SentimentIntensityAnalyzer()

# Earnings call themes — expand lists as you add real transcripts.
EARNINGS_THEMES: dict[str, list[str]] = {
    "demand": [
        "demand",
        "volume",
        "orders",
        "backlog",
        "traffic",
        "consumption",
        "sales",
        "revenue growth",
        "utilization",
    ],
    "hiring": [
        "hiring",
        "headcount",
        "workforce",
        "labor",
        "attrition",
        "recruiting",
        "seasonal roles",
        "engineering talent",
        "technicians",
    ],
    "pricing": [
        "pricing",
        "price",
        "yield",
        "margin",
        "inflation",
        "surge",
        "rent spreads",
        "premium",
    ],
    "capex": [
        "capex",
        "capital expenditure",
        "investment in",
        "facility",
        "expansion",
        "modernization",
        "remodel",
        "datacenter",
    ],
    "AI": [
        "ai",
        "artificial intelligence",
        "machine learning",
        "generative",
        "copilot",
        "automation",
        "inference",
        "algorithm",
    ],
    "efficiency": [
        "efficiency",
        "productivity",
        "lean",
        "cost actions",
        "cost per unit",
        "yield improvement",
        "automation",
    ],
}

# AI labor / productivity proxy — distinct from generic “AI” theme density.
AI_LABOR: dict[str, list[str]] = {
    "ai_tech": [
        "artificial intelligence",
        "machine learning",
        "generative ai",
        "gen ai",
        "copilot",
        "inference",
        "ai services",
        "ai chip",
        "ai-driven",
        "ai assistant",
        "ai",
    ],
    "labor_down": [
        "layoff",
        "layoffs",
        "headcount reduction",
        "rightsizing",
        "restructuring",
        "hiring freeze",
        "paused hiring",
        "reduce headcount",
        "fewer employees",
    ],
    "labor_up": [
        "hiring aggressively",
        "adding headcount",
        "increase headcount",
        "hiring engineers",
        "seasonal roles",
        "recruiting",
        "adding roles",
        "hiring in",
    ],
    "productivity": [
        "productivity",
        "output per",
        "efficiency gains",
        "route optimization",
        "predictive maintenance",
        "digital twin",
        "labor scheduling",
        "uptime",
    ],
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def read_transcript(path: Path) -> str:
    p = path if path.is_absolute() else repo_root() / path
    return p.read_text(encoding="utf-8", errors="replace")


def word_count(text: str) -> int:
    words = re.findall(r"[A-Za-z']+", text.lower())
    return max(len(words), 1)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def count_theme_hits(text: str, keywords: Iterable[str]) -> int:
    t = text.lower()
    n = 0
    for kw in keywords:
        k = kw.lower().strip()
        if not k:
            continue
        if " " in k:
            n += t.count(k)
        else:
            n += len(re.findall(r"\b" + re.escape(k) + r"\b", t))
    return n


def sentences_touching_keywords(text: str, keywords: list[str]) -> list[str]:
    sents = split_sentences(text)
    out: list[str] = []
    for s in sents:
        low = s.lower()
        if any(
            (k in low if " " in k else bool(re.search(r"\b" + re.escape(k.lower()) + r"\b", low)))
            for k in keywords
        ):
            out.append(s)
    return out


def vader_compound(text: str) -> float:
    if not text.strip():
        return 0.0
    return float(_ANALYZER.polarity_scores(text)["compound"])


def theme_sentiment(text: str, keywords: list[str]) -> float:
    sents = sentences_touching_keywords(text, keywords)
    if not sents:
        return vader_compound(text)
    return sum(vader_compound(s) for s in sents) / len(sents)


def theme_density_score(text: str, keywords: list[str], words: int) -> float:
    hits = count_theme_hits(text, keywords)
    return (hits / words) * 1000.0


def theme_adjusted_score(text: str, keywords: list[str], words: int) -> float:
    density = theme_density_score(text, keywords, words)
    sent = theme_sentiment(text, keywords)
    mult = 0.5 + 0.5 * max(-1.0, min(1.0, sent))
    return density * mult


def load_companies_rows(path: Path) -> list[dict[str, str]]:
    p = path if path.is_absolute() else repo_root() / path
    with p.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"symbol", "name", "region", "sector", "transcript_path"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"companies csv must have columns: {required}")
        return list(reader)


def mean_by_region(rows: list[dict], region_key: str = "region", value_key: str = "score") -> list[dict]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        reg = r[region_key]
        sums[reg] += float(r[value_key])
        counts[reg] += 1
    out: list[dict] = []
    for reg in sorted(sums.keys()):
        n = counts[reg]
        out.append({region_key: reg, value_key: round(sums[reg] / n, 4), "n_companies": n})
    return out
