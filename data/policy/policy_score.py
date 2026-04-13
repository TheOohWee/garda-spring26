from pathlib import Path
import csv
import hashlib
import re
from typing import Dict, List, Iterable

import pandas as pd

NOISE_PREFIXES = (
    "chart ",
    "source:",
    "sources:",
    "note:",
    "last observation:",
    "last data plotted:",
    "sync to video time",
    "all from federal reserve",
    "endnotes",
)

QUESTION_LIKE_PREFIXES = (
    "thank you",
    "thanks",
    "hi chair",
    "hi governor",
    "hi president",
    "good afternoon",
    "i wanted to ask",
    "my question is",
    "can you",
    "could you",
    "would you",
    "how do you",
    "how would you",
    "what do you",
    "why do you",
    "do you",
    "is there a risk",
    "question",
)

QUESTION_LIKE_SUBSTRINGS = (
    "your view",
    "president trump says",
    "your second question",
    "it looks like",
    "you know",
    " uh ",
    " um ",
    "first of all",
    "sorry",
    "so calling that",
)

KEYWORD_PATTERN_CACHE: Dict[str, re.Pattern] = {}


def clean_text(text: str) -> str:
    """Normalize central-bank transcripts before sentence splitting."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\ufeff", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u00a0", " ")
    text = text.replace("Â¼", "1/4").replace("Â½", "1/2").replace("Â¾", "3/4")
    text = text.replace("â€‘", "-").replace("â€”", "-").replace("â€™", "'")
    text = text.replace("â€œ", '"').replace("â€\x9d", '"').replace("â†", " ")
    text = re.sub(r"\b\d{1,2}:\d{2}\s+\d+\s+minutes?(?:,\s*\d+\s+seconds?)?\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{1,2}:\d{2}\s+\d+\s+seconds?\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPage\s+\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\s+Chart\s+\d+:", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bChart\s+\d+:", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bTable\s+\d+:", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text.replace("\n", " "))
    text = re.sub(r"(?<=\D)\d+\s+(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_sentence(sentence: str) -> str:
    sentence = clean_text(sentence)
    sentence = re.sub(r"^\d+\s+", "", sentence)
    sentence = re.sub(r"^[\-\.:;,\)\]]+\s*", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


def _is_noise_sentence(sentence: str) -> bool:
    lowered = sentence.lower().strip()
    if not lowered:
        return True
    if lowered.startswith(NOISE_PREFIXES):
        return True
    if "minutes, " in lowered or "seconds" in lowered[:80]:
        return True
    if "quarterly data" in lowered or "monthly data" in lowered:
        return True
    if "balance of opinion" in lowered or "percent or percentage points" in lowered:
        return True
    if "numbers may not sum" in lowered:
        return True
    if not re.search(r"[A-Za-z]", sentence):
        return True
    letters = sum(char.isalpha() for char in sentence)
    digits = sum(char.isdigit() for char in sentence)
    if letters < 8:
        return True
    if digits > letters and "%" not in sentence:
        return True
    return False


def is_question_like_sentence(sentence: str) -> bool:
    lowered = sentence.lower().strip()
    if not lowered:
        return False
    if lowered.startswith(QUESTION_LIKE_PREFIXES):
        return True
    if any(marker in lowered for marker in QUESTION_LIKE_SUBSTRINGS):
        return True
    if lowered.endswith("?"):
        return True
    if "?" in lowered:
        return True
    if lowered.count(" you ") >= 2 and any(
        marker in lowered for marker in ["would you", "could you", "can you", "do you", "how do you", "what do you"]
    ):
        return True
    return False


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using lightweight regex rules."""
    cleaned = clean_text(text)
    if not cleaned:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    cleaned_sentences = []
    for sentence in sentences:
        normalized = _normalize_sentence(sentence)
        if not normalized or _is_noise_sentence(normalized):
            continue
        cleaned_sentences.append(normalized)
    return cleaned_sentences


def keyword_in_text(text: str, keyword: str) -> bool:
    pattern = KEYWORD_PATTERN_CACHE.get(keyword)
    if pattern is None:
        escaped = re.escape(keyword.lower()).replace(r"\ ", r"\s+")
        pattern = re.compile(rf"\b{escaped}\b")
        KEYWORD_PATTERN_CACHE[keyword] = pattern
    return pattern.search(text) is not None


def count_keyword_matches(text: str, keywords: List[str]) -> int:
    return sum(keyword_in_text(text, keyword) for keyword in keywords)

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "inflation": [
        "inflation",
        "price",
        "prices",
        "cpi",
        "pce",
        "cost pressures",
        "disinflation",
        "inflation expectations",
        "headline inflation",
        "core inflation",
        "underlying inflation",
        "services inflation",
        "goods inflation",
        "price stability",
        "price pressures",
        "second-round effects",
        "inflation target",
        "core measures",
        "inflationary",
        "disinflationary",
        "costs",
        "temporary",
        "transitory",
    ],
    "labor": [
        "labor market",
        "employment",
        "unemployment",
        "jobs",
        "payrolls",
        "wage",
        "wages",
        "hiring",
        "labour market",
        "labour",
        "job vacancy",
        "vacancy",
        "labor shortage",
        "labour shortage",
        "compensation",
        "labour demand",
        "labor demand",
        "hiring intentions",
        "labour market slack",
        "labor market slack",
        "job gains",
        "layoffs",
        "workers",
        "worker",
    ],
    "growth": [
        "growth",
        "economic activity",
        "economy",
        "output",
        "demand",
        "spending",
        "expanding",
        "slowing",
        "weak",
        "recession",
        "domestic demand",
        "consumption",
        "investment",
        "business fixed investment",
        "private consumption",
        "downside risks for economic growth",
        "real incomes",
        "corporate profits",
        "outlook",
        "vulnerable",
        "trade",
        "tariffs",
        "stalled",
        "stall",
        "subdued",
        "restrained",
        "flat",
        "volatile",
        "slowdown",
        "slow",
        "weighed down",
        "weighed heavily",
        "excess supply",
        "gdp",
        "gross domestic product",
        "exports",
        "export",
    ],
    "guidance": [
        "policy",
        "rate",
        "rates",
        "restrictive",
        "easing",
        "normalization",
        "future adjustments",
        "stance",
        "hold",
        "higher for longer",
        "policy rate",
        "policy interest rate",
        "monetary accommodation",
        "monetary policy stance",
        "restrictiveness",
        "data-dependent",
        "meeting-by-meeting",
        "rate path",
        "pre-committing",
        "interest rate decisions",
        "target range",
        "federal funds rate",
        "adjustments to our policy rate",
        "policy stance",
    ],
}


HAWKISH_WORDS: Dict[str, List[str]] = {
    "inflation": [
        "elevated",
        "persistent",
        "sticky",
        "upside",
        "upside risks",
        "firm inflation",
        "firm price pressures",
        "higher inflation",
        "inflationary pressures",
        "price pressures",
        "above target",
        "reaccelerat",
        "second-round effects",
        "broad-based",
        "not enough confidence",
        "inflation expectations have moved up",
        "inflation expectations have risen",
        "inflation remains elevated",
        "remain elevated",
        "still elevated",
        "persistent inflation",
        "moved up significantly",
        "risen significantly",
        "upside risks to inflation",
        "inflation persistence has risen",
        "higher energy prices",
        "price rises",
        "doing what it takes",
        "keep inflation expectations anchored",
        "pressure on inflation",
        "pushing inflation up",
        "inflation compensation had increased",
        "inflation compensation had increased markedly",
        "inflation had risen",
        "inflation had risen relative to",
        "higher frequency measures of underlying services inflation had picked up",
        "services inflation had been",
        "delay the return of cpi inflation",
        "cpi inflation could increase",
        "inflationary shock",
        "higher energy costs",
        "higher fuel prices",
        "oil prices had increased",
        "energy prices would delay the return",
        "push up cpi inflation",
        "rise above 3%",
        "return of cpi inflation to the 2% target",
        "inflation outlook had risen",
    ],
    "labor": [
        "tight",
        "strong",
        "firm wage",
        "wage pressures",
        "labor shortage",
        "labour shortage",
        "strong employment",
        "solid job gains",
        "robust payroll",
        "elevated wage growth",
        "wages steadily",
        "labour market remains tight",
        "labor market remains tight",
        "employment remains strong",
    ],
    "growth": [
        "resilient",
        "solid",
        "strong demand",
        "robust",
        "stronger domestic demand",
        "economy has been resilient",
        "economy remains resilient",
        "growth has been revised up",
        "underpinning growth",
        "supportive effects",
        "rebounding",
        "expanding rapidly",
        "continued expansion",
        "strong growth",
        "grew strongly",
        "strengthened",
        "expected to rise",
        "rise in consumption growth",
    ],
    "guidance": [
        "restrictive",
        "higher for longer",
        "tightening",
        "not ready to cut",
        "sufficiently restrictive",
        "remain restrictive",
        "keep policy sufficiently restrictive",
        "continue to raise",
        "raise the policy interest rate",
        "raise rates",
        "higher policy rate",
        "further tightening",
        "reduce the degree of monetary accommodation",
        "adjust the degree of monetary accommodation",
        "not materially restrictive",
        "hold the policy rate steady",
        "hold policy rate steady",
        "easing may not be warranted",
        "not be warranted until",
        "hold rates unchanged",
        "prepared to do what needs to be done",
        "upside risks to inflation",
        "higher rates",
        "market-implied path for bank rate had increased",
        "bank rate had increased",
        "hold or increase in bank rate",
        "longer hold",
        "even a hike",
        "pausing to reassess",
        "stand ready to act as necessary",
        "assessing the implications for inflation",
    ],
}


DOVISH_WORDS: Dict[str, List[str]] = {
    "inflation": [
        "disinflation",
        "moderating",
        "cooling",
        "easing",
        "below target",
        "return to target",
        "stabilises at our two per cent target",
        "stabilises at its 2% target",
        "well anchored",
        "anchored inflation expectations",
        "on track towards target",
        "on track to target",
        "towards target",
        "decline in inflation",
        "inflation declined",
        "inflation fell",
        "subdued inflation",
        "inflation has eased",
        "inflation has softened",
        "close to 2%",
        "close to target",
        "near target",
        "near the 2% target",
        "within the 1% to 3% band",
        "briefly push up",
    ],
    "labor": [
        "softening",
        "cooling",
        "weaker hiring",
        "slower payroll",
        "labour demand cooled",
        "labor demand cooled",
        "rising unemployment",
        "weaken significantly further",
        "slack",
        "job losses",
        "layoffs",
        "labour market has loosened",
        "labor market has loosened",
        "loosened materially",
        "cooling wages",
        "wage growth will ease",
        "employment has risen",
        "hiring intentions are still weak",
        "job vacancies have fallen",
        "unemployment remains elevated",
    ],
    "growth": [
        "slowing",
        "weak",
        "sluggish",
        "loss of momentum",
        "downside risks for economic growth",
        "downward revision",
        "weigh on growth",
        "dampen demand",
        "weaker growth",
        "slowdown",
        "recession",
        "subdued demand",
        "challenging environment",
        "stalled",
        "stall",
        "flat",
        "close to zero",
        "restrained",
        "weighed down",
        "weighed heavily",
        "vulnerable",
        "slow population growth",
        "weakness in exports",
        "stagnant",
        "stagnation",
        "growth likely stalled",
        "activity weakened",
        "economic activity weakened",
        "demand softened",
    ],
    "guidance": [
        "less restrictive",
        "lower rates",
        "rate cuts",
        "reduce rates",
        "policy support",
        "accommodative financial conditions",
        "accommodative financial environment",
        "maintain an accommodative financial environment",
        "maintain accommodative financial conditions",
        "continue with monetary easing",
        "continue monetary easing",
        "maintain monetary easing",
        "monetary easing will continue",
        "additional easing",
        "support economic activity",
        "lower the target range",
        "lower our policy rate",
        "reduce the policy rate",
        "policy support",
        "rate reduction",
        "rate cut",
    ],
}


EXTREME_WORDS = [
    "significantly",
    "sharply",
    "substantially",
    "considerably",
    "materially",
    "markedly",
    "pronounced",
    "highly likely",
    "firmly",
    "strongly",
    "sharper",
    "significant",
]


SLIGHT_WORDS = [
    "slightly",
    "somewhat",
    "modestly",
    "marginally",
    "gradually",
    "a bit",
    "some",
    "little",
    "roughly",
    "close to",
    "near",
]


TIER_WEIGHTS = {
    1: 1.5,
    2: 1.0,
    3: 0.5,
}


RECENCY_HALFLIFE_DAYS = 180
RECENCY_WEIGHT_FLOOR = 0.1


CATEGORY_SENTIMENT_WEIGHTS = {
    "inflation": 1.6,
    "guidance": 1.8,
    "labor": 0.8,
    "growth": 0.45,
}


CATEGORY_SIGNAL_WEIGHTS = {
    "inflation": 1.0,
    "guidance": 1.0,
    "labor": 0.75,
    "growth": 0.45,
}


DOC_TYPE_REGIONAL_WEIGHTS = {
    "policy_statement": 1.0,
    "press_conference": 1.0,
    "minutes": 0.9,
    "speech": 0.45,
}

STANDARDIZATION_DENOMINATOR = 0.15


def detect_categories(sentence: str) -> List[str]:
    sentence_lower = sentence.lower()
    matches = []

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword_in_text(sentence_lower, keyword) for keyword in keywords):
            matches.append(category)

    if not matches:
        fallback_category_rules = {
            "inflation": ["target", "disinflation", "eased", "softened", "inflationary"],
            "labor": ["vacancies", "hiring intentions", "layoffs", "job gains", "employment", "workers"],
            "growth": ["stalled", "subdued", "vulnerable", "uncertainty", "flat", "weighed down", "exports", "gdp"],
            "guidance": ["target range", "policy rate", "federal funds rate", "adjustments to our policy"],
        }
        for category, keywords in fallback_category_rules.items():
            if any(keyword_in_text(sentence_lower, keyword) for keyword in keywords):
                matches.append(category)

    if not matches:
        if any(keyword_in_text(sentence_lower, keyword) for keyword in ["inflation", "price", "prices", "target"]):
            matches.append("inflation")
        elif any(keyword_in_text(sentence_lower, keyword) for keyword in ["employment", "unemployment", "labour", "labor", "wage", "worker"]):
            matches.append("labor")
        elif any(keyword_in_text(sentence_lower, keyword) for keyword in ["rate", "policy", "stance", "committee", "bank rate"]):
            matches.append("guidance")

    return matches


def sentiment_score(sentence: str, categories: List[str]) -> int:
    """Return -1 for hawkish, +1 for dovish, 0 for neutral."""
    sentence_lower = sentence.lower()
    hawk_count = 0.0
    dove_count = 0.0

    explicit_hawkish_guidance = [
        "we didn't cut rates",
        "didn't cut rates",
        "did not cut rates",
        "adjust the degree of monetary easing",
        "market-implied path for bank rate had increased",
        "bank rate had increased",
        "hold or increase in bank rate",
        "longer hold",
        "even a hike",
        "not be warranted until",
        "easing may not be warranted",
        "continue to raise",
        "raise the policy interest rate",
        "change in the policy interest rate",
        "increase in the policy interest rate",
        "adjust the degree of monetary accommodation",
        "normalisation of monetary policy",
        "normalization of monetary policy",
        "exit from monetary easing",
        "reduce the bank's market presence",
        "reducing the bank's market presence",
        "side effects of monetary easing",
        "easing exerted significant side effects",
        "end new lending",
        "real interest rates are currently at significantly low levels",
        "real interest rates are at significantly low levels",
        "real interest rates will remain significantly negative",
        "real interest rates are significantly negative",
        "well positioned to determine the extent and timing",
        "higher for longer",
        "remain restrictive",
        "restrictive level for too long",
        "upside risks appeared to have increased",
    ]
    explicit_dovish_guidance = [
        "lower the target range",
        "lower our policy rate",
        "reduce the policy rate",
        "make policy less restrictive",
        "policy less restrictive",
        "gradual downward path",
        "continue on a gradual downward path",
        "continue to lower bank rate",
        "voted to reduce bank rate",
        "cut bank rate",
        "rate cut",
        "faster and potentially deeper rate cuts",
        "policy should continue on a gradual downward path",
    ]
    explicit_hawkish_inflation = [
        "remains somewhat elevated",
        "remains elevated",
        "still elevated",
        "elevated relative to",
        "inflation expectations have moved up",
        "inflation expectations have risen",
        "moved up significantly",
        "risen significantly",
        "upside risks to inflation",
        "inflation persistence has risen",
        "higher energy prices",
        "price rises",
        "doing what it takes",
        "keep inflation expectations anchored",
        "pushing inflation up",
        "overshoot",
        "oil prices had increased significantly",
        "higher energy prices",
        "delay the return of cpi inflation",
        "cpi inflation was now expected to be",
        "cpi inflation could increase",
        "near-term outlook for cpi inflation had risen",
        "inflationary impulse",
        "inflation had increased markedly",
    ]
    explicit_dovish_inflation = [
        "on track towards target",
        "on track to meet the 2% target",
        "downward trend toward 2%",
        "inflation has eased",
        "inflation has softened",
        "continued disinflation",
        "disinflation is on track",
        "upside risks have eased",
    ]
    explicit_dovish_labor = [
        "unemployment rate remains elevated",
        "unemployment remains elevated",
        "few businesses say they plan to hire more workers",
        "plan to hire more workers",
        "building slack in the labour market",
        "building slack in the labor market",
        "loosening labour market",
        "loosening labor market",
    ]
    explicit_dovish_growth = [
        "risks to growth are tilted to the downside",
        "risks to growth look tilted to the downside",
        "downside risks to growth",
        "subdued economic growth",
        "weaker demand",
        "growth are tilted to the downside",
        "growth remains weak",
    ]

    if "guidance" in categories:
        hawk_count += 3.0 * count_keyword_matches(sentence_lower, explicit_hawkish_guidance)
        dove_count += 2.0 * count_keyword_matches(sentence_lower, explicit_dovish_guidance)
    if "inflation" in categories:
        hawk_count += 3.0 * count_keyword_matches(sentence_lower, explicit_hawkish_inflation)
        dove_count += 2.0 * count_keyword_matches(sentence_lower, explicit_dovish_inflation)
    if "labor" in categories:
        dove_count += 2.0 * count_keyword_matches(sentence_lower, explicit_dovish_labor)
    if "growth" in categories:
        dove_count += 2.0 * count_keyword_matches(sentence_lower, explicit_dovish_growth)

    for category in categories:
        category_weight = CATEGORY_SENTIMENT_WEIGHTS.get(category, 1.0)
        hawk_hits = count_keyword_matches(sentence_lower, HAWKISH_WORDS.get(category, []))
        dove_hits = count_keyword_matches(sentence_lower, DOVISH_WORDS.get(category, []))
        if hawk_hits > dove_hits:
            hawk_count += category_weight * min(1.5, 1.0 + 0.25 * (hawk_hits - dove_hits - 1))
        elif dove_hits > hawk_hits:
            dove_count += category_weight * min(1.35, 1.0 + 0.2 * (dove_hits - hawk_hits - 1))

    contrast_markers = [" but ", " however ", " although ", " while "]
    for marker in contrast_markers:
        if marker in sentence_lower:
            trailing_clause = sentence_lower.split(marker, 1)[1]
            for category in categories:
                category_weight = CATEGORY_SENTIMENT_WEIGHTS.get(category, 1.0)
                hawk_hits = count_keyword_matches(trailing_clause, HAWKISH_WORDS.get(category, []))
                dove_hits = count_keyword_matches(trailing_clause, DOVISH_WORDS.get(category, []))
                if hawk_hits > dove_hits:
                    hawk_count += 1.35 * category_weight
                elif dove_hits > hawk_hits:
                    dove_count += 1.2 * category_weight
            hawk_count += 1.5 * count_keyword_matches(trailing_clause, explicit_hawkish_guidance)
            dove_count += 1.25 * count_keyword_matches(trailing_clause, explicit_dovish_guidance)
            hawk_count += 1.5 * count_keyword_matches(trailing_clause, explicit_hawkish_inflation)
            dove_count += 1.25 * count_keyword_matches(trailing_clause, explicit_dovish_inflation)
            break

    if hawk_count == 0 and dove_count == 0:
        generic_hawkish = [
            "elevated",
            "persistent",
            "sticky",
            "resilient",
            "robust",
            "tight",
            "higher for longer",
            "restrictive",
        ]
        generic_dovish = [
            "stalled",
            "stall",
            "softened",
            "eased",
            "slowing",
            "subdued",
            "modest",
            "flat",
            "close to target",
            "near target",
            "weighed down",
            "weighed heavily",
        ]
        hawk_count += count_keyword_matches(sentence_lower, generic_hawkish)
        dove_count += count_keyword_matches(sentence_lower, generic_dovish)

    if hawk_count > dove_count:
        return -1
    if dove_count > hawk_count:
        return 1
    return 0


def magnitude_score(sentence: str) -> float:
    sentence_lower = sentence.lower()

    if any(keyword_in_text(sentence_lower, word) for word in EXTREME_WORDS):
        return 2.0
    if any(keyword_in_text(sentence_lower, word) for word in SLIGHT_WORDS):
        return 0.5
    return 1.0


def category_adjustment_weight(categories: List[str]) -> float:
    if not categories:
        return 0.0
    weights = [CATEGORY_SIGNAL_WEIGHTS.get(category, 1.0) for category in categories]
    return sum(weights) / len(weights)


def recency_weight(document_date: str, max_document_date: pd.Timestamp) -> float:
    parsed_date = pd.to_datetime(document_date, errors="coerce")
    if pd.isna(parsed_date) or pd.isna(max_document_date):
        return 1.0
    age_days = max((max_document_date - parsed_date).days, 0)
    decay_component = 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)
    return round(RECENCY_WEIGHT_FLOOR + (1 - RECENCY_WEIGHT_FLOOR) * decay_component, 3)


def tier_weight(tier: int) -> float:
    return TIER_WEIGHTS.get(tier, 1.0)

from pathlib import Path
import re
from typing import Iterable

import pandas as pd



CANONICAL_COLUMNS = [
    "doc_id",
    "central_bank",
    "region",
    "date",
    "tier",
    "doc_type",
    "speaker",
    "text",
]

TARGET_CENTRAL_BANKS = ["Fed", "ECB", "BOE", "BOJ", "BOC"]

CENTRAL_BANK_ALIASES = {
    "fed": "Fed",
    "federal reserve": "Fed",
    "fomc": "Fed",
    "ecb": "ECB",
    "european central bank": "ECB",
    "boe": "BOE",
    "bank of england": "BOE",
    "boj": "BOJ",
    "bank of japan": "BOJ",
    "boc": "BOC",
    "bank of canada": "BOC",
}

REGION_ALIASES = {
    "us": "US",
    "united states": "US",
    "eu": "Eurozone",
    "eurozone": "Eurozone",
    "euro area": "Eurozone",
    "euro-area": "Eurozone",
    "jp": "Japan",
    "japan": "Japan",
    "uk": "UK",
    "united kingdom": "UK",
    "ca": "Canada",
    "canada": "Canada",
}

DOC_TYPE_ALIASES = {
    "press conference": "press_conference",
    "press_conference": "press_conference",
    "press conference transcript": "press_conference",
    "pressconf": "press_conference",
    "policy statement": "policy_statement",
    "policy_statement": "policy_statement",
    "statement": "policy_statement",
    "speech": "speech",
    "remarks": "speech",
    "testimony": "testimony",
    "minutes": "minutes",
    "meeting minutes": "minutes",
}

DEFAULT_SPEAKERS = {
    "Fed": {
        "press_conference": "Jerome Powell",
        "speech": "Jerome Powell",
        "policy_statement": "Federal Reserve",
        "minutes": "FOMC",
        "testimony": "Federal Reserve",
        "unknown": "Federal Reserve",
    },
    "ECB": {
        "press_conference": "Christine Lagarde",
        "speech": "Christine Lagarde",
        "policy_statement": "European Central Bank",
        "minutes": "ECB",
        "testimony": "European Central Bank",
        "unknown": "European Central Bank",
    },
    "BOE": {
        "press_conference": "Andrew Bailey",
        "speech": "Andrew Bailey",
        "policy_statement": "Bank of England",
        "minutes": "Bank of England",
        "testimony": "Bank of England",
        "unknown": "Bank of England",
    },
    "BOJ": {
        "press_conference": "Kazuo Ueda",
        "speech": "Kazuo Ueda",
        "policy_statement": "Bank of Japan",
        "minutes": "Bank of Japan",
        "testimony": "Bank of Japan",
        "unknown": "Bank of Japan",
    },
    "BOC": {
        "press_conference": "Tiff Macklem",
        "speech": "Tiff Macklem",
        "policy_statement": "Bank of Canada",
        "minutes": "Bank of Canada",
        "testimony": "Bank of Canada",
        "unknown": "Bank of Canada",
    },
}


TARGET_START = pd.Timestamp("2020-01-01")
TARGET_END = pd.Timestamp("2026-12-31")


RESEARCH_DOC_SOURCES = [
    ("Fed", "policy_statement", Path("Central_Bank_Research_Data/fed/statement")),
    ("Fed", "minutes", Path("Central_Bank_Research_Data/fed/minutes")),
    ("Fed", "press_conference", Path("Central_Bank_Research_Data/fed/press_conference")),
    ("ECB", "policy_statement", Path("Central_Bank_Research_Data/ecb/decision")),
    ("ECB", "minutes", Path("Central_Bank_Research_Data/ecb/minutes")),
    ("ECB", "press_conference", Path("Central_Bank_Research_Data/ecb/press_conference")),
    ("BOE", "policy_statement", Path("Central_Bank_Research_Data/boe/decision")),
    ("BOJ", "policy_statement", Path("Central_Bank_Research_Data/boj/statement")),
    ("BOJ", "minutes", Path("Central_Bank_Research_Data/boj/minutes")),
    ("BOC", "policy_statement", Path("Central_Bank_Research_Data/boc/statement")),
]


BIS_BANK_PATTERNS = {
    "Fed": [
        "federal reserve system",
        "federal open market committee",
        "board of governors of the federal reserve system",
    ],
    "ECB": [
        "european central bank",
        "ecb",
    ],
    "BOE": [
        "bank of england",
    ],
    "BOJ": [
        "bank of japan",
    ],
    "BOC": [
        "bank of canada",
    ],
}


def slugify(value: str, max_length: int = 48) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    if not text:
        text = "document"
    return text[:max_length].strip("_") or "document"


def clean_source_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.replace("\ufeff", " ").replace("\u00a0", " ")
    cleaned = cleaned.replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" ?\n ?", "\n", cleaned)
    return cleaned.strip()


def normalize_for_fingerprint(text: str) -> str:
    lowered = clean_source_text(text).lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def text_fingerprint(text: str) -> str:
    normalized = normalize_for_fingerprint(text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def parse_source_date(value: str) -> str:
    parsed = pd.to_datetime(str(value).strip(), errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def source_date_in_range(date_text: str) -> bool:
    parsed = pd.to_datetime(date_text, errors="coerce")
    if pd.isna(parsed):
        return False
    return TARGET_START <= parsed <= TARGET_END


def corpus_default_speaker(bank: str, doc_type: str) -> str:
    return DEFAULT_SPEAKERS.get(bank, {}).get(doc_type, bank)


def infer_bis_bank(description: str, title: str, author: str) -> str:
    haystack = " ".join([str(description), str(title), str(author)]).lower()
    for bank, patterns in BIS_BANK_PATTERNS.items():
        if any(pattern in haystack for pattern in patterns):
            return bank
    return ""


def parse_research_date(file_path: Path) -> str:
    match = re.match(r"(\d{4}-\d{2}-\d{2})", file_path.stem)
    return match.group(1) if match else ""


def stable_doc_id(bank: str, date_text: str, doc_type: str, speaker: str, title: str, fingerprint: str) -> str:
    speaker_slug = slugify(speaker, 28)
    title_slug = slugify(title, 28)
    suffix_parts = [part for part in [speaker_slug, title_slug] if part and part != "document"]
    suffix = suffix_parts[0] if suffix_parts else fingerprint[:8]
    return f"{bank.lower()}_{date_text}_{doc_type}_{suffix}_{fingerprint[:8]}"


def add_corpus_record(
    records: list[dict],
    seen_fingerprints: set[str],
    *,
    bank: str,
    date_text: str,
    doc_type: str,
    speaker: str,
    text: str,
    source_family: str,
    source_path: str,
    title: str = "",
) -> None:
    cleaned_text = clean_source_text(text)
    if len(cleaned_text) < 100:
        return
    if not source_date_in_range(date_text):
        return

    fingerprint = text_fingerprint(cleaned_text)
    if not fingerprint or fingerprint in seen_fingerprints:
        return

    doc_id = stable_doc_id(bank, date_text, doc_type, speaker, title, fingerprint)
    seen_fingerprints.add(fingerprint)
    records.append(
        {
            "doc_id": doc_id,
            "central_bank": bank,
            "region": standardize_region("", bank),
            "date": date_text,
            "tier": 1 if doc_type != "minutes" else 2,
            "doc_type": doc_type,
            "speaker": speaker or corpus_default_speaker(bank, doc_type),
            "text": cleaned_text,
            "source_family": source_family,
            "source_path": source_path.replace("\\", "/"),
        }
    )


def collect_research_documents(data_dir: Path, records: list[dict], seen_fingerprints: set[str]) -> None:
    for bank, doc_type, relative_folder in RESEARCH_DOC_SOURCES:
        folder = data_dir / relative_folder
        if not folder.exists():
            continue

        for file_path in sorted(folder.glob("*.txt")):
            date_text = parse_research_date(file_path)
            if not source_date_in_range(date_text):
                continue
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            add_corpus_record(
                records,
                seen_fingerprints,
                bank=bank,
                date_text=date_text,
                doc_type=doc_type,
                speaker=corpus_default_speaker(bank, doc_type),
                text=text,
                source_family="central_bank_research",
                source_path=str(file_path.relative_to(data_dir)),
                title=file_path.stem,
            )


def collect_bis_speeches(data_dir: Path, records: list[dict], seen_fingerprints: set[str]) -> None:
    for year in range(2020, 2026):
        csv_path = data_dir / f"speeches_{year}" / f"speeches_{year}.csv"
        if not csv_path.exists():
            continue

        speech_df = pd.read_csv(csv_path, dtype=str).fillna("")
        for row in speech_df.to_dict(orient="records"):
            bank = infer_bis_bank(row.get("description", ""), row.get("title", ""), row.get("author", ""))
            if not bank:
                continue
            date_text = parse_source_date(row.get("date", ""))
            add_corpus_record(
                records,
                seen_fingerprints,
                bank=bank,
                date_text=date_text,
                doc_type="speech",
                speaker=str(row.get("author", "")).strip() or corpus_default_speaker(bank, "speech"),
                text=row.get("text", ""),
                source_family="bis_speeches_csv",
                source_path=str(csv_path.relative_to(data_dir)),
                title=row.get("title", ""),
            )


def write_raw_docs(base_dir: Path, records: list[dict]) -> pd.DataFrame:
    raw_docs_dir = base_dir / "raw_docs"
    metadata_path = base_dir / "raw_docs_metadata.csv"

    raw_docs_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    for record in sorted(records, key=lambda item: (item["central_bank"], item["date"], item["doc_type"], item["doc_id"])):
        relative_file = Path(record["central_bank"].lower()) / record["doc_type"] / f"{record['doc_id']}.txt"
        text_path = raw_docs_dir / relative_file
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(record["text"], encoding="utf-8")

        metadata_rows.append(
            {
                "doc_id": record["doc_id"],
                "central_bank": record["central_bank"],
                "region": record["region"],
                "date": record["date"],
                "tier": record["tier"],
                "doc_type": record["doc_type"],
                "speaker": record["speaker"],
                "text_file": str(relative_file).replace("\\", "/"),
                "source_family": record["source_family"],
                "source_path": record["source_path"],
            }
        )

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    return metadata_df


def prepare_raw_docs_corpus(base_dir: Path) -> pd.DataFrame:
    data_dir = base_dir / "Data"
    if not data_dir.exists():
        return pd.DataFrame()

    records: list[dict] = []
    seen_fingerprints: set[str] = set()

    collect_research_documents(data_dir, records, seen_fingerprints)
    collect_bis_speeches(data_dir, records, seen_fingerprints)

    return write_raw_docs(base_dir, records)


def corpus_needs_refresh(base_dir: Path) -> bool:
    data_dir = base_dir / "Data"
    metadata_path = base_dir / "raw_docs_metadata.csv"
    raw_docs_dir = base_dir / "raw_docs"

    if not data_dir.exists():
        return False
    if not metadata_path.exists() or not raw_docs_dir.exists():
        return True

    metadata_mtime = metadata_path.stat().st_mtime
    source_files = [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".csv", ".xlsx"}
    ]
    if not source_files:
        return False

    latest_source_mtime = max(path.stat().st_mtime for path in source_files)
    return latest_source_mtime > metadata_mtime


def parse_tier(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def normalize_key(value: str) -> str:
    return str(value).strip().lower()


def looks_like_text(value: str) -> bool:
    value = str(value).strip()
    if not value:
        return False
    if parse_tier(value) is not None:
        return False
    return len(value) >= 20 and " " in value


def standardize_central_bank(value: str) -> str:
    cleaned = normalize_key(value)
    return CENTRAL_BANK_ALIASES.get(cleaned, str(value).strip())


def standardize_region(value: str, central_bank: str = "") -> str:
    cleaned = normalize_key(value)
    if cleaned in REGION_ALIASES:
        return REGION_ALIASES[cleaned]

    bank = standardize_central_bank(central_bank)
    fallback_regions = {
        "Fed": "US",
        "ECB": "Eurozone",
        "BOE": "UK",
        "BOJ": "Japan",
        "BOC": "Canada",
    }
    return fallback_regions.get(bank, str(value).strip())


def standardize_doc_type(value: str) -> str:
    cleaned = normalize_key(value)
    if not cleaned:
        return "unknown"
    if cleaned in DOC_TYPE_ALIASES:
        return DOC_TYPE_ALIASES[cleaned]
    return cleaned.replace(" ", "_")


def standardize_date(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text
    return parsed.strftime("%Y-%m-%d")


def normalize_record(record: dict) -> dict:
    normalized = {key: str(value).strip() for key, value in record.items()}

    doc_id = normalized.get("doc_id", "")
    central_bank = normalized.get("central_bank", "") or infer_central_bank_from_text(doc_id)
    region = normalized.get("region", "")
    raw_date = normalized.get("date", "")
    raw_tier = normalized.get("tier", "")
    doc_type = normalized.get("doc_type", "")
    speaker = normalized.get("speaker", "")
    text = normalized.get("text", "")

    tier = parse_tier(raw_tier)
    date = raw_date

    if tier is None and parse_tier(raw_date) is not None and looks_like_text(raw_tier):
        tier = parse_tier(raw_date)
        date = ""
        text = raw_tier

    if not text:
        for candidate in [speaker, doc_type, raw_tier, raw_date]:
            if looks_like_text(candidate):
                text = candidate
                break

    if not doc_id:
        fallback_bank = standardize_central_bank(central_bank) or "unknown_bank"
        fallback_date = standardize_date(date) or "undated"
        doc_id = f"{fallback_bank.lower()}_{fallback_date}_{standardize_doc_type(doc_type)}"

    standardized_bank = standardize_central_bank(central_bank)
    standardized_doc_type = standardize_doc_type(doc_type)
    standardized_speaker = speaker.strip() or DEFAULT_SPEAKERS.get(standardized_bank, {}).get(
        standardized_doc_type,
        standardized_bank,
    )

    return {
        "doc_id": doc_id,
        "central_bank": standardized_bank,
        "region": standardize_region(region, central_bank),
        "date": standardize_date(date),
        "tier": tier if tier is not None else 2,
        "doc_type": standardized_doc_type,
        "speaker": standardized_speaker,
        "text": clean_text(text),
    }


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = "" if column != "tier" else 2

    df = df[CANONICAL_COLUMNS].copy()
    df["doc_id"] = df["doc_id"].astype(str).str.strip()
    df["central_bank"] = df["central_bank"].astype(str).str.strip()
    df["region"] = df["region"].astype(str).str.strip()
    df["date"] = df["date"].astype(str).str.strip()
    df["doc_type"] = df["doc_type"].astype(str).str.strip()
    df["speaker"] = df["speaker"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).map(clean_text)
    df["tier"] = df["tier"].apply(lambda value: parse_tier(value) or 2)
    df = df[
        (df["text"] != "")
        & (df["doc_id"] != "")
        & (df["central_bank"] != "")
        & (df["region"] != "")
    ].copy()
    return df.reset_index(drop=True)


def is_snippet_document(text: str) -> bool:
    cleaned = clean_text(text)
    sentence_count = len(split_sentences(cleaned))
    return len(cleaned) < 500 and sentence_count <= 4


def infer_central_bank_from_text(value: str) -> str:
    text = normalize_key(value)
    if "fed" in text or "federal reserve" in text or "fomc" in text:
        return "Fed"
    if "ecb" in text or "european central bank" in text:
        return "ECB"
    if "boe" in text or "bank of england" in text:
        return "BOE"
    if "boj" in text or "bank of japan" in text:
        return "BOJ"
    if "boc" in text or "bank of canada" in text:
        return "BOC"
    return ""


def infer_central_bank_from_content(text: str) -> str:
    preview = normalize_key(text[:1500])
    bank_markers = {
        "Fed": ["federal reserve", "fomc", "chair powell", "jerome powell"],
        "ECB": ["european central bank", "ecb", "christine lagarde", "governing council"],
        "BOE": ["bank of england", "monetary policy committee", "andrew bailey"],
        "BOJ": ["bank of japan", "ueda kazuo", "summary of opinions at the monetary policy meeting"],
        "BOC": ["bank of canada", "governing council", "tiff macklem", "monetary policy report"],
    }

    scores = {}
    for bank, markers in bank_markers.items():
        scores[bank] = sum(marker in preview for marker in markers)

    best_bank = max(scores, key=scores.get)
    return best_bank if scores[best_bank] > 0 else ""


def infer_doc_type_from_content(text: str) -> str:
    preview = normalize_key(text[:2000])
    if "summary of opinions" in preview or "minutes of the monetary policy meeting" in preview:
        return "minutes"
    if "press conference" in preview or "questions and answers" in preview:
        return "press_conference"
    if "speech" in preview or "remarks" in preview:
        return "speech"
    if "statement on monetary policy" in preview or "policy statement" in preview or "monetary policy report" in preview:
        return "policy_statement"
    return ""


def infer_date_from_content(text: str) -> str:
    preview = text[:1500]
    patterns = [
        r"([A-Z][a-z]+ \d{1,2}, \d{4})",
        r"(\d{4}-\d{2}-\d{2})",
    ]

    for pattern in patterns:
        match = re.search(pattern, preview)
        if match:
            return standardize_date(match.group(1))
    return ""


def sanitize_stem(stem: str) -> str:
    cleaned = stem.strip()
    cleaned = re.sub(r"\s*-\s*copy(?:\s*\(\d+\))?$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.rstrip("_ ")
    return cleaned


def infer_metadata_from_filename(text_path: Path) -> dict:
    stem = sanitize_stem(text_path.stem)
    stem_parts = stem.split("_")

    central_bank = infer_central_bank_from_text(stem_parts[0] if stem_parts else stem)
    inferred_date = ""
    if len(stem_parts) >= 4 and all(part.isdigit() for part in stem_parts[1:4]):
        inferred_date = f"{stem_parts[1]}-{stem_parts[2]}-{stem_parts[3]}"

    doc_type_parts = stem_parts[4:] if inferred_date else stem_parts[1:]
    doc_type = standardize_doc_type("_".join(doc_type_parts)) if doc_type_parts else "unknown"

    tier_map = {
        "policy_statement": 1,
        "statement": 1,
        "press_conference": 1,
        "minutes": 2,
        "speech": 1,
        "testimony": 1,
        "unknown": 2,
    }

    speaker = DEFAULT_SPEAKERS.get(central_bank, {}).get(doc_type, central_bank)

    return {
        "doc_id": stem.replace(" ", "_"),
        "central_bank": central_bank,
        "region": standardize_region("", central_bank),
        "date": inferred_date,
        "tier": tier_map.get(doc_type, 2),
        "doc_type": doc_type,
        "speaker": speaker,
        "text": "",
    }


def apply_content_overrides(record: dict) -> dict:
    text = str(record.get("text", ""))
    if not text:
        return record

    original_bank = standardize_central_bank(record.get("central_bank", ""))
    original_doc_type = standardize_doc_type(record.get("doc_type", ""))
    content_bank = infer_central_bank_from_content(text)
    if content_bank:
        record["central_bank"] = content_bank
        record["region"] = standardize_region("", content_bank)

    content_doc_type = infer_doc_type_from_content(text)
    if content_doc_type:
        record["doc_type"] = content_doc_type

    content_date = infer_date_from_content(text)
    if content_date and (not str(record.get("date", "")).strip()):
        record["date"] = content_date

    bank = standardize_central_bank(record.get("central_bank", ""))
    doc_type = standardize_doc_type(record.get("doc_type", ""))
    if (
        not str(record.get("speaker", "")).strip()
        or bank != original_bank
        or doc_type != original_doc_type
    ):
        record["speaker"] = DEFAULT_SPEAKERS.get(bank, {}).get(doc_type, bank)

    tier_map = {
        "policy_statement": 1,
        "press_conference": 1,
        "speech": 1,
        "minutes": 2,
        "testimony": 1,
        "unknown": 2,
    }
    record["tier"] = tier_map.get(standardize_doc_type(record.get("doc_type", "")), record.get("tier", 2))

    if content_bank or content_doc_type or content_date:
        doc_date = standardize_date(record.get("date", "")) or "undated"
        doc_type = standardize_doc_type(record.get("doc_type", ""))
        record["doc_id"] = f"{standardize_central_bank(record.get('central_bank', '')).lower()}_{doc_date}_{doc_type}"

    return record


def dataframe_from_records(records: Iterable[dict]) -> pd.DataFrame:
    normalized_records = [normalize_record(record) for record in records]
    df = pd.DataFrame(normalized_records)
    return enforce_schema(df)


def ingest_csv(csv_path: Path) -> pd.DataFrame:
    csv_df = pd.read_csv(csv_path, dtype=str).fillna("")
    if csv_df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    normalized_df = dataframe_from_records(csv_df.to_dict(orient="records"))
    normalized_df["row_order"] = range(len(normalized_df))

    group_cols = [column for column in CANONICAL_COLUMNS if column != "text"]
    canonical_df = (
        normalized_df.sort_values("row_order")
        .groupby(group_cols, dropna=False)
        .agg(text=("text", lambda values: " ".join(str(v).strip() for v in values if str(v).strip())))
        .reset_index()
    )

    return enforce_schema(canonical_df)


def ingest_metadata(metadata_path: Path, raw_docs_dir: Path) -> pd.DataFrame:
    metadata_df = pd.read_csv(metadata_path, dtype=str).fillna("")
    if metadata_df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    records = []
    for row in metadata_df.to_dict(orient="records"):
        text_file = str(row.get("text_file", "")).strip()
        if not text_file:
            continue

        text_path = raw_docs_dir / text_file
        if not text_path.exists():
            continue

        record = dict(row)
        record["text"] = text_path.read_text(encoding="utf-8")
        record = apply_content_overrides(record)
        records.append(record)

    return dataframe_from_records(records)


def ingest_raw_docs_dir(raw_docs_dir: Path) -> pd.DataFrame:
    records = []
    for text_path in sorted(raw_docs_dir.rglob("*.txt")):
        record = infer_metadata_from_filename(text_path)
        record["text"] = text_path.read_text(encoding="utf-8")
        record = apply_content_overrides(record)
        records.append(record)

    return dataframe_from_records(records)


def ingest_text_file(text_path: Path, metadata: dict | None = None) -> pd.DataFrame:
    metadata = metadata or {}
    record = dict(metadata)
    record["text"] = text_path.read_text(encoding="utf-8")
    if not record.get("doc_id"):
        record["doc_id"] = text_path.stem
    return dataframe_from_records([record])


def ingest_direct_text(text: str, metadata: dict | None = None) -> pd.DataFrame:
    metadata = metadata or {}
    record = dict(metadata)
    record["text"] = text
    if not record.get("doc_id"):
        bank = standardize_central_bank(record.get("central_bank", "")) or "manual"
        date = standardize_date(record.get("date", "")) or "undated"
        record["doc_id"] = f"{bank.lower()}_{date}_manual_input"
    return dataframe_from_records([record])


def load_all_inputs(base_dir: Path) -> pd.DataFrame:
    raw_docs_dir = base_dir / "raw_docs"
    metadata_path = base_dir / "raw_docs_metadata.csv"
    canonical_df = pd.DataFrame(columns=CANONICAL_COLUMNS)

    if raw_docs_dir.exists() and metadata_path.exists():
        metadata_df = ingest_metadata(metadata_path, raw_docs_dir)
        if not metadata_df.empty:
            canonical_df = metadata_df
    elif raw_docs_dir.exists():
        raw_docs_df = ingest_raw_docs_dir(raw_docs_dir)
        if not raw_docs_df.empty:
            canonical_df = raw_docs_df

    if canonical_df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    canonical_df = canonical_df.drop_duplicates(subset=["doc_id"], keep="last").reset_index(drop=True)
    canonical_df = enforce_schema(canonical_df)
    canonical_df = canonical_df[~canonical_df["text"].astype(str).map(is_snippet_document)].reset_index(drop=True)
    return canonical_df

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
CLEANED_DOCS_DIR = OUTPUT_DIR / "cleaned_docs"


def safe_text_filename(doc_id: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(doc_id))
    return f"{cleaned}.txt"


def build_document_sentence_map(documents: pd.DataFrame) -> Dict[str, List[str]]:
    sentence_map: Dict[str, List[str]] = {}
    for document in documents.to_dict(orient="records"):
        sentence_map[document["doc_id"]] = split_sentences(document["text"])
    return sentence_map


def build_cleaned_documents(documents: pd.DataFrame, sentence_map: Dict[str, List[str]] | None = None) -> pd.DataFrame:
    CLEANED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    sentence_map = sentence_map or {}

    for document in documents.to_dict(orient="records"):
        cleaned_sentences = sentence_map.get(document["doc_id"]) or split_sentences(document["text"])
        cleaned = " ".join(cleaned_sentences)
        cleaned_file_name = safe_text_filename(document["doc_id"])
        cleaned_file_path = CLEANED_DOCS_DIR / cleaned_file_name
        cleaned_file_path.write_text(cleaned, encoding="utf-8")
        rows.append(
            {
                "doc_id": document["doc_id"],
                "central_bank": document["central_bank"],
                "region": document["region"],
                "date": document["date"],
                "tier": document["tier"],
                "doc_type": document["doc_type"],
                "speaker": document["speaker"],
                "cleaned_text_file": str(Path("cleaned_docs") / cleaned_file_name).replace("\\", "/"),
                "character_count": len(cleaned),
                "sentence_count": len(cleaned_sentences),
            }
        )

    return pd.DataFrame(rows)


def build_scored_sentences(documents: pd.DataFrame, sentence_map: Dict[str, List[str]] | None = None) -> pd.DataFrame:
    rows = []
    max_document_date = pd.to_datetime(documents["date"], errors="coerce").max()
    sentence_map = sentence_map or {}

    for document in documents.to_dict(orient="records"):
        sentences = sentence_map.get(document["doc_id"]) or split_sentences(document["text"])
        current_tier_weight = tier_weight(document["tier"])
        current_recency_weight = recency_weight(document["date"], max_document_date)
        last_categories: list[str] = []

        for sentence in sentences:
            if document["doc_type"] == "press_conference" and is_question_like_sentence(sentence):
                continue
            categories = detect_categories(sentence)
            if not categories and last_categories:
                referential_openers = ("this ", "these ", "it ", "they ", "such ", "those ")
                if sentence.lower().startswith(referential_openers):
                    categories = last_categories.copy()
            sentiment = sentiment_score(sentence, categories) if categories else 0
            magnitude = magnitude_score(sentence) if sentiment != 0 else 0.0
            category_adjustment = category_adjustment_weight(categories) if sentiment != 0 else 0.0
            weighted_score = sentiment * magnitude * current_tier_weight * category_adjustment * current_recency_weight
            if categories:
                last_categories = categories.copy()
            if sentiment == 0:
                continue

            rows.append(
                {
                    "doc_id": document["doc_id"],
                    "central_bank": document["central_bank"],
                    "region": document["region"],
                    "date": document["date"],
                    "tier": document["tier"],
                    "doc_type": document["doc_type"],
                    "speaker": document["speaker"],
                    "sentence": sentence,
                    "categories": ", ".join(categories),
                    "sentiment": sentiment,
                    "magnitude": magnitude,
                    "tier_weight": current_tier_weight,
                    "recency_weight": current_recency_weight,
                    "weighted_score": weighted_score,
                }
            )

    return pd.DataFrame(rows)


def build_excerpts_dataset(documents: pd.DataFrame, sentence_map: Dict[str, List[str]] | None = None) -> pd.DataFrame:
    rows = []
    sentence_map = sentence_map or {}

    for document in documents.to_dict(orient="records"):
        sentences = sentence_map.get(document["doc_id"]) or split_sentences(document["text"])

        for index, sentence in enumerate(sentences, start=1):
            rows.append(
                {
                    "doc_id": document["doc_id"],
                    "central_bank": document["central_bank"],
                    "region": document["region"],
                    "date": document["date"],
                    "tier": document["tier"],
                    "doc_type": document["doc_type"],
                    "speaker": document["speaker"],
                    "text": sentence,
                    "excerpt_id": f"{document['doc_id']}_excerpt_{index:04d}",
                }
            )

    return pd.DataFrame(rows)


def build_document_scores(scored_sentences: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["doc_id", "central_bank", "region", "date", "tier", "doc_type", "speaker"]

    document_scores = (
        scored_sentences.groupby(group_cols, dropna=False)
        .agg(
            document_score=("weighted_score", "sum"),
            sentence_count=("sentence", "count"),
            nonzero_signal_sentences=("sentiment", lambda values: (values != 0).sum()),
            document_recency_weight=("recency_weight", "mean"),
        )
        .reset_index()
    )
    nonzero_counts = document_scores["nonzero_signal_sentences"].replace(0, pd.NA)
    document_scores["avg_sentence_signal"] = (
        document_scores["document_score"] / nonzero_counts
    ).fillna(0.0)

    return document_scores


def build_regional_scores(document_scores: pd.DataFrame) -> pd.DataFrame:
    doc_type_rows = []
    for (region, central_bank, doc_type), group in document_scores.groupby(["region", "central_bank", "doc_type"], dropna=False):
        recency_weights = group["document_recency_weight"].fillna(1.0)
        weight_sum = recency_weights.sum()
        if weight_sum == 0:
            raw_policy_score = group["document_score"].mean()
            avg_document_signal = group["avg_sentence_signal"].mean()
        else:
            raw_policy_score = (group["document_score"] * recency_weights).sum() / weight_sum
            avg_document_signal = (group["avg_sentence_signal"] * recency_weights).sum() / weight_sum

        doc_type_rows.append(
            {
                "region": region,
                "central_bank": central_bank,
                "doc_type": doc_type,
                "raw_policy_score": raw_policy_score,
                "avg_document_signal": avg_document_signal,
                "doc_type_documents": int(group["doc_id"].count()),
            }
        )

    doc_type_balanced_scores = pd.DataFrame(doc_type_rows)
    doc_type_balanced_scores["doc_type_weight"] = doc_type_balanced_scores["doc_type"].map(DOC_TYPE_REGIONAL_WEIGHTS).fillna(1.0)

    regional_rows = []
    for (region, central_bank), group in doc_type_balanced_scores.groupby(["region", "central_bank"], dropna=False):
        weight_sum = group["doc_type_weight"].sum()
        if weight_sum == 0:
            weighted_raw = group["raw_policy_score"].mean()
            weighted_avg_signal = group["avg_document_signal"].mean()
        else:
            weighted_raw = (group["raw_policy_score"] * group["doc_type_weight"]).sum() / weight_sum
            weighted_avg_signal = (group["avg_document_signal"] * group["doc_type_weight"]).sum() / weight_sum

        regional_rows.append(
            {
                "region": region,
                "central_bank": central_bank,
                "raw_policy_score": weighted_raw,
                "avg_document_signal": weighted_avg_signal,
                "documents_scored": int(group["doc_type_documents"].sum()),
                "doc_types_scored": int(group["doc_type"].nunique()),
            }
        )

    regional_scores = pd.DataFrame(regional_rows)

    regional_scores["standardized_policy_score"] = (
        regional_scores["avg_document_signal"] / STANDARDIZATION_DENOMINATOR
    ).clip(-1.0, 1.0)

    regional_scores["policy_score"] = regional_scores["standardized_policy_score"].round(3)
    regional_scores["policy_score_100"] = (regional_scores["standardized_policy_score"] * 100).round(1)
    regional_scores["raw_policy_score"] = regional_scores["raw_policy_score"].round(3)
    regional_scores["avg_document_signal"] = regional_scores["avg_document_signal"].round(3)

    regional_scores = regional_scores[
        [
            "region",
            "central_bank",
            "policy_score",
            "policy_score_100",
            "avg_document_signal",
            "raw_policy_score",
            "documents_scored",
            "doc_types_scored",
        ]
    ].sort_values("policy_score", ascending=False)

    return regional_scores


def build_category_scores(scored_sentences: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    category_rows = []

    for row in scored_sentences.to_dict(orient="records"):
        category_text = str(row.get("categories", "")).strip()
        if not category_text:
            continue

        for category in [item.strip() for item in category_text.split(",") if item.strip()]:
            category_rows.append(
                {
                    "doc_id": row.get("doc_id"),
                    "central_bank": row.get("central_bank"),
                    "region": row.get("region"),
                    "date": row.get("date"),
                    "tier": row.get("tier"),
                    "doc_type": row.get("doc_type"),
                    "speaker": row.get("speaker"),
                    "category": category,
                    "category_score": row.get("weighted_score", 0.0),
                }
            )

    category_df = pd.DataFrame(category_rows)
    if category_df.empty:
        empty_document_scores = pd.DataFrame(
            columns=["doc_id", "central_bank", "region", "date", "tier", "doc_type", "speaker", "category", "category_score"]
        )
        empty_regional_scores = pd.DataFrame(
            columns=["region", "central_bank", "category", "policy_category_score"]
        )
        return empty_document_scores, empty_regional_scores

    document_category_scores = (
        category_df.groupby(
            ["doc_id", "central_bank", "region", "date", "tier", "doc_type", "speaker", "category"],
            dropna=False,
        )
        .agg(category_score=("category_score", "sum"))
        .reset_index()
    )

    doc_type_category_scores = (
        document_category_scores.groupby(["region", "central_bank", "doc_type", "category"], dropna=False)
        .agg(policy_category_score=("category_score", "mean"))
        .reset_index()
    )

    regional_category_scores = (
        doc_type_category_scores.groupby(["region", "central_bank", "category"], dropna=False)
        .agg(policy_category_score=("policy_category_score", "mean"))
        .reset_index()
        .sort_values(["region", "category"])
    )

    return document_category_scores, regional_category_scores


def build_sample_scored_excerpts(scored_sentences: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    if scored_sentences.empty:
        return scored_sentences.copy()

    sample_df = scored_sentences.copy()
    sample_df["abs_weighted_score"] = sample_df["weighted_score"].abs()
    sample_df["category_count"] = sample_df["categories"].astype(str).str.count(",") + 1

    selections = []
    for central_bank in sorted(sample_df["central_bank"].dropna().unique()):
        bank_df = sample_df[sample_df["central_bank"] == central_bank].copy()
        if bank_df.empty:
            continue

        negative = bank_df[bank_df["weighted_score"] < 0].sort_values(
            ["abs_weighted_score", "category_count"],
            ascending=[False, False],
        ).head(2)
        positive = bank_df[bank_df["weighted_score"] > 0].sort_values(
            ["abs_weighted_score", "category_count"],
            ascending=[False, False],
        ).head(2)
        moderate = bank_df[bank_df["magnitude"] == 1.0].sort_values(
            ["abs_weighted_score", "category_count"],
            ascending=[False, False],
        ).head(2)

        selections.extend([negative, positive, moderate])

    if selections:
        sample_df = pd.concat(selections, ignore_index=True)
        sample_df = sample_df.drop_duplicates(subset=["doc_id", "sentence"]).copy()
    else:
        sample_df = sample_df.sort_values(["abs_weighted_score"], ascending=False).head(top_n)

    sample_df = sample_df.sort_values(
        ["central_bank", "weighted_score", "abs_weighted_score"],
        ascending=[True, True, False],
    ).head(top_n)

    return sample_df.drop(columns=["abs_weighted_score", "category_count"], errors="ignore")


def export_outputs(
    cleaned_documents: pd.DataFrame,
    excerpts_dataset: pd.DataFrame,
    scored_sentences: pd.DataFrame,
    regional_scores: pd.DataFrame,
    sample_scored_excerpts: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    export_kwargs = {"index": False, "encoding": "utf-8-sig", "quoting": csv.QUOTE_ALL}
    cleaned_documents.to_csv(OUTPUT_DIR / "cleaned_documents.csv", **export_kwargs)
    excerpts_dataset.to_csv(OUTPUT_DIR / "all_excerpts_dataset.csv", **export_kwargs)
    scored_sentences.to_csv(OUTPUT_DIR / "scored_sentences.csv", **export_kwargs)
    regional_scores.to_csv(OUTPUT_DIR / "regional_policy_scores.csv", **export_kwargs)
    sample_scored_excerpts.to_csv(OUTPUT_DIR / "sample_scored_excerpts.csv", **export_kwargs)


def main() -> None:
    if corpus_needs_refresh(BASE_DIR):
        prepare_raw_docs_corpus(BASE_DIR)
    documents = load_all_inputs(BASE_DIR)
    sentence_map = build_document_sentence_map(documents)
    cleaned_documents = build_cleaned_documents(documents, sentence_map)
    excerpts_dataset = build_excerpts_dataset(documents, sentence_map)
    scored_sentences = build_scored_sentences(documents, sentence_map)
    document_scores = build_document_scores(scored_sentences)
    regional_scores = build_regional_scores(document_scores)
    sample_scored_excerpts = build_sample_scored_excerpts(scored_sentences)

    export_outputs(
        cleaned_documents,
        excerpts_dataset,
        scored_sentences,
        regional_scores,
        sample_scored_excerpts,
    )

    print("Finished scoring policy documents.")
    print(f"Loaded documents: {len(documents)}")
    print(f"Scored sentences: {len(scored_sentences)}")
    print("\nRegional policy scores:")
    print(regional_scores.to_string(index=False))


if __name__ == "__main__":
    main()
