from __future__ import annotations

import logging
import math
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

from common import LEXICONS_DIR, clean_text_basic, count_keyword_hits, word_count

LOGGER = logging.getLogger(__name__)
LM_SOURCE_PAGE = "https://sraf.nd.edu/loughranmcdonald-master-dictionary/"


def split_sentences(text: str) -> list[str]:
    text = clean_text_basic(text)
    if not text:
        return []
    candidates = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [segment.strip() for segment in candidates if segment.strip()]


@dataclass(slots=True)
class SentenceScore:
    label: str
    score: float
    signed_score: float
    weight: int
    text: str


class FinBERTScorer:
    def __init__(self, model_name: str = "ProsusAI/finbert", cache_dir: str | None = None) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        cache_path = cache_dir or str(LEXICONS_DIR / "hf_cache")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_path)
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=512,
            top_k=None,
        )

    def _chunk_sentence(self, sentence: str) -> list[str]:
        tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
        if len(tokens) <= 510:
            return [sentence]

        chunks: list[str] = []
        for idx in range(0, len(tokens), 510):
            chunk_tokens = tokens[idx : idx + 510]
            chunks.append(self.tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        return chunks

    def score_sentences(self, sentences: Iterable[str]) -> list[SentenceScore]:
        scores: list[SentenceScore] = []
        for sentence in sentences:
            for chunk in self._chunk_sentence(sentence):
                labels = self.pipeline(chunk)[0]
                label_scores = {item["label"].lower(): float(item["score"]) for item in labels}
                pos = label_scores.get("positive", 0.0)
                neg = label_scores.get("negative", 0.0)
                neu = label_scores.get("neutral", 0.0)
                dominant = max(label_scores, key=label_scores.get)
                signed = pos - neg
                scores.append(
                    SentenceScore(
                        label=dominant,
                        score=max(pos, neg, neu),
                        signed_score=signed,
                        weight=max(word_count(chunk), 1),
                        text=chunk,
                    )
                )
        return scores

    def summarize(self, text: str) -> dict[str, float]:
        sentence_scores = self.score_sentences(split_sentences(text))
        if not sentence_scores:
            return {
                "sentiment_score": 0.0,
                "sentiment_std": 0.0,
                "sentiment_positive_pct": 0.0,
                "sentiment_negative_pct": 0.0,
                "sentiment_neutral_pct": 0.0,
            }

        total_weight = sum(item.weight for item in sentence_scores) or 1
        weighted_scores = [item.signed_score * item.weight for item in sentence_scores]
        weighted_mean = sum(weighted_scores) / total_weight
        weighted_variance = sum(item.weight * ((item.signed_score - weighted_mean) ** 2) for item in sentence_scores) / total_weight

        label_weights = {"positive": 0, "negative": 0, "neutral": 0}
        for item in sentence_scores:
            label_weights[item.label] = label_weights.get(item.label, 0) + item.weight

        return {
            "sentiment_score": weighted_mean,
            "sentiment_std": math.sqrt(weighted_variance),
            "sentiment_positive_pct": label_weights.get("positive", 0) / total_weight,
            "sentiment_negative_pct": label_weights.get("negative", 0) / total_weight,
            "sentiment_neutral_pct": label_weights.get("neutral", 0) / total_weight,
        }


class LoughranMcDonaldLexicon:
    CATEGORY_COLUMNS = {
        "negative_word_ratio": "Negative",
        "positive_word_ratio": "Positive",
        "uncertainty_score": "Uncertainty",
        "litigious_score": "Litigious",
        "constraining_score": "Constraining",
        "forward_looking_score": "Strong_Modal",
    }

    def __init__(self, source_page: str = LM_SOURCE_PAGE, lexicon_path: Path | None = None) -> None:
        self.source_page = source_page
        self.lexicon_path = lexicon_path or self._ensure_lexicon()
        self.frame = self._load_frame(self.lexicon_path)
        self.category_sets = self._category_sets(self.frame)

    def _ensure_lexicon(self) -> Path:
        existing = sorted(LEXICONS_DIR.glob("LoughranMcDonald_MasterDictionary_*"))
        if existing:
            return existing[0]

        LOGGER.info("Downloading Loughran-McDonald lexicon from %s", self.source_page)
        response = requests.get(self.source_page, timeout=60)
        response.raise_for_status()
        page_text = response.text
        soup = BeautifulSoup(page_text, "lxml")
        href = None
        link_text = ""
        for anchor in soup.select("a[href]"):
            candidate = anchor["href"]
            lower = candidate.lower()
            text = anchor.get_text(" ", strip=True)
            if "masterdictionary" in lower and lower.endswith((".csv", ".xlsx", ".xls")):
                href = candidate
                link_text = text
                break
            if "masterdictionary" in lower or "csv format" in text.lower() or "xlsx format" in text.lower():
                href = candidate
                link_text = text
                break
        if not href:
            href, link_text = self._fallback_find_download_link(page_text)
        if not href:
            raise RuntimeError("Could not locate a downloadable Loughran-McDonald dictionary file.")
        if href.startswith("/"):
            href = f"https://sraf.nd.edu{href}"

        href = self._normalize_download_url(href, link_text)

        target = LEXICONS_DIR / Path(href).name
        download = requests.get(href, timeout=120)
        download.raise_for_status()
        target.write_bytes(download.content)
        return target

    def _fallback_find_download_link(self, page_text: str) -> tuple[str | None, str]:
        patterns = [
            r'https://docs\.google\.com/spreadsheets/d/[^"\']+',
            r'https://drive\.google\.com/[^"\']+',
        ]
        for pattern in patterns:
            match = re.search(pattern, page_text, flags=re.IGNORECASE)
            if match:
                return match.group(0), "xlsx"
        return None, ""

    def _normalize_download_url(self, href: str, link_text: str) -> str:
        parsed = urllib.parse.urlparse(href)
        host = parsed.netloc.lower()
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if "docs.google.com" in host and "/spreadsheets/d/" in path:
            parts = [part for part in path.split("/") if part]
            doc_id = parts[2] if len(parts) >= 3 else None
            if doc_id:
                return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=xlsx"

        if "drive.google.com" in host and "/file/d/" in path:
            parts = [part for part in path.split("/") if part]
            file_id = parts[2] if len(parts) >= 3 else None
            extension = ".csv" if "csv" in link_text.lower() else ".xlsx"
            if file_id:
                return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"

        if "drive.google.com" in host and "id" in query:
            return f"https://drive.google.com/uc?export=download&id={query['id'][0]}&confirm=t"

        return href

    def _load_frame(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_excel(path)

    def _category_sets(self, frame: pd.DataFrame) -> dict[str, set[str]]:
        frame = frame.copy()
        frame["Word"] = frame["Word"].astype(str).str.lower()
        category_sets: dict[str, set[str]] = {}
        for output_name, source_column in self.CATEGORY_COLUMNS.items():
            if source_column not in frame.columns:
                category_sets[output_name] = set()
                continue
            active = frame.loc[frame[source_column].fillna(0).astype(float) > 0, "Word"]
            category_sets[output_name] = set(active.tolist())
        return category_sets

    def score(self, text: str) -> dict[str, float]:
        tokens = re.findall(r"\b[a-zA-Z][a-zA-Z'-]+\b", text.lower())
        total = len(tokens) or 1
        token_counts = pd.Series(tokens).value_counts().to_dict()
        result: dict[str, float] = {}
        for output_name, words in self.category_sets.items():
            hits = sum(token_counts.get(word, 0) for word in words)
            result[output_name] = hits / total
        return result


def keyword_density(text: str, keywords: Iterable[str]) -> float:
    total_words = max(word_count(text), 1)
    return count_keyword_hits(text, keywords) / total_words
