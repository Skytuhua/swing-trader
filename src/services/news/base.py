"""Abstract base classes and data models for the news analysis engine."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Raw data structures — provider output before NLP enrichment
# ---------------------------------------------------------------------------


@dataclass
class RawNewsItem:
    """Minimal news item returned directly from a provider adapter."""

    source: str
    """Provider name, e.g. 'finnhub', 'newsapi'."""

    headline: str
    """Article headline / title."""

    url: str
    """Canonical article URL."""

    published_at: datetime
    """Publication timestamp (UTC)."""

    summary: str = ""
    """Article body snippet or lead paragraph."""

    tickers: list[str] = field(default_factory=list)
    """Related ticker symbols extracted or provided by the source."""

    sectors: list[str] = field(default_factory=list)
    """Sector tags, e.g. 'technology', 'healthcare'."""

    category: str = "general"
    """News category: general | earnings | analyst | macro | merger | crypto | forex."""

    image_url: str = ""
    """Optional thumbnail URL."""

    source_id: str = ""
    """Provider-specific article identifier."""

    @property
    def fingerprint(self) -> str:
        """Stable content fingerprint used for deduplication."""
        content = f"{self.headline}|{self.url}"
        return hashlib.sha256(content.encode()).hexdigest()

    @property
    def text(self) -> str:
        """Combined headline + summary for NLP input."""
        return f"{self.headline}. {self.summary}".strip() if self.summary else self.headline


# ---------------------------------------------------------------------------
# Enriched item — after FinBERT analysis
# ---------------------------------------------------------------------------


@dataclass
class NewsAnalysis:
    """Output of the FinBERT analysis step for a single piece of text."""

    sentiment_score: float
    """Net sentiment: positive_prob − negative_prob, range [-1, +1]."""

    positive_prob: float
    negative_prob: float
    neutral_prob: float

    confidence: float
    """Max of the three probabilities — how decisive the model is."""

    @property
    def label(self) -> str:
        if self.positive_prob >= self.negative_prob and self.positive_prob >= self.neutral_prob:
            return "positive"
        if self.negative_prob >= self.positive_prob and self.negative_prob >= self.neutral_prob:
            return "negative"
        return "neutral"


@dataclass
class NewsItem:
    """Fully enriched news item stored in the database."""

    source: str
    headline: str
    url: str
    published_at: datetime
    ingested_at: datetime
    summary: str = ""
    tickers: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    category: str = "general"
    sentiment_score: float = 0.0
    positive_prob: float = 0.0
    negative_prob: float = 0.0
    neutral_prob: float = 0.0
    confidence: float = 0.0
    relevance_score: float = 1.0
    """Relevance to the specific ticker/sector being queried, 0–1."""
    impact_score: float = 0.5
    """Estimated market-impact magnitude, 0–1."""
    novelty_score: float = 1.0
    """Near-duplicate detection result; 1.0 = fully novel."""
    is_confirmed_fact: bool = False
    fingerprint: str = ""

    @classmethod
    def from_raw(cls, raw: RawNewsItem, analysis: NewsAnalysis, ingested_at: datetime) -> "NewsItem":
        return cls(
            source=raw.source,
            headline=raw.headline,
            url=raw.url,
            published_at=raw.published_at,
            ingested_at=ingested_at,
            summary=raw.summary,
            tickers=raw.tickers,
            sectors=raw.sectors,
            category=raw.category,
            sentiment_score=analysis.sentiment_score,
            positive_prob=analysis.positive_prob,
            negative_prob=analysis.negative_prob,
            neutral_prob=analysis.neutral_prob,
            confidence=analysis.confidence,
            fingerprint=raw.fingerprint,
        )


# ---------------------------------------------------------------------------
# NewsScore — output of the scoring layer
# ---------------------------------------------------------------------------


@dataclass
class NewsScore:
    """Aggregated news impact score for a single ticker over a look-back window."""

    score: float
    """Final 0–100 composite score (50 = neutral)."""

    catalyst_present: bool = False
    """True when an earnings release or analyst upgrade is detected."""

    negative_catalyst: bool = False
    """True when a clearly negative event is detected (sentiment < -0.3)."""

    conflicting_signals: bool = False
    """True when both strongly positive and strongly negative items exist."""

    item_count: int = 0
    explanation: str = ""

    weighted_sentiment: float = 0.0
    """Average time-decay-weighted sentiment score."""

    decay_weighted_count: float = 0.0
    """Effective recency-adjusted count of articles."""


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------


class NewsProvider(ABC):
    """Abstract base class that all news source adapters must implement."""

    @abstractmethod
    async def fetch_general_news(self, category: str = "general") -> list[RawNewsItem]:
        """Fetch broad market/macro news.

        Args:
            category: Provider-specific category filter.
        Returns:
            List of raw news items, newest first.
        """
        ...

    @abstractmethod
    async def fetch_company_news(self, ticker: str, from_date: date) -> list[RawNewsItem]:
        """Fetch news specific to a single company/ticker.

        Args:
            ticker: Equity symbol, e.g. 'AAPL'.
            from_date: Inclusive start date (UTC).
        Returns:
            List of raw news items.
        """
        ...

    @abstractmethod
    async def fetch_sector_news(self, sector: str) -> list[RawNewsItem]:
        """Fetch news related to a broad market sector.

        Args:
            sector: Sector name, e.g. 'technology', 'healthcare'.
        Returns:
            List of raw news items.
        """
        ...

    async def health_check(self) -> bool:
        """Return True if the provider endpoint is reachable."""
        return True
