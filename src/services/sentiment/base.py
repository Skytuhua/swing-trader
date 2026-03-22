"""Abstract base classes and data models for the social sentiment engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import structlog

from src.core.enums import SentimentPhase

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data model: individual mention
# ---------------------------------------------------------------------------


class MentionSource(str, Enum):
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    TWITTER = "twitter"
    OTHER = "other"


@dataclass
class SocialMention:
    """A single social media mention of a ticker.

    Attributes
    ----------
    source: Provider name (reddit, stocktwits, …).
    ticker: The equity symbol this mention pertains to.
    text: The raw post/comment text.
    author: Author handle or anonymised ID.
    timestamp: When this mention was published (UTC).
    polarity: Sentiment polarity in [-1, +1].
              -1 = strongly bearish, 0 = neutral, +1 = strongly bullish.
    polarity_method: How polarity was determined (keyword | finbert | label).
    bullish_label: True if the source explicitly labelled the post bullish.
    bearish_label: True if the source explicitly labelled the post bearish.
    upvotes: Number of upvotes/likes (0 if not available).
    post_id: Provider-specific post identifier.
    url: Link to the original post.
    is_spam: True if the mention was flagged as spam/noise.
    subreddit: Subreddit name if source = reddit (empty string otherwise).
    """

    source: MentionSource
    ticker: str
    text: str
    author: str
    timestamp: datetime
    polarity: float = 0.0

    polarity_method: str = "keyword"
    bullish_label: Optional[bool] = None
    bearish_label: Optional[bool] = None
    upvotes: int = 0
    post_id: str = ""
    url: str = ""
    is_spam: bool = False
    subreddit: str = ""

    def __post_init__(self) -> None:
        # Clamp polarity to valid range
        self.polarity = max(-1.0, min(1.0, self.polarity))
        # Normalise ticker
        self.ticker = self.ticker.upper()

    @property
    def is_bullish(self) -> bool:
        """True when the mention carries a net positive signal."""
        if self.bullish_label is True:
            return True
        if self.bearish_label is True:
            return False
        return self.polarity > 0.1

    @property
    def is_bearish(self) -> bool:
        """True when the mention carries a net negative signal."""
        if self.bearish_label is True:
            return True
        if self.bullish_label is True:
            return False
        return self.polarity < -0.1

    @property
    def weight(self) -> float:
        """Influence weight — higher-karma posts count more, capped at 10×."""
        if self.upvotes <= 0:
            return 1.0
        return min(10.0, 1.0 + 0.1 * self.upvotes ** 0.5)


# ---------------------------------------------------------------------------
# Aggregated result
# ---------------------------------------------------------------------------


@dataclass
class SentimentProfile:
    """Aggregated sentiment state for a single ticker.

    Produced by SentimentAggregator and consumed by SentimentScorer.
    """

    ticker: str
    score: float
    """Final 0-100 composite score (50 = neutral)."""

    polarity: float = 0.0
    """Weighted average polarity across all clean mentions [-1, +1]."""

    intensity: float = 0.0
    """Z-score of mention volume vs baseline (how unusual the volume is)."""

    phase: SentimentPhase = SentimentPhase.EARLY
    """Detected sentiment lifecycle phase."""

    crowding_risk: float = 0.0
    """0-100 score; high values indicate crowded/hyped sentiment."""

    mention_count: int = 0
    """Total clean (non-spam) mention count."""

    bullish_pct: float = 0.0
    """Fraction of clean mentions that are bullish (0–1)."""

    bearish_pct: float = 0.0
    """Fraction of clean mentions that are bearish (0–1)."""

    sources: list[str] = field(default_factory=list)
    """Data sources that contributed mentions."""

    data_quality: str = "good"
    """Data quality flag: good | degraded | unavailable."""

    explanation: str = ""


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------


class SentimentProvider(ABC):
    """Abstract base class that all social sentiment adapters must implement."""

    @abstractmethod
    async def fetch_mentions(self, ticker: str, since: datetime) -> list[SocialMention]:
        """Fetch social mentions of ``ticker`` since ``since``.

        Args:
            ticker: Equity symbol, e.g. 'AAPL'.
            since: Earliest UTC timestamp to include.

        Returns:
            List of SocialMention objects, newest first.
        """
        ...

    async def health_check(self) -> bool:
        """Return True if the provider endpoint is reachable."""
        return True
