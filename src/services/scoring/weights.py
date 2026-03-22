"""
Scoring weight profiles: conservative, moderate, aggressive.

Each profile defines the relative importance of the six scoring components
plus the minimum confidence threshold and no-trade score floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class WeightProfile:
    """A named set of scoring weights and thresholds.

    All component weights must sum to 1.0 (100%).

    Attributes:
        name:                Identifier for the profile.
        technical_weight:    Weight for the technical composite score (0-1).
        news_weight:         Weight for the news/catalyst score (0-1).
        sentiment_weight:    Weight for the social sentiment score (0-1).
        liquidity_weight:    Weight for the liquidity score (0-1).
        regime_weight:       Weight for regime alignment score (0-1).
        risk_reward_weight:  Weight for the R:R quality score (0-1).
        confidence_threshold: Minimum confidence needed to trade (0-100).
        no_trade_threshold:  Minimum total score needed to trade (0-100).
        min_risk_reward:     Minimum R:R ratio required (e.g. 1.5 = 1.5:1).
    """

    name: str
    technical_weight: float
    news_weight: float
    sentiment_weight: float
    liquidity_weight: float
    regime_weight: float
    risk_reward_weight: float
    confidence_threshold: float = 60.0
    no_trade_threshold: float = 55.0
    min_risk_reward: float = 1.5

    # Class-level registry for look-up by name
    _registry: ClassVar[dict[str, "WeightProfile"]] = {}

    def __post_init__(self) -> None:
        total = (
            self.technical_weight
            + self.news_weight
            + self.sentiment_weight
            + self.liquidity_weight
            + self.regime_weight
            + self.risk_reward_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"WeightProfile '{self.name}' weights must sum to 1.0, got {total:.4f}."
            )
        WeightProfile._registry[self.name] = self

    @classmethod
    def get(cls, name: str) -> "WeightProfile":
        """Retrieve a registered profile by name."""
        try:
            return cls._registry[name]
        except KeyError:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Unknown weight profile '{name}'. Available: {available}"
            ) from None

    @classmethod
    def registered_names(cls) -> list[str]:
        return list(cls._registry.keys())

    def as_dict(self) -> dict[str, float]:
        """Return only the component weights as a plain dict."""
        return {
            "technical": self.technical_weight,
            "news": self.news_weight,
            "sentiment": self.sentiment_weight,
            "liquidity": self.liquidity_weight,
            "regime": self.regime_weight,
            "risk_reward": self.risk_reward_weight,
        }


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

# Conservative: emphasises regime alignment and technical quality;
# lower position in news/sentiment; high confidence threshold.
CONSERVATIVE = WeightProfile(
    name="conservative",
    technical_weight=0.35,
    news_weight=0.10,
    sentiment_weight=0.05,
    liquidity_weight=0.15,
    regime_weight=0.20,
    risk_reward_weight=0.15,
    confidence_threshold=70.0,
    no_trade_threshold=65.0,
    min_risk_reward=2.0,
)

# Moderate: balanced across all components.  This matches the defaults
# described in the spec and is the recommended starting profile.
MODERATE = WeightProfile(
    name="moderate",
    technical_weight=0.30,
    news_weight=0.15,
    sentiment_weight=0.10,
    liquidity_weight=0.10,
    regime_weight=0.15,
    risk_reward_weight=0.20,
    confidence_threshold=60.0,
    no_trade_threshold=55.0,
    min_risk_reward=1.5,
)

# Aggressive: heavier on news/catalyst and sentiment; more willing to
# trade with lower confidence and in mixed regimes.
AGGRESSIVE = WeightProfile(
    name="aggressive",
    technical_weight=0.25,
    news_weight=0.20,
    sentiment_weight=0.15,
    liquidity_weight=0.10,
    regime_weight=0.10,
    risk_reward_weight=0.20,
    confidence_threshold=50.0,
    no_trade_threshold=45.0,
    min_risk_reward=1.2,
)

# Convenience dict for external access
WEIGHT_PROFILES: dict[str, WeightProfile] = {
    "conservative": CONSERVATIVE,
    "moderate": MODERATE,
    "aggressive": AGGRESSIVE,
}

# Alias – default profile used by ScoringEngine when no config is provided
DEFAULT_PROFILE = MODERATE
