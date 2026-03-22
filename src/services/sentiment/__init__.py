"""Social sentiment engine — Reddit, StockTwits, aggregation, and scoring."""

from src.services.sentiment.base import SentimentProvider, SocialMention, SentimentProfile
from src.services.sentiment.aggregator import SentimentAggregator
from src.services.sentiment.scorer import SentimentScorer

__all__ = [
    "SentimentProvider",
    "SocialMention",
    "SentimentProfile",
    "SentimentAggregator",
    "SentimentScorer",
]
