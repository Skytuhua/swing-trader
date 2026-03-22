"""News analysis engine — providers, deduplication, NLP processing, and scoring."""

from src.services.news.base import NewsProvider, RawNewsItem, NewsItem, NewsAnalysis, NewsScore
from src.services.news.deduplicator import NewsDeduplicator
from src.services.news.processor import NewsProcessor
from src.services.news.scorer import NewsScorer

__all__ = [
    "NewsProvider",
    "RawNewsItem",
    "NewsItem",
    "NewsAnalysis",
    "NewsScore",
    "NewsDeduplicator",
    "NewsProcessor",
    "NewsScorer",
]
