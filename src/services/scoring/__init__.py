"""Scoring engine: weighted scoring, penalties, confidence, explanation."""

from .engine import ScoringEngine, ScoredCandidate
from .weights import WeightProfile, WEIGHT_PROFILES
from .explainer import ScoreExplainer

__all__ = [
    "ScoringEngine",
    "ScoredCandidate",
    "WeightProfile",
    "WEIGHT_PROFILES",
    "ScoreExplainer",
]
