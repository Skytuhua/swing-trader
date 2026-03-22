from dataclasses import dataclass, field

@dataclass
class NewsConfig:
    providers: list = field(default_factory=list)
    api_keys: dict = field(default_factory=dict)
    finbert_model: str = "ProsusAI/finbert"

@dataclass
class SentimentConfig:
    providers: list = field(default_factory=list)
    api_keys: dict = field(default_factory=dict)
