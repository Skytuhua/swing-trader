from enum import Enum


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class MarketRegime(str, Enum):
    FAVORABLE = "favorable"
    MIXED = "mixed"
    UNFAVORABLE = "unfavorable"


class SentimentPhase(str, Enum):
    EARLY = "early"
    RISING = "rising"
    PEAKING = "peaking"
    FADING = "fading"


class DataQuality(str, Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class EntryMethod(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    BREAKOUT_CONFIRMATION = "breakout_confirmation"
    PULLBACK = "pullback"
    STOP_ENTRY = "stop_entry"


class ExitReason(str, Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT_1 = "take_profit_1"
    TAKE_PROFIT_2 = "take_profit_2"
    TRAILING_STOP = "trailing_stop"
    TIME_STOP = "time_stop"
    THESIS_DETERIORATION = "thesis_deterioration"
    NEWS_SHOCK = "news_shock"
    REGIME_DETERIORATION = "regime_deterioration"
    MANUAL = "manual"
    EMERGENCY = "emergency"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class CandidateStatus(str, Enum):
    SCREENED = "screened"
    SCORED = "scored"
    RANKED = "ranked"
    SELECTED = "selected"
    REJECTED = "rejected"


class SignalStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


class AlertLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(str, Enum):
    RISK = "risk"
    EXECUTION = "execution"
    DATA = "data"
    SYSTEM = "system"
    STRATEGY = "strategy"
