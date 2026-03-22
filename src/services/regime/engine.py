"""RegimeEngine — classify the current market regime for swing trading.

Classification
--------------
The engine combines five independent signals into a composite ``regime_score``
and maps it to one of three regimes:

  FAVORABLE  (regime_score > 25)  : Conditions support swing-long entries.
  MIXED      (-10 ≤ score ≤ 25)   : Selective / reduced-size entries only.
  UNFAVORABLE (score < -10)        : No new entries; tight risk controls.

Signals
-------
1. Benchmark trend (40 pts max):
   SPY, QQQ, IWM each scored [-1, +1] via moving-average alignment.
   Average of all three × 40.

2. Breadth proxy (20 pts max):
   ``breadth_pct_above_50ma`` from 0–100; centred at 50.
   Contribution: (breadth_score − 50) × 0.4  → range [−20, +20].

3. Breakout failure rate (20 pts max):
   Recent proportion of breakout failures (0 = none fail, 1 = all fail).
   Contribution: (1 − failure_rate) × 20  → range [0, +20].

4. VIX level (20 pts max):
   VIX < 15 → +20 (calm), 15–20 → +10, 20–25 → 0, 25–30 → −10,
   30–35 → −20, > 35 → −20 (extreme fear).

5. (Optional) Trend confirmation:
   When SPY, QQQ, and IWM are all in the same direction, a ±5 confirmation
   bonus/penalty is applied.

Input ``market_data`` dict keys
--------------------------------
- 'SPY': pd.DataFrame with OHLCV columns ('close' at minimum).
- 'QQQ': pd.DataFrame with OHLCV columns.
- 'IWM': pd.DataFrame with OHLCV columns.
- 'VIX_level': float  (current VIX index value, default 20).
- 'breadth_pct_above_50ma': float 0-100 (% of S&P 500 stocks above 50-day MA).
- 'breakout_failure_rate': float 0-1 (fraction of recent breakouts that failed).

All DataFrames must have at least 50 rows to compute SMAs reliably.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

try:
    import pandas as pd
    _PD_AVAILABLE = True
except ImportError:
    _PD_AVAILABLE = False

from src.core.enums import MarketRegime
from src.core.exceptions import DataProviderError

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# RegimeAssessment dataclass
# ---------------------------------------------------------------------------


@dataclass
class RegimeAssessment:
    """Full regime classification output."""

    regime: MarketRegime
    """Classified regime: FAVORABLE | MIXED | UNFAVORABLE."""

    score: float
    """Raw composite score in approximately [-80, +80]."""

    confidence: float
    """0-1 confidence: abs(score) / 80; higher = less ambiguous."""

    spy_trend: float
    """SPY trend assessment: -1 (downtrend) → +1 (uptrend)."""

    qqq_trend: float
    """QQQ trend."""

    iwm_trend: float
    """IWM trend."""

    vix_level: float
    """Current VIX level."""

    vix_regime: str
    """Descriptive VIX bucket: low | normal | elevated | extreme."""

    breadth_score: float
    """% of stocks above 50-day MA, 0-100."""

    breakout_failure_rate: float
    """Fraction of recent breakouts that failed, 0-1."""

    breadth_contribution: float = 0.0
    vix_contribution: float = 0.0
    trend_contribution: float = 0.0
    breakout_contribution: float = 0.0

    confirmation_bonus: float = 0.0
    """Trend confirmation bonus when all three benchmarks agree."""

    data_quality: str = "good"
    """Data quality flag for this assessment."""

    explanation: str = ""
    timestamp: Optional[object] = None  # datetime, set by caller if needed

    sub_scores: dict = field(default_factory=dict)
    """Breakdown of each signal's raw contribution."""


# ---------------------------------------------------------------------------
# Trend assessment helpers
# ---------------------------------------------------------------------------

_SMA_SHORT = 20
_SMA_MEDIUM = 50
_SMA_LONG = 200
_MIN_ROWS = 50  # minimum rows to compute reliable SMAs


def _compute_sma(closes: np.ndarray, period: int) -> float:
    """Return the simple moving average of the last ``period`` closes."""
    if len(closes) < period:
        return float(closes[-1]) if len(closes) > 0 else float("nan")
    return float(np.mean(closes[-period:]))


def _assess_trend(df: object) -> float:
    """Assess the trend of a DataFrame as a value in [-1, +1].

    Methodology:
    1. Compute SMA-20, SMA-50, SMA-200.
    2. Assess alignment (SMA-20 > SMA-50 > SMA-200 = bullish).
    3. Price vs SMA-20 (above = +).
    4. SMA-20 slope over last 5 bars (positive = +).

    Returns a blended score in [-1, +1].
    """
    if not _PD_AVAILABLE:
        return 0.0

    try:
        closes = np.array(df["close"].dropna().tolist(), dtype=float)  # type: ignore[index]
    except (KeyError, AttributeError, TypeError):
        try:
            closes = np.array(df["Close"].dropna().tolist(), dtype=float)  # type: ignore[index]
        except Exception:
            logger.warning("regime_engine_no_close_column")
            return 0.0

    if len(closes) < 5:
        return 0.0

    current_price = closes[-1]

    sma20 = _compute_sma(closes, _SMA_SHORT)
    sma50 = _compute_sma(closes, _SMA_MEDIUM)
    sma200 = _compute_sma(closes, _SMA_LONG)

    score = 0.0
    components = 0

    # --- MA alignment (3 checkpoints) ---
    # SMA-20 vs SMA-50
    if not np.isnan(sma20) and not np.isnan(sma50):
        score += 1.0 if sma20 > sma50 else -1.0
        components += 1

    # SMA-50 vs SMA-200
    if not np.isnan(sma50) and not np.isnan(sma200):
        score += 1.0 if sma50 > sma200 else -1.0
        components += 1

    # Price vs SMA-20
    if not np.isnan(sma20) and sma20 != 0:
        score += 1.0 if current_price > sma20 else -1.0
        components += 1

    # --- SMA-20 slope (past 5 bars) ---
    if len(closes) >= _SMA_SHORT + 5:
        sma20_now = float(np.mean(closes[-_SMA_SHORT:]))
        sma20_5ago = float(np.mean(closes[-((_SMA_SHORT + 5)):-5]))
        if sma20_5ago != 0:
            slope = (sma20_now - sma20_5ago) / sma20_5ago
            # Normalise slope: ±2% move in 5 bars is a strong signal
            score += max(-1.0, min(1.0, slope / 0.02))
            components += 1

    if components == 0:
        return 0.0

    return round(max(-1.0, min(1.0, score / components)), 4)


# ---------------------------------------------------------------------------
# VIX contribution
# ---------------------------------------------------------------------------

def _vix_contribution(vix: float) -> tuple[float, str]:
    """Return (score_contribution, regime_label) for a VIX level."""
    if vix < 15:
        return +20.0, "low"
    if vix < 20:
        return +10.0, "normal"
    if vix < 25:
        return   0.0, "normal"
    if vix < 30:
        return -10.0, "elevated"
    if vix < 35:
        return -20.0, "elevated"
    return -20.0, "extreme"


# ---------------------------------------------------------------------------
# RegimeEngine
# ---------------------------------------------------------------------------


class RegimeEngine:
    """Classify the current market regime for swing trading.

    Usage::

        engine = RegimeEngine()
        assessment = await engine.classify(market_data)
        if assessment.regime == MarketRegime.FAVORABLE:
            ...
    """

    def __init__(
        self,
        favorable_threshold: float = 25.0,
        unfavorable_threshold: float = -10.0,
        trend_confirmation_bonus: float = 5.0,
    ) -> None:
        self.favorable_threshold = favorable_threshold
        self.unfavorable_threshold = unfavorable_threshold
        self.trend_confirmation_bonus = trend_confirmation_bonus

    async def classify(self, market_data: dict) -> RegimeAssessment:
        """Classify the current market regime.

        Args:
            market_data: Dict containing:
                - 'SPY', 'QQQ', 'IWM': DataFrames with at least 'close' column.
                - 'VIX_level': float (current VIX), default 20.
                - 'breadth_pct_above_50ma': float 0-100, default 50.
                - 'breakout_failure_rate': float 0-1, default 0.3.

        Returns:
            RegimeAssessment with regime, score, confidence, and explanation.

        Raises:
            DataProviderError: If required benchmark data is missing or invalid.
        """
        spy_df = market_data.get("SPY")
        qqq_df = market_data.get("QQQ")
        iwm_df = market_data.get("IWM")
        vix = float(market_data.get("VIX_level", 20.0))
        breadth_score = float(market_data.get("breadth_pct_above_50ma", 50.0))
        breakout_failure_rate = float(market_data.get("breakout_failure_rate", 0.30))

        # Validate inputs
        if spy_df is None or qqq_df is None or iwm_df is None:
            missing = [k for k, v in {"SPY": spy_df, "QQQ": qqq_df, "IWM": iwm_df}.items() if v is None]
            raise DataProviderError(
                f"RegimeEngine missing required benchmark data: {missing}"
            )

        # Data quality check
        data_quality = "good"
        for name, df in [("SPY", spy_df), ("QQQ", qqq_df), ("IWM", iwm_df)]:
            try:
                n = len(df)
            except Exception:
                n = 0
            if n < _MIN_ROWS:
                logger.warning("regime_engine_short_df", symbol=name, rows=n, required=_MIN_ROWS)
                data_quality = "degraded"

        # ----------------------------------------------------------------
        # 1. Benchmark trend scores
        # ----------------------------------------------------------------
        spy_trend = _assess_trend(spy_df)
        qqq_trend = _assess_trend(qqq_df)
        iwm_trend = _assess_trend(iwm_df)

        trend_avg = (spy_trend + qqq_trend + iwm_trend) / 3.0
        trend_contribution = trend_avg * 40.0  # range [-40, +40]

        # ----------------------------------------------------------------
        # 2. Breadth
        # ----------------------------------------------------------------
        breadth_score_clamped = max(0.0, min(100.0, breadth_score))
        breadth_contribution = (breadth_score_clamped - 50.0) * 0.4  # [-20, +20]

        # ----------------------------------------------------------------
        # 3. Breakout failure rate
        # ----------------------------------------------------------------
        failure_clamped = max(0.0, min(1.0, breakout_failure_rate))
        breakout_contribution = (1.0 - failure_clamped) * 20.0  # [0, +20]

        # ----------------------------------------------------------------
        # 4. VIX
        # ----------------------------------------------------------------
        vix_contribution, vix_regime = _vix_contribution(vix)

        # ----------------------------------------------------------------
        # 5. Trend confirmation bonus
        # ----------------------------------------------------------------
        confirmation_bonus = 0.0
        if spy_trend > 0 and qqq_trend > 0 and iwm_trend > 0:
            confirmation_bonus = +self.trend_confirmation_bonus  # all bullish → bonus
        elif spy_trend < 0 and qqq_trend < 0 and iwm_trend < 0:
            confirmation_bonus = -self.trend_confirmation_bonus  # all bearish → penalty

        # ----------------------------------------------------------------
        # 6. Composite score
        # ----------------------------------------------------------------
        regime_score = (
            trend_contribution
            + breadth_contribution
            + breakout_contribution
            + vix_contribution
            + confirmation_bonus
        )

        # ----------------------------------------------------------------
        # 7. Regime classification
        # ----------------------------------------------------------------
        if regime_score > self.favorable_threshold:
            regime = MarketRegime.FAVORABLE
        elif regime_score > self.unfavorable_threshold:
            regime = MarketRegime.MIXED
        else:
            regime = MarketRegime.UNFAVORABLE

        # ----------------------------------------------------------------
        # 8. Confidence: normalised distance from the decision boundary
        # ----------------------------------------------------------------
        # Max achievable raw score ≈ 40+20+20+20+5 = 105; use 80 for normalisation
        confidence = min(1.0, abs(regime_score) / 80.0)

        # ----------------------------------------------------------------
        # 9. Explanation
        # ----------------------------------------------------------------
        explanation = self._build_explanation(
            regime=regime,
            regime_score=regime_score,
            spy_trend=spy_trend,
            qqq_trend=qqq_trend,
            iwm_trend=iwm_trend,
            trend_contribution=trend_contribution,
            breadth_score=breadth_score,
            breadth_contribution=breadth_contribution,
            breakout_failure_rate=breakout_failure_rate,
            breakout_contribution=breakout_contribution,
            vix=vix,
            vix_regime=vix_regime,
            vix_contribution=vix_contribution,
            confirmation_bonus=confirmation_bonus,
        )

        logger.info(
            "regime_classified",
            regime=regime.value,
            score=round(regime_score, 2),
            confidence=round(confidence, 3),
            vix=vix,
            vix_regime=vix_regime,
            spy_trend=round(spy_trend, 3),
            qqq_trend=round(qqq_trend, 3),
            iwm_trend=round(iwm_trend, 3),
            breadth_score=breadth_score,
        )

        return RegimeAssessment(
            regime=regime,
            score=round(regime_score, 4),
            confidence=round(confidence, 4),
            spy_trend=round(spy_trend, 4),
            qqq_trend=round(qqq_trend, 4),
            iwm_trend=round(iwm_trend, 4),
            vix_level=vix,
            vix_regime=vix_regime,
            breadth_score=breadth_score,
            breakout_failure_rate=breakout_failure_rate,
            breadth_contribution=round(breadth_contribution, 4),
            vix_contribution=vix_contribution,
            trend_contribution=round(trend_contribution, 4),
            breakout_contribution=round(breakout_contribution, 4),
            confirmation_bonus=confirmation_bonus,
            data_quality=data_quality,
            explanation=explanation,
            sub_scores={
                "trend": round(trend_contribution, 2),
                "breadth": round(breadth_contribution, 2),
                "breakout": round(breakout_contribution, 2),
                "vix": vix_contribution,
                "confirmation": confirmation_bonus,
                "total": round(regime_score, 2),
            },
        )

    @staticmethod
    def _build_explanation(
        regime: MarketRegime,
        regime_score: float,
        spy_trend: float,
        qqq_trend: float,
        iwm_trend: float,
        trend_contribution: float,
        breadth_score: float,
        breadth_contribution: float,
        breakout_failure_rate: float,
        breakout_contribution: float,
        vix: float,
        vix_regime: str,
        vix_contribution: float,
        confirmation_bonus: float,
    ) -> str:
        def trend_label(t: float) -> str:
            if t > 0.4:
                return "strong uptrend"
            if t > 0.1:
                return "uptrend"
            if t < -0.4:
                return "strong downtrend"
            if t < -0.1:
                return "downtrend"
            return "sideways"

        lines = [
            f"Market Regime: {regime.value.upper()} (composite score {regime_score:+.1f}).",
            f"Benchmark trends → SPY: {trend_label(spy_trend)} ({spy_trend:+.2f}), "
            f"QQQ: {trend_label(qqq_trend)} ({qqq_trend:+.2f}), "
            f"IWM: {trend_label(iwm_trend)} ({iwm_trend:+.2f}). "
            f"Trend contribution: {trend_contribution:+.1f} pts.",
            f"Market breadth: {breadth_score:.0f}% above 50-day MA → {breadth_contribution:+.1f} pts.",
            f"Breakout failure rate: {breakout_failure_rate:.0%} → {breakout_contribution:+.1f} pts.",
            f"VIX: {vix:.1f} ({vix_regime}) → {vix_contribution:+.1f} pts.",
        ]

        if confirmation_bonus != 0:
            direction = "bullish" if confirmation_bonus > 0 else "bearish"
            lines.append(
                f"All three benchmarks confirm {direction} direction: {confirmation_bonus:+.1f} bonus."
            )

        # Trading guidance
        if regime == MarketRegime.FAVORABLE:
            lines.append(
                "Guidance: Conditions are favourable for swing-long entries with full sizing."
            )
        elif regime == MarketRegime.MIXED:
            lines.append(
                "Guidance: Mixed conditions — prefer high-quality setups with reduced position sizes."
            )
        else:
            lines.append(
                "Guidance: Unfavourable conditions — avoid new long entries; "
                "tighten stops on existing positions."
            )

        return " ".join(lines)

    def update_thresholds(
        self,
        favorable: float | None = None,
        unfavorable: float | None = None,
    ) -> None:
        """Adjust classification thresholds dynamically (e.g., for regime-aware backtesting)."""
        if favorable is not None:
            self.favorable_threshold = favorable
        if unfavorable is not None:
            self.unfavorable_threshold = unfavorable
        logger.info(
            "regime_thresholds_updated",
            favorable=self.favorable_threshold,
            unfavorable=self.unfavorable_threshold,
        )
