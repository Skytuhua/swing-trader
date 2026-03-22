"""
Overnight gap risk assessment for open positions.

A "gap" occurs when the next session's opening price differs materially from
the previous close, bypassing any stop-loss order placed during the prior
session.  This module estimates the probability and expected magnitude of such
a gap and recommends a protective action.

Factors considered
------------------
1. Historical gap frequency – what fraction of daily opens gapped >1%
2. Average gap magnitude – mean |open - prev_close| / prev_close
3. ATR ratio – current ATR relative to recent average (volatility regime)
4. Upcoming earnings – known binary risk event greatly elevates probability
5. VIX level – systemic fear amplifies individual stock gap risk
6. Sector-specific behaviour – some sectors gap more than others
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog

if TYPE_CHECKING:
    from src.services.market_data.manager import DataManager

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Sector gap-risk multipliers (calibrated heuristics)
# ---------------------------------------------------------------------------

_SECTOR_GAP_MULTIPLIERS: dict[str, float] = {
    "biotechnology": 1.80,
    "biotech": 1.80,
    "pharmaceutical": 1.50,
    "pharma": 1.50,
    "healthcare": 1.20,
    "technology": 1.15,
    "software": 1.15,
    "semiconductor": 1.20,
    "energy": 1.10,
    "oil": 1.10,
    "gas": 1.10,
    "financial": 1.05,
    "bank": 1.05,
    "consumer": 1.00,
    "industrial": 1.00,
    "utility": 0.85,
    "utilities": 0.85,
    "real estate": 0.90,
    "reit": 0.90,
    "materials": 1.00,
    "communication": 1.05,
    "telecom": 1.05,
}

# VIX level → systemic multiplier
_VIX_BREAKPOINTS: list[tuple[float, float]] = [
    (12.0, 0.80),   # very low VIX – calm market
    (16.0, 0.90),
    (20.0, 1.00),   # baseline
    (25.0, 1.15),
    (30.0, 1.30),
    (40.0, 1.55),
    (60.0, 1.80),   # extreme stress
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GapRiskAssessment:
    """Overnight gap risk evaluation for a single position.

    Attributes
    ----------
    gap_probability:
        Estimated probability (0–1) that the next open will gap by more than
        the position's stop distance.
    expected_gap_magnitude_pct:
        Expected magnitude of any gap as a percentage of the close price.
    risk_score:
        Composite risk score (0–100).  Higher = more dangerous.
    recommended_action:
        One of: ``"hold"``, ``"tighten_stop"``, ``"reduce_size"``, ``"close"``.
    stop_at_risk:
        True when the expected gap magnitude is large enough to bypass the
        stop-loss order.
    factors:
        Dict of individual factor contributions for transparency.
    reason:
        Human-readable summary of the assessment.
    """

    gap_probability: float
    expected_gap_magnitude_pct: float
    risk_score: float
    recommended_action: str
    stop_at_risk: bool
    factors: dict[str, float] = field(default_factory=dict)
    reason: str = ""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class GapRiskAnalyzer:
    """Assess overnight gap risk for open positions.

    Parameters
    ----------
    high_risk_threshold:
        Normalised risk score (0–1, internally converted to 0–100 scale)
        above which the recommended action escalates beyond ``"hold"``.
    gap_definition_pct:
        The minimum open/close change (absolute %) that qualifies as a gap.
    atr_lookback:
        Number of trading days used to compute ATR.
    history_lookback_days:
        Calendar days of history used to compute historical gap frequency.
    """

    def __init__(
        self,
        high_risk_threshold: float = 0.6,
        gap_definition_pct: float = 1.0,
        atr_lookback: int = 14,
        history_lookback_days: int = 252,
    ) -> None:
        self.high_risk_threshold = high_risk_threshold
        # Convert 0-1 to 0-100 scale used internally
        self._high_risk_score = high_risk_threshold * 100.0
        self.gap_definition_pct = gap_definition_pct
        self.atr_lookback = atr_lookback
        self.history_lookback_days = history_lookback_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def assess_gap_risk(
        self,
        ticker: str,
        position: Any,
        data_manager: "DataManager",
    ) -> GapRiskAssessment:
        """Assess overnight gap risk for a position.

        Parameters
        ----------
        ticker:
            Ticker symbol.
        position:
            Position object.  Accessed attributes: ``stop_price`` (float),
            ``entry_price`` (float), ``quantity`` (int).  Any missing
            attribute is handled gracefully.
        data_manager:
            DataManager used to fetch price history and market snapshot.

        Returns
        -------
        GapRiskAssessment
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=self.history_lookback_days + 30)

        try:
            df = await data_manager.get_daily_ohlcv(ticker, start_date, end_date)
        except Exception as exc:
            logger.warning(
                "gap_risk.fetch_failed", ticker=ticker, error=str(exc)
            )
            df = pd.DataFrame()

        # Attempt to fetch VIX from market snapshot
        vix_level = await self._get_vix(data_manager)

        # Attempt to fetch sector info
        sector = await self._get_sector(ticker, data_manager)

        # --- Factor 1: Historical gap frequency and magnitude ---
        gap_freq, avg_gap_mag = self._compute_historical_gaps(df)

        # --- Factor 2: ATR ratio (current vs 60-day mean) ---
        atr_ratio = self._compute_atr_ratio(df)

        # --- Factor 3: Earnings proximity (placeholder – uses company info) ---
        earnings_factor = await self._earnings_proximity_factor(ticker, data_manager)

        # --- Factor 4: VIX multiplier ---
        vix_multiplier = self._vix_multiplier(vix_level)

        # --- Factor 5: Sector multiplier ---
        sector_multiplier = self._sector_multiplier(sector)

        # --- Composite probability estimate ---
        # Base probability derived from historical frequency, adjusted by
        # volatility regime and systemic/sector conditions
        base_probability = gap_freq
        adjusted_probability = (
            base_probability
            * atr_ratio
            * vix_multiplier
            * sector_multiplier
            * earnings_factor
        )
        # Clamp to [0, 1]
        gap_probability = float(np.clip(adjusted_probability, 0.0, 1.0))

        # Expected magnitude (also adjusted by multipliers)
        expected_gap_magnitude_pct = float(
            avg_gap_mag * atr_ratio * vix_multiplier * sector_multiplier * earnings_factor
        )
        expected_gap_magnitude_pct = max(0.0, expected_gap_magnitude_pct)

        # --- Composite risk score 0-100 ---
        # Weighted combination of probability and magnitude components
        prob_component = gap_probability * 50.0   # contributes up to 50 pts
        mag_component = min(expected_gap_magnitude_pct * 5.0, 30.0)  # up to 30 pts
        earnings_component = (earnings_factor - 1.0) * 20.0  # up to ~20 pts for high earnings risk
        risk_score = float(np.clip(prob_component + mag_component + earnings_component, 0.0, 100.0))

        # --- Stop at risk? ---
        stop_price = getattr(position, "stop_price", None)
        entry_price = getattr(position, "entry_price", None)
        stop_at_risk = False

        if stop_price is not None and entry_price is not None and entry_price > 0:
            stop_distance_pct = abs(entry_price - stop_price) / entry_price * 100.0
            if expected_gap_magnitude_pct > stop_distance_pct:
                stop_at_risk = True

        # --- Recommended action ---
        recommended_action = self._recommend_action(
            risk_score=risk_score,
            stop_at_risk=stop_at_risk,
            earnings_factor=earnings_factor,
        )

        factors = {
            "gap_frequency_historical": round(gap_freq, 4),
            "avg_gap_magnitude_pct_historical": round(avg_gap_mag, 4),
            "atr_ratio": round(atr_ratio, 4),
            "vix_level": round(vix_level, 2),
            "vix_multiplier": round(vix_multiplier, 4),
            "sector_multiplier": round(sector_multiplier, 4),
            "earnings_factor": round(earnings_factor, 4),
        }

        reason = (
            f"Risk score {risk_score:.1f}/100 for {ticker}. "
            f"Historical gap freq: {gap_freq:.1%}, avg magnitude: {avg_gap_mag:.2f}%. "
            f"VIX={vix_level:.1f} (×{vix_multiplier:.2f}), "
            f"sector='{sector}' (×{sector_multiplier:.2f}), "
            f"earnings factor: ×{earnings_factor:.2f}. "
            f"Stop at risk: {stop_at_risk}. "
            f"Action: {recommended_action}."
        )

        logger.info(
            "gap_risk.assessed",
            ticker=ticker,
            risk_score=round(risk_score, 1),
            recommended_action=recommended_action,
            stop_at_risk=stop_at_risk,
        )

        return GapRiskAssessment(
            gap_probability=gap_probability,
            expected_gap_magnitude_pct=expected_gap_magnitude_pct,
            risk_score=risk_score,
            recommended_action=recommended_action,
            stop_at_risk=stop_at_risk,
            factors=factors,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_historical_gaps(
        self, df: pd.DataFrame
    ) -> tuple[float, float]:
        """Return (gap_frequency, avg_gap_magnitude_pct).

        gap_frequency  = fraction of sessions where |open/prev_close - 1| > gap_definition_pct
        avg_gap_magnitude_pct = mean of all |open/prev_close - 1| * 100
        """
        if df.empty or not {"open", "close"}.issubset(df.columns):
            return 0.15, 1.5  # sensible defaults when no data available

        df = df.copy().dropna(subset=["open", "close"])
        if len(df) < 5:
            return 0.15, 1.5

        prev_close = df["close"].shift(1)
        gap_pct = ((df["open"] - prev_close) / prev_close * 100.0).abs().dropna()

        if gap_pct.empty:
            return 0.15, 1.5

        gap_freq = float((gap_pct > self.gap_definition_pct).mean())
        avg_gap_mag = float(gap_pct.mean())

        return gap_freq, avg_gap_mag

    def _compute_atr_ratio(self, df: pd.DataFrame) -> float:
        """Compute current ATR relative to 60-day mean ATR.

        Returns a ratio > 1.0 when recent volatility is elevated.
        """
        if df.empty or not {"high", "low", "close"}.issubset(df.columns):
            return 1.0

        df = df.copy().dropna(subset=["high", "low", "close"])
        if len(df) < self.atr_lookback + 2:
            return 1.0

        # True range
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(window=self.atr_lookback, min_periods=self.atr_lookback).mean()
        atr = atr.dropna()

        if len(atr) < 2:
            return 1.0

        current_atr = float(atr.iloc[-1])
        # Compare to mean of the prior 60 bars (excluding the most recent)
        baseline_window = min(60, len(atr) - 1)
        baseline_atr = float(atr.iloc[-(baseline_window + 1) : -1].mean())

        if baseline_atr <= 0:
            return 1.0

        ratio = current_atr / baseline_atr
        # Clamp to [0.5, 3.0] to avoid extreme outliers dominating
        return float(np.clip(ratio, 0.5, 3.0))

    def _vix_multiplier(self, vix: float) -> float:
        """Map VIX level to a gap-risk multiplier via linear interpolation."""
        if vix <= _VIX_BREAKPOINTS[0][0]:
            return _VIX_BREAKPOINTS[0][1]
        if vix >= _VIX_BREAKPOINTS[-1][0]:
            return _VIX_BREAKPOINTS[-1][1]

        for i in range(len(_VIX_BREAKPOINTS) - 1):
            v_lo, m_lo = _VIX_BREAKPOINTS[i]
            v_hi, m_hi = _VIX_BREAKPOINTS[i + 1]
            if v_lo <= vix <= v_hi:
                t = (vix - v_lo) / (v_hi - v_lo)
                return m_lo + t * (m_hi - m_lo)

        return 1.0  # fallback

    def _sector_multiplier(self, sector: str) -> float:
        """Return sector-specific gap-risk multiplier."""
        sector_lower = sector.lower()
        for key, mult in _SECTOR_GAP_MULTIPLIERS.items():
            if key in sector_lower:
                return mult
        return 1.0  # unknown sector – baseline

    async def _earnings_proximity_factor(
        self,
        ticker: str,
        data_manager: "DataManager",
    ) -> float:
        """Estimate earnings proximity factor.

        Returns a multiplier:
          1.0  = no known near-term earnings event
          2.5  = earnings within the next 7 days (high binary risk)
          1.5  = earnings within 15 days

        Falls back to 1.0 if company info is unavailable.
        """
        try:
            info = await data_manager.get_company_info(ticker)
            # Look for earnings date in company info dict
            # Different providers use different keys
            for key in ("earnings_date", "next_earnings_date", "earningsDate"):
                raw = info.get(key)
                if raw is not None:
                    try:
                        if isinstance(raw, str):
                            earnings_dt = pd.Timestamp(raw).date()
                        elif isinstance(raw, (int, float)):
                            # Unix timestamp
                            earnings_dt = pd.Timestamp(raw, unit="s").date()
                        else:
                            earnings_dt = date.fromisoformat(str(raw))

                        days_away = (earnings_dt - date.today()).days
                        if 0 <= days_away <= 7:
                            return 2.5
                        elif 0 <= days_away <= 15:
                            return 1.5
                        elif 0 <= days_away <= 30:
                            return 1.2
                        else:
                            return 1.0
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug(
                "gap_risk.earnings_lookup_failed",
                ticker=ticker,
                error=str(exc),
            )
        return 1.0

    async def _get_vix(self, data_manager: "DataManager") -> float:
        """Try to retrieve current VIX from the market snapshot."""
        try:
            snapshot = await data_manager.get_market_snapshot()
            for key in ("vix", "VIX", "volatility_index"):
                if key in snapshot:
                    return float(snapshot[key])
        except Exception:
            pass
        return 20.0  # baseline VIX (neutral)

    async def _get_sector(self, ticker: str, data_manager: "DataManager") -> str:
        """Try to retrieve sector from company info."""
        try:
            info = await data_manager.get_company_info(ticker)
            for key in ("sector", "industry", "gics_sector"):
                if key in info and info[key]:
                    return str(info[key])
        except Exception:
            pass
        return "unknown"

    def _recommend_action(
        self,
        risk_score: float,
        stop_at_risk: bool,
        earnings_factor: float,
    ) -> str:
        """Map risk score and qualitative flags to a recommended action."""
        # Immediate close if extremely high risk + stop can be gapped through
        if stop_at_risk and earnings_factor >= 2.0:
            return "close"

        if risk_score >= 80.0:
            if stop_at_risk:
                return "close"
            return "reduce_size"

        if risk_score >= 60.0:
            if stop_at_risk:
                return "reduce_size"
            return "tighten_stop"

        if risk_score >= 40.0:
            return "tighten_stop"

        return "hold"
