from __future__ import annotations

"""
catalyst_pipeline.py

Dedicated catalyst pipeline that aggregates upstream evidence into a
structured input for the LLM:
- External company news (FMP, Finnhub)
- Fundamental inferred catalysts (acceleration, slowdown, dilution, leverage)
- Technical / market-implied catalysts (breakout, breakdown, volume spike, trend regime)

This module does NOT ask the LLM to hallucinate catalysts. It only:
- fetches and normalizes raw evidence
- constructs machine-readable candidate lists for Prompt 1D.
"""

from dataclasses import dataclass, asdict
import datetime as dt
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from news_clients import (
    fetch_finnhub_company_news,
    fetch_fmp_company_news,
    news_items_to_dicts,
)


@dataclass
class CatalystCandidate:
    category: str  # "company" | "industry" | "speculative"
    description: str
    source_type: str  # "reported" | "inferred" | "market_implied" | "speculative"
    direction: str  # "positive" | "negative" | "mixed"
    evidence: Dict[str, Any]


def _safe_pct_change(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.pct_change(fill_method=None)


def _latest(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    val = pd.to_numeric(series, errors="coerce").iloc[-1]
    if isinstance(val, (float, int)) and np.isfinite(val):
        return float(val)
    return None


def infer_fundamental_catalysts(metrics: pd.DataFrame) -> List[CatalystCandidate]:
    """
    Simple rule-based fundamental catalysts from annual metrics:
    - revenue acceleration / slowdown
    - margin inflection
    - leverage improving / worsening
    """
    if metrics is None or metrics.empty:
        return []

    cands: List[CatalystCandidate] = []

    rev = pd.to_numeric(metrics.get("TotalRevenue", pd.Series(dtype=float)), errors="coerce")
    net_margin = pd.to_numeric(metrics.get("Net_Margin", pd.Series(dtype=float)), errors="coerce")
    d2e = pd.to_numeric(metrics.get("Debt_to_Equity", pd.Series(dtype=float)), errors="coerce")

    rev_growth = _safe_pct_change(rev)
    margin_change = _safe_pct_change(net_margin)
    d2e_change = _safe_pct_change(d2e)

    if len(rev_growth.dropna()) >= 2:
        last = rev_growth.iloc[-1]
        prev = rev_growth.iloc[-2]
        if last > 0 and last > prev + 0.1:
            cands.append(
                CatalystCandidate(
                    category="company",
                    source_type="inferred",
                    direction="positive",
                    description="Revenue growth has recently accelerated on a year-over-year basis.",
                    evidence={
                        "rev_growth_last": float(last),
                        "rev_growth_prev": float(prev),
                    },
                )
            )
        elif last < prev - 0.1:
            cands.append(
                CatalystCandidate(
                    category="company",
                    source_type="inferred",
                    direction="negative",
                    description="Revenue growth has decelerated or turned down recently.",
                    evidence={
                        "rev_growth_last": float(last),
                        "rev_growth_prev": float(prev),
                    },
                )
            )

    if len(margin_change.dropna()) >= 1:
        mc = margin_change.iloc[-1]
        if mc > 0.05:
            cands.append(
                CatalystCandidate(
                    category="company",
                    source_type="inferred",
                    direction="positive",
                    description="Net margin has improved meaningfully, indicating margin expansion.",
                    evidence={"net_margin_change_last": float(mc)},
                )
            )
        elif mc < -0.05:
            cands.append(
                CatalystCandidate(
                    category="company",
                    source_type="inferred",
                    direction="negative",
                    description="Net margin has compressed materially, signaling profitability pressure.",
                    evidence={"net_margin_change_last": float(mc)},
                )
            )

    if len(d2e_change.dropna()) >= 1:
        dc = d2e_change.iloc[-1]
        if dc < -0.1:
            cands.append(
                CatalystCandidate(
                    category="company",
                    source_type="inferred",
                    direction="positive",
                    description="Leverage (debt-to-equity) has decreased, improving balance sheet resilience.",
                    evidence={"d2e_change_last": float(dc)},
                )
            )
        elif dc > 0.1:
            cands.append(
                CatalystCandidate(
                    category="company",
                    source_type="inferred",
                    direction="negative",
                    description="Leverage (debt-to-equity) has increased, weakening the balance sheet.",
                    evidence={"d2e_change_last": float(dc)},
                )
            )

    return cands


def infer_technical_catalysts(tech_snapshot: Dict[str, Any]) -> List[CatalystCandidate]:
    """
    Infer market-implied catalysts from the technical snapshot:
    - breakout / breakdown vs key levels
    - abnormal volume
    - recent strong positive / negative returns
    """
    if not isinstance(tech_snapshot, dict):
        return []

    cands: List[CatalystCandidate] = []
    last = tech_snapshot.get("last_close")
    support = tech_snapshot.get("support")
    resistance_near = tech_snapshot.get("resistance_near")
    ma50 = tech_snapshot.get("ma_50")
    ma200 = tech_snapshot.get("ma_200")
    vol_last = tech_snapshot.get("volume_last")
    avg_vol_20 = tech_snapshot.get("avg_volume_20")
    r1m = tech_snapshot.get("return_1m")
    r3m = tech_snapshot.get("return_3m")

    def _f(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            v = float(x)
            if not np.isfinite(v):
                return None
            return v
        except (TypeError, ValueError):
            return None

    last_f = _f(last)
    support_f = _f(support)
    res_f = _f(resistance_near)
    ma50_f = _f(ma50)
    ma200_f = _f(ma200)
    vol_last_f = _f(vol_last)
    avg_vol_20_f = _f(avg_vol_20)
    r1m_f = _f(r1m)
    r3m_f = _f(r3m)

    # Breakout / breakdown vs resistance/support
    if last_f is not None and res_f is not None and last_f > res_f * 1.02:
        cands.append(
            CatalystCandidate(
                category="company",
                source_type="market_implied",
                direction="positive",
                description="Price has broken out above recent resistance on a closing basis.",
                evidence={"last_close": last_f, "resistance_near": res_f},
            )
        )
    if last_f is not None and support_f is not None and last_f < support_f * 0.98:
        cands.append(
            CatalystCandidate(
                category="company",
                source_type="market_implied",
                direction="negative",
                description="Price has broken down below recent support on a closing basis.",
                evidence={"last_close": last_f, "support": support_f},
            )
        )

    # Trend regime vs 50/200-day
    if last_f is not None and ma50_f is not None and ma200_f is not None:
        if last_f > ma50_f > ma200_f:
            cands.append(
                CatalystCandidate(
                    category="company",
                    source_type="market_implied",
                    direction="positive",
                    description="Price is in a strong uptrend above both 50-day and 200-day moving averages.",
                    evidence={"last_close": last_f, "ma_50": ma50_f, "ma_200": ma200_f},
                )
            )
        if last_f < ma50_f < ma200_f:
            cands.append(
                CatalystCandidate(
                    category="company",
                    source_type="market_implied",
                    direction="negative",
                    description="Price is in a weak downtrend below both 50-day and 200-day moving averages.",
                    evidence={"last_close": last_f, "ma_50": ma50_f, "ma_200": ma200_f},
                )
            )

    # Volume spike
    if vol_last_f is not None and avg_vol_20_f is not None and avg_vol_20_f > 0:
        ratio = vol_last_f / avg_vol_20_f
        if ratio >= 2.0:
            cands.append(
                CatalystCandidate(
                    category="company",
                    source_type="market_implied",
                    direction="mixed",
                    description="Unusual volume spike relative to recent history, indicating elevated attention.",
                    evidence={"volume_last": vol_last_f, "avg_volume_20": avg_vol_20_f},
                )
            )

    # Recent strong performance
    if r3m_f is not None and r3m_f > 0.3:
        cands.append(
            CatalystCandidate(
                category="speculative",
                source_type="market_implied",
                direction="positive",
                description="Strong multi-month positive return, potentially reflecting speculative momentum.",
                evidence={"return_3m": r3m_f},
            )
        )
    if r3m_f is not None and r3m_f < -0.3:
        cands.append(
            CatalystCandidate(
                category="speculative",
                source_type="market_implied",
                direction="negative",
                description="Severe multi-month drawdown, indicating strong negative sentiment or thesis break.",
                evidence={"return_3m": r3m_f},
            )
        )

    return cands


def fetch_external_news_candidates(
    ticker: str,
    analysis_date: str,
    lookback_days: int = 60,
    limit_per_source: int = 30,
) -> List[Dict[str, Any]]:
    """Fetch external news/press releases and return as JSON-safe dicts."""
    end = dt.date.fromisoformat(analysis_date)
    start = end - dt.timedelta(days=lookback_days)
    fmp_items = fetch_fmp_company_news(ticker, start.isoformat(), end.isoformat(), limit=limit_per_source)
    finnhub_items = fetch_finnhub_company_news(ticker, start.isoformat(), end.isoformat(), limit=limit_per_source)
    all_items = fmp_items + finnhub_items
    return news_items_to_dicts(all_items)


def build_catalyst_inputs(
    ticker: str,
    analysis_date: str,
    fundamentals: Dict[str, Any],
    tech_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a structured catalyst input dict for Prompt 1D. This is the only
    place where raw external + inferred catalyst evidence is assembled.
    """
    metrics_df = fundamentals.get("metrics") if isinstance(fundamentals, dict) else None
    if not isinstance(metrics_df, pd.DataFrame):
        metrics_df = pd.DataFrame()

    inferred_fund_cats = [asdict(c) for c in infer_fundamental_catalysts(metrics_df)]
    inferred_tech_cats = [asdict(c) for c in infer_technical_catalysts(tech_snapshot)]
    news_items = fetch_external_news_candidates(ticker, analysis_date)

    return {
        "ticker": ticker,
        "analysis_date": analysis_date,
        "company_news": news_items,
        "fundamental_inferred": inferred_fund_cats,
        "technical_inferred": inferred_tech_cats,
        # Industry / ecosystem catalysts can be added later via peers / sector feeds.
        "industry_candidates": [],
    }

