from __future__ import annotations

"""
archetype.py

Company archetype classification layer.

Uses lightweight, rule-based heuristics on:
- sector / industry
- market cap
- profitability and leverage signals (where available)

The archetype is intended to guide the LLM:
- which metrics to emphasize
- how to interpret catalysts
- how to frame the rating horizon and risk/reward.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


ArchetypeType = str


@dataclass
class Archetype:
    type: ArchetypeType
    confidence: str
    rationale: str


def _safe_float(val: Any) -> Optional[float]:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def classify_archetype(
    meta_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> Archetype:
    """
    Classify the company into a coarse archetype using meta and annual metrics.

    Expected columns:
    - meta_df: Sector, Industry, MarketCap
    - metrics_df: NetIncome, TotalRevenue, Debt_to_Equity
    """
    if meta_df is None or meta_df.empty:
        sector = None
        industry = None
        mcap = None
    else:
        row = meta_df.iloc[0]
        sector = str(row.get("Sector") or "") or None
        industry = str(row.get("Industry") or "") or None
        mcap = _safe_float(row.get("MarketCap"))

    # Use latest metrics row
    if metrics_df is None or metrics_df.empty:
        latest = {}
    else:
        latest = metrics_df.iloc[-1].to_dict()

    net_income = _safe_float(latest.get("NetIncome"))
    revenue = _safe_float(latest.get("TotalRevenue"))
    d2e = _safe_float(latest.get("Debt_to_Equity"))

    archetype: ArchetypeType = "mature_large_cap_compounder"
    rationale_parts = []
    confidence = "Medium"

    # Regulated utility
    if sector and "Utilities" in sector:
        archetype = "regulated_utility"
        rationale_parts.append("Sector classified as Utilities.")

    # Pre-profit / deep tech (small revenue, negative net income)
    if revenue is not None and net_income is not None:
        if revenue < 5e8 and net_income < 0:
            archetype = "pre_profit_deep_tech"
            rationale_parts.append("Low revenue with persistent losses suggests pre-profit / deep tech profile.")

    # Turnaround / balance-sheet-stressed
    if d2e is not None and d2e > 2.5:
        archetype = "turnaround_balance_sheet_stressed"
        rationale_parts.append("High debt-to-equity indicates stressed balance sheet.")

    # Cyclical industrial
    if sector and any(s in sector for s in ("Industrials", "Materials")):
        if archetype not in ("regulated_utility", "pre_profit_deep_tech", "turnaround_balance_sheet_stressed"):
            archetype = "cyclical_industrial"
            rationale_parts.append("Sector classified as Industrials/Materials (cyclical).")

    # Mature compounder by default if large cap and profitable
    if mcap is not None and mcap >= 1e10 and net_income is not None and net_income > 0:
        if archetype not in ("regulated_utility", "turnaround_balance_sheet_stressed"):
            archetype = "mature_large_cap_compounder"
            rationale_parts.append("Large market cap with positive earnings.")

    if not rationale_parts:
        rationale_parts.append("Insufficient specific signals; defaulting to mature_large_cap_compounder style lens.")

    # Confidence heuristics
    num_signals = len(rationale_parts)
    if num_signals >= 2:
        confidence = "High"
    elif num_signals == 1:
        confidence = "Medium"
    else:
        confidence = "Low"

    return Archetype(
        type=archetype,
        confidence=confidence,
        rationale=" ".join(rationale_parts),
    )


def archetype_to_dict(arc: Archetype) -> Dict[str, Any]:
    return {
        "type": arc.type,
        "confidence": arc.confidence,
        "rationale": arc.rationale,
    }

