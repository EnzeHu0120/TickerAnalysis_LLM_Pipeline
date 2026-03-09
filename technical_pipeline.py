"""
technical_pipeline.py — Technical data source (standalone module).

Uses the ta library (bukosabino/ta) to compute technical indicators from OHLCV
and build a high-info-density summary for the LLM.
- build_technical_features(df): trend, momentum, volatility, volume indicators
- technical_summary_for_llm(df): compact text summary for prompts
- build_technical_snapshot_dict(): latest values dict + summary for llm_pipeline
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# ta: https://github.com/bukosabino/ta
from ta.momentum import (
    RSIIndicator,
    StochasticOscillator,
    WilliamsRIndicator,
)
from ta.trend import (
    MACD,
    SMAIndicator,
    EMAIndicator,
)
from ta.volatility import (
    AverageTrueRange,
    BollingerBands,
)
from ta.volume import (
    OnBalanceVolumeIndicator,
    ChaikinMoneyFlowIndicator,
    MFIIndicator,
    VolumeWeightedAveragePrice,
)


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to Open/High/Low/Close/Volume (capitalized)."""
    mapping = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    out = df.copy()
    for lower, cap in mapping.items():
        if lower in out.columns and cap not in out.columns:
            out[cap] = out[lower]
    return out


def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a curated set of technical indicators using the `ta` library.
    Requires columns: Open, High, Low, Close, Volume (case-insensitive).
    Returns a copy of the DataFrame with new columns; early rows may have NaN due to windows.
    """
    out = _ensure_ohlcv(df).copy()
    if "Volume" not in out.columns:
        out["Volume"] = np.nan

    high = out["High"]
    low = out["Low"]
    close = out["Close"]
    volume = out["Volume"]

    # ----- Trend -----
    out["sma_20"] = SMAIndicator(close=close, window=20).sma_indicator()
    out["sma_50"] = SMAIndicator(close=close, window=50).sma_indicator()
    out["sma_200"] = SMAIndicator(close=close, window=200).sma_indicator()
    out["ema_20"] = EMAIndicator(close=close, window=20).ema_indicator()

    macd = MACD(close=close)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    # ----- Momentum -----
    out["rsi_14"] = RSIIndicator(close=close, window=14).rsi()
    stoch = StochasticOscillator(
        high=high, low=low, close=close, window=14, smooth_window=3
    )
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()
    out["williams_r"] = WilliamsRIndicator(high=high, low=low, close=close, lbp=14).williams_r()

    # ----- Volatility -----
    out["atr_14"] = AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()

    bb = BollingerBands(close=close, window=20, window_dev=2)
    out["bb_mid"] = bb.bollinger_mavg()
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    out["bb_width"] = bb.bollinger_wband()
    out["bb_pos"] = bb.bollinger_pband()

    # ----- Volume -----
    out["obv"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    out["cmf"] = ChaikinMoneyFlowIndicator(
        high=high, low=low, close=close, volume=volume, window=20
    ).chaikin_money_flow()
    out["mfi_14"] = MFIIndicator(
        high=high, low=low, close=close, volume=volume, window=14
    ).money_flow_index()
    # Rolling VWAP over window (ta's VWAP is period-based; on daily data this is a proxy)
    vwap_ind = VolumeWeightedAveragePrice(
        high=high, low=low, close=close, volume=volume, window=14
    )
    out["vwap_14"] = vwap_ind.volume_weighted_average_price()

    return out


def _maybe_float(val: Any) -> Optional[float]:
    try:
        if pd.isna(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _percentile_label(pct: Optional[float]) -> str:
    if pct is None:
        return "N/A"
    if pct >= 0.8:
        return "80th+ percentile"
    if pct >= 0.6:
        return "60th–80th percentile"
    if pct >= 0.4:
        return "40th–60th percentile"
    if pct >= 0.2:
        return "20th–40th percentile"
    return "below 20th percentile"


def technical_summary_for_llm(
    df: pd.DataFrame,
    last_n: int = 5,
) -> str:
    """
    Turn the last few rows of a feature-enriched OHLCV DataFrame into a short,
    high-info-density summary for LLM consumption (e.g. prompt 1C).
    """
    if df is None or df.empty:
        return "No price/indicator data available."

    tail = df.tail(last_n)
    if tail.empty:
        return "Insufficient data for technical summary."

    lines: list[str] = []
    close = df["Close"]
    last_close = _maybe_float(close.iloc[-1])
    if last_close is not None:
        lines.append(f"last_close = {last_close:.2f}")

    # RSI
    if "rsi_14" in df.columns:
        rsi = _maybe_float(df["rsi_14"].iloc[-1])
        if rsi is not None:
            lines.append(f"rsi_14 = {rsi:.1f}")

    # MACD trend
    if "macd_diff" in df.columns:
        diff = df["macd_diff"].dropna()
        if len(diff) >= 2:
            recent = diff.tail(last_n)
            rising = (recent.diff() > 0).sum()
            last_diff = _maybe_float(diff.iloc[-1])
            direction = "rising" if rising >= last_n - 1 else "falling"
            days = min(last_n, int(recent.diff().fillna(0).gt(0).sum()))
            lines.append(f"macd_hist {direction} for {days} day(s), last = {last_diff:.4f}" if last_diff is not None else f"macd_hist {direction}")

    # Close vs SMAs
    sma_20 = _maybe_float(df["sma_20"].iloc[-1]) if "sma_20" in df.columns else None
    sma_50 = _maybe_float(df["sma_50"].iloc[-1]) if "sma_50" in df.columns else None
    sma_200 = _maybe_float(df["sma_200"].iloc[-1]) if "sma_200" in df.columns else None
    if last_close is not None and any(x is not None for x in (sma_20, sma_50, sma_200)):
        above_below = []
        if sma_20 is not None:
            above_below.append("above sma_20" if last_close >= sma_20 else "below sma_20")
        if sma_50 is not None:
            above_below.append("above sma_50" if last_close >= sma_50 else "below sma_50")
        if sma_200 is not None:
            above_below.append("above sma_200" if last_close >= sma_200 else "below sma_200")
        lines.append("close " + " and ".join(above_below))

    # BB width percentile (relative to recent history)
    if "bb_width" in df.columns:
        bw = df["bb_width"].dropna()
        if len(bw) >= 20:
            last_bw = bw.iloc[-1]
            pct = (bw < last_bw).sum() / len(bw)
            lines.append(f"bb_width at {_percentile_label(pct)}")

    # Optional: Stoch, ATR, volume one-liners
    if "stoch_k" in df.columns:
        sk = _maybe_float(df["stoch_k"].iloc[-1])
        if sk is not None:
            lines.append(f"stoch_k = {sk:.1f}")
    if "atr_14" in df.columns:
        atr = _maybe_float(df["atr_14"].iloc[-1])
        if atr is not None and last_close:
            lines.append(f"atr_14 = {atr:.2f} ({100 * atr / last_close:.1f}% of price)")
    if "Volume" in df.columns and df["Volume"].notna().any():
        vol = df["Volume"]
        last_vol = _maybe_float(vol.iloc[-1])
        avg_vol_20 = vol.rolling(20).mean().iloc[-1] if len(vol) >= 20 else None
        if last_vol is not None and avg_vol_20 is not None and avg_vol_20 > 0:
            ratio = last_vol / avg_vol_20
            lines.append(f"volume last vs 20d avg = {ratio:.2f}x")

    return "\n".join(lines)


def build_technical_snapshot_dict(
    df: pd.DataFrame,
    ticker: str,
    analysis_date: str,
) -> Dict[str, Any]:
    """
    Build a compact dict of latest indicator values + summary text, for JSON
    injection into prompts and for use by llm_pipeline.
    """
    out = _ensure_ohlcv(df).copy()
    if out.empty:
        return {"ticker": ticker, "analysis_date": analysis_date, "note": "No price history."}

    out = build_technical_features(out)
    close = out["Close"]
    last_close = _maybe_float(close.iloc[-1])
    window = min(60, len(close))
    support = float(close.tail(window).min()) if window else None
    resistance_near = float(close.tail(window).max()) if window else None
    resistance_major = float(close.max())

    def get_last(col: str) -> Optional[float]:
        if col not in out.columns:
            return None
        return _maybe_float(out[col].iloc[-1])

    vol = out.get("Volume", pd.Series(dtype=float))
    avg_vol_20 = None
    if vol.notna().any() and len(vol) >= 20:
        avg_vol_20 = _maybe_float(vol.rolling(20).mean().iloc[-1])
    vol_last = _maybe_float(vol.iloc[-1]) if len(vol) else None

    return_1m = None
    return_3m = None
    if len(close) > 21:
        return_1m = _maybe_float(close.pct_change(21).iloc[-1])
    if len(close) > 63:
        return_3m = _maybe_float(close.pct_change(63).iloc[-1])

    snapshot = {
        "ticker": ticker,
        "analysis_date": analysis_date,
        "last_close": last_close,
        "return_1m": return_1m,
        "return_3m": return_3m,
        "ma_50": get_last("sma_50"),
        "ma_200": get_last("sma_200"),
        "sma_20": get_last("sma_20"),
        "ema_20": get_last("ema_20"),
        "rsi_14": get_last("rsi_14"),
        "macd": get_last("macd"),
        "macd_signal": get_last("macd_signal"),
        "macd_hist": get_last("macd_diff"),
        "stoch_k": get_last("stoch_k"),
        "stoch_d": get_last("stoch_d"),
        "williams_r": get_last("williams_r"),
        "atr_14": get_last("atr_14"),
        "bb_width": get_last("bb_width"),
        "bb_pos": get_last("bb_pos"),
        "obv": get_last("obv"),
        "cmf": get_last("cmf"),
        "mfi_14": get_last("mfi_14"),
        "volume_last": vol_last,
        "avg_volume_20": avg_vol_20,
        "support": support,
        "resistance_near": resistance_near,
        "resistance_major": resistance_major,
        "data_confidence": "High" if len(close) >= 200 else ("Medium" if len(close) >= 60 else "Low"),
    }
    snapshot["technical_summary_text"] = technical_summary_for_llm(out, last_n=5)
    return snapshot
