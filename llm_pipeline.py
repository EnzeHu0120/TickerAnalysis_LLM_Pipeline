"""
llm_pipeline.py — LLM prompt runner (standalone module).

Pulls fundamental data (fundamental_pipeline) and technical data (technical_pipeline),
builds user prompts, runs Gemini/OpenAI for 1A/1B/1C + synthesis (Prompt 2), outputs JSON report.

Data sources:
- Fundamental: fundamental_pipeline.py
- Technical: technical_pipeline.py
- This module: LLM client and prompt orchestration only.

Run: python llm_pipeline.py AAPL
"""

from __future__ import annotations

import os
import re
import sys
import json
import datetime as dt
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from google import genai
from google.genai import types
from openai import OpenAI

# -------------------------
# Load .env from project dir (secrets stay local, not in repo)
# -------------------------
THIS_DIR = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    load_dotenv(THIS_DIR / ".env")
except ImportError:
    pass

# -------------------------
# Data sources: fundamental + technical
# -------------------------
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import fundamental_pipeline as fund_mod
import technical_pipeline as tech_mod


# =========================
# 1) Helpers: serialize DF -> prompt string
# =========================
def df_to_csv_str(df: pd.DataFrame, max_rows: int = 12, max_cols: int = 25) -> str:
    """Compact CSV for prompt injection (keeps index)."""
    if df is None or df.empty:
        return "N/A (empty dataframe)"

    _df = df.copy()

    # Make index readable
    try:
        _df.index = pd.to_datetime(_df.index).strftime("%Y-%m-%d")
    except Exception:
        _df.index = _df.index.astype(str)

    # Cap size for token control
    if _df.shape[1] > max_cols:
        _df = _df.iloc[:, :max_cols]
    if _df.shape[0] > max_rows:
        _df = _df.tail(max_rows)

    return _df.to_csv(index=True)


def prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [f"{prefix}{c}" for c in out.columns]
    return out


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


# =========================
# 2) Technical snapshot: fetch price -> technical_pipeline
# =========================
def fetch_technical_snapshot(ticker: str, analysis_date: str, lookback_days: int = 365) -> Dict[str, Any]:
    """Fetch price history and build technical snapshot via technical_pipeline (ta + summary for LLM)."""
    tk = yf.Ticker(ticker)
    end = pd.to_datetime(analysis_date) + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=lookback_days + 30)
    hist = tk.history(start=start, end=end, interval="1d")

    if hist is None or hist.empty:
        return {"ticker": ticker, "analysis_date": analysis_date, "note": "No price history returned."}

    hist = hist.rename(columns={c: c.replace(" ", "") for c in hist.columns})
    return tech_mod.build_technical_snapshot_dict(hist, ticker, analysis_date)


# =========================
# 3) Quarterly fetch (latest 5 quarters) — from fundamental_pipeline
# =========================
def fetch_quarterly_5_periods(ticker: str, periods: int = 5) -> Dict[str, pd.DataFrame]:
    tk = yf.Ticker(ticker)

    income_raw = fund_mod.merge_statement_sources(tk, ["quarterly_income_stmt", "quarterly_financials"])
    balance_raw = fund_mod.merge_statement_sources(tk, ["quarterly_balance_sheet", "quarterly_balancesheet"])
    cash_raw = fund_mod.merge_statement_sources(tk, ["quarterly_cashflow", "quarterly_cash_flow"])

    income_rows = fund_mod.statement_to_rows(income_raw)
    balance_rows = fund_mod.statement_to_rows(balance_raw)
    cash_rows = fund_mod.statement_to_rows(cash_raw)

    def tail_n(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        return df.sort_index().tail(periods)

    return {
        "quarterly_income_statement": tail_n(income_rows),
        "quarterly_balance_sheet": tail_n(balance_rows),
        "quarterly_cash_flow": tail_n(cash_rows),
    }


# =========================
# 4) Prompt templates
# =========================
RATING_OPTIONS = "Overweight | Equal-weight | Hold | Underweight | Reduce"

SYSTEM_PROMPT = (
    "You are a professional financial analyst. "
    "Always respond in valid JSON format only — no markdown, no preamble, no extra text outside JSON. "
    "All monetary values should be raw numbers (no currency symbols). "
    'Sentiment labels must be one of: "Strongly Positive", "Positive", "Neutral", '
    '"Cautionary", "Negative", "Strongly Negative". '
    f'When outputting a rating, you MUST use exactly one of: {RATING_OPTIONS}.'
)

PROMPT_1A = """Analyze the annual balance sheet and income statement data below for {TICKER}.
Return the top 5 financial signals observed, combining both balance sheet structure and income/profitability trends.

=== ANNUAL BALANCE SHEET (last 5 fiscal years) ===
{ANNUAL_BALANCE_SHEET_DATA}

=== ANNUAL INCOME STATEMENT & CASH FLOW (last 5 fiscal years) ===
{ANNUAL_INCOME_STATEMENT_DATA}

=== DERIVED ANNUAL METRICS (computed in Python) ===
{ANNUAL_METRICS_DATA}

Return a JSON object with this exact schema:
{{
 "report_metadata": {{
   "company": string,
   "ticker": string,
   "period": string,
   "analysis_type": "Annual Fundamental Analysis"
 }},
 "financial_signals": [
   {{
     "rank": number,
     "signal": string,
     "sentiment": string,
     "observation": string,
     "key_metrics": {{ "metric_name": value }},
     "strategic_impact": string
   }}
 ],
 "summary": string
}}
"""

PROMPT_1B = """Analyze the latest quarterly balance sheet and income statement data below for {TICKER}. Focus on signals that DEVIATE FROM or AMPLIFY typical annual trends.
Identify acceleration, reversals, or anomalies.

=== QUARTERLY BALANCE SHEET (latest 5 quarters) ===
{QUARTERLY_BALANCE_SHEET_DATA}

=== QUARTERLY INCOME STATEMENT (latest 5 quarters) ===
{QUARTERLY_INCOME_STATEMENT_DATA}

Return a JSON object with this exact schema:
{{
 "report_metadata": {{
   "company": string,
   "ticker": string,
   "period": string,
   "analysis_type": "Quarterly Deviation Analysis"
 }},
 "deviation_signals": [
   {{
     "rank": number,
     "signal": string,
     "sentiment": string,
     "deviation_type": "Acceleration | Reversal | Anomaly | Confirmation",
     "observation": string,
     "quarterly_trend": string,
     "key_metrics": {{ "metric_name": value }}
   }}
 ],
 "yoy_highlights": {{
   "revenue_growth": string,
   "operating_income_growth": string,
   "interest_expense_growth": string,
   "depreciation_growth": string
 }},
 "summary": string
}}
"""

PROMPT_1C = """Using the technical snapshot below for {TICKER} as of {ANALYSIS_DATE}, generate the top 5 technical analysis signals.
The "TECHNICAL SUMMARY (for LLM)" is a compact, high-info-density summary of key indicators; the JSON snapshot contains full numeric values.
Use moving averages (SMA/EMA), RSI, MACD, Stoch, Williams %R, ATR, Bollinger Bands, volume (OBV/CMF/MFI), and support/resistance as relevant.
If data is incomplete, flag uncertainty in the relevant fields but still provide the best available technical assessment.

=== TECHNICAL SUMMARY (for LLM) ===
{TECHNICAL_SUMMARY_TEXT}

=== FULL TECHNICAL SNAPSHOT (JSON) ===
{TECHNICAL_SNAPSHOT_JSON}

Return a JSON object with this exact schema:
{{
 "report_metadata": {{
   "ticker": string,
   "analysis_date": string,
   "current_price": number,
   "analysis_type": "Technical Analysis",
   "data_confidence": "High | Medium | Low"
 }},
 "technical_signals": [
   {{
     "rank": number,
     "signal": string,
     "sentiment": string,
     "indicator_value": string,
     "observation": string,
     "action_implication": string
   }}
 ],
 "key_levels": {{
   "support": number,
   "resistance_near": number,
   "resistance_major": number,
   "ma_50": number,
   "ma_200": number
 }},
 "technical_summary": string
}}
"""

PROMPT_2 = """Synthesize the fundamental and technical analyses below for {TICKER}. Identify where fundamentals and technicals diverge or converge,
and produce a unified investment outlook with a price target matrix and a single, hard rating.

=== ANNUAL FUNDAMENTAL ANALYSIS ===
{OUTPUT_FROM_PROMPT_1A}

=== QUARTERLY DEVIATION ANALYSIS ===
{OUTPUT_FROM_PROMPT_1B}

=== TECHNICAL ANALYSIS ===
{OUTPUT_FROM_PROMPT_1C}

Return a JSON object with this exact schema. The "rating" field is MANDATORY and must be exactly one of: Overweight | Equal-weight | Hold | Underweight | Reduce.
{{
 "report_metadata": {{
   "company": string,
   "ticker": string,
   "analysis_date": string,
   "analysis_type": "Fundamental + Technical Synthesis"
 }},
 "rating": "Overweight | Equal-weight | Hold | Underweight | Reduce",
 "rating_rationale": string,
 "core_divergence": {{
   "fundamental_stance": string,
   "technical_stance": string,
   "divergence_summary": string
 }},
 "synthesized_signals": [
   {{
     "rank": number,
     "signal": string,
     "sentiment": string,
     "fundamental_driver": string,
     "technical_driver": string,
     "synthesis": string
   }}
 ],
 "price_target_matrix": [
   {{
     "scenario": "Bear | Consensus | Bull",
     "timeline": string,
     "price_target_range": {{ "low": number, "high": number }},
     "key_assumption": string
   }}
 ],
 "overall_outlook": string,
 "key_risks": [string],
 "key_catalysts": [string]
}}
"""


# =========================
# 5) LLM wrapper (JSON Mode + defensive parsing)
# =========================
def extract_json(text: str) -> Any:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


@dataclass
class GeminiRunner:
    model: str
    client: Any

    def run_json(self, prompt: str, temperature: float = 0.2) -> Any:
        cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=temperature,
        )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=cfg,
        )
        return extract_json(resp.text)


@dataclass
class OpenAIRunner:
    model: str
    client: Any

    def run_json(self, prompt: str, temperature: float = 0.2) -> Any:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        content = resp.choices[0].message.content
        if isinstance(content, list):
            text = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        else:
            text = str(content or "")
        return extract_json(text)


# =========================
# 6) Utilities
# =========================
def get_company_name(ticker: str) -> Optional[str]:
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("shortName") or info.get("longName") or info.get("displayName")
    except Exception:
        return None


def ensure_outputs_dir() -> Path:
    out_dir = THIS_DIR / "outputs"
    out_dir.mkdir(exist_ok=True)
    return out_dir


# Canonical ratings: Overweight | Equal-weight | Hold | Underweight | Reduce
RATING_ALIASES = {
    "overweight": "Overweight",
    "equal-weight": "Equal-weight",
    "equalweight": "Equal-weight",
    "hold": "Hold",
    "underweight": "Underweight",
    "reduce": "Reduce",
    "buy": "Overweight",
    "sell": "Reduce",
    "neutral": "Equal-weight",
}


def normalize_rating(report: Dict[str, Any]) -> None:
    """Enforce rating to a canonical English value; set rationale if missing."""
    raw = (report.get("rating") or "").strip()
    canonical = "Hold"
    if raw:
        raw_lower = raw.lower()
        for alias, value in RATING_ALIASES.items():
            if alias in raw_lower:
                canonical = value
                break
        else:
            if "overweight" in raw_lower or "buy" in raw_lower:
                canonical = "Overweight"
            elif "underweight" in raw_lower or "reduce" in raw_lower or "sell" in raw_lower:
                canonical = "Reduce"
            elif "equal" in raw_lower or "neutral" in raw_lower:
                canonical = "Equal-weight"
            elif "hold" in raw_lower:
                canonical = "Hold"
    report["rating"] = canonical
    if not report.get("rating_rationale"):
        report["rating_rationale"] = "Combined fundamental and technical synthesis."


# =========================
# 7) Orchestration: pull fundamental + technical -> prompts -> LLM
# =========================
def main():
    if len(sys.argv) >= 2:
        ticker = sys.argv[1].strip().upper()
    else:
        ticker = input("Enter ticker (e.g., ORCL, AAPL, MSFT, ^GSPC): ").strip().upper()

    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    analysis_date = os.getenv("ANALYSIS_DATE") or dt.date.today().isoformat()
    company_name = get_company_name(ticker)

    # ---------- Fundamental data ----------
    annual_out = fund_mod.fetch_annual_5_periods_with_metrics(ticker, periods=5)
    annual_bs = annual_out.get("balance_sheet", pd.DataFrame())
    annual_is = annual_out.get("income_statement", pd.DataFrame())
    annual_cf = annual_out.get("cash_flow", pd.DataFrame())
    annual_metrics = annual_out.get("metrics", pd.DataFrame())

    annual_bs_str = df_to_csv_str(annual_bs, max_rows=6)
    annual_is_cf = pd.concat(
        [prefix_columns(annual_is, "IS_"), prefix_columns(annual_cf, "CF_")],
        axis=1,
    )
    annual_is_cf_str = df_to_csv_str(annual_is_cf, max_rows=6)
    annual_metrics_str = df_to_csv_str(annual_metrics.round(6), max_rows=6)

    # ---------- Quarterly (latest 5) ----------
    q_out = fetch_quarterly_5_periods(ticker, periods=5)
    q_bs_str = df_to_csv_str(q_out["quarterly_balance_sheet"], max_rows=6)
    q_is_str = df_to_csv_str(q_out["quarterly_income_statement"], max_rows=6)

    # ---------- Technical data ----------
    tech_snap = fetch_technical_snapshot(ticker, analysis_date)
    tech_snap_json = safe_json_dumps(tech_snap)
    technical_summary_text = tech_snap.get("technical_summary_text", "N/A")

    # ---------- Build prompts ----------
    company_hint = f"\nCompany hint: {company_name}\n" if company_name else ""

    p1a = (company_hint + PROMPT_1A).format(
        TICKER=ticker,
        ANNUAL_BALANCE_SHEET_DATA=annual_bs_str,
        ANNUAL_INCOME_STATEMENT_DATA=annual_is_cf_str,
        ANNUAL_METRICS_DATA=annual_metrics_str,
    )
    p1b = (company_hint + PROMPT_1B).format(
        TICKER=ticker,
        QUARTERLY_BALANCE_SHEET_DATA=q_bs_str,
        QUARTERLY_INCOME_STATEMENT_DATA=q_is_str,
    )
    p1c = PROMPT_1C.format(
        TICKER=ticker,
        ANALYSIS_DATE=analysis_date,
        TECHNICAL_SUMMARY_TEXT=technical_summary_text,
        TECHNICAL_SNAPSHOT_JSON=tech_snap_json,
    )

    # ---------- LLM client ----------
    backend = os.getenv("LLM_BACKEND", "gemini").lower()
    if backend == "openai":
        base_url = os.getenv("OPENAI_BASE_URL", "https://cbsai.business.columbia.edu/api/v1")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAI(base_url=base_url)
        runner = OpenAIRunner(model=model, client=client)
    else:
        model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
        client = genai.Client()
        runner = GeminiRunner(model=model, client=client)

    # ---------- 1A/1B/1C parallel ----------
    results: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {
            ex.submit(runner.run_json, p1a, 0.2): "1A",
            ex.submit(runner.run_json, p1b, 0.2): "1B",
            ex.submit(runner.run_json, p1c, 0.2): "1C",
        }
        for fut in as_completed(futs):
            tag = futs[fut]
            try:
                results[tag] = fut.result()
            except Exception as e:
                results[tag] = {"error": str(e), "stage": tag}

    # ---------- Prompt 2 synthesis ----------
    p2 = (company_hint + PROMPT_2).format(
        TICKER=ticker,
        OUTPUT_FROM_PROMPT_1A=safe_json_dumps(results.get("1A", {})),
        OUTPUT_FROM_PROMPT_1B=safe_json_dumps(results.get("1B", {})),
        OUTPUT_FROM_PROMPT_1C=safe_json_dumps(results.get("1C", {})),
    )
    final_report = runner.run_json(p2, temperature=0.2)
    normalize_rating(final_report)

    # ---------- Save ----------
    out_dir = ensure_outputs_dir()
    out_path = out_dir / f"{ticker}_{analysis_date}_report.json"
    out_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== FINAL REPORT (JSON) ===")
    print(json.dumps(final_report, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
