"""
llm_pipeline.py — LLM prompt runner.

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

from llm_config import load_llm_config

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
from archetype import classify_archetype, archetype_to_dict
from catalyst_pipeline import build_catalyst_inputs
from report_validation import validate_report


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
    "You must balance growth, profitability, balance sheet quality, valuation, and risk; "
    "do NOT mechanically favor high growth or strong share price momentum when the balance sheet is weak, "
    "cash burn is high, or shareholder dilution is severe. "
    "For highly speculative companies with weak or nonexistent profitability, you must treat the risk as elevated "
    "even if technical momentum is strong. "
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

PROMPT_1D_CATALYSTS = """You are evaluating catalysts for {TICKER} based ONLY on the structured evidence provided.

You MUST NOT invent catalysts that are not supported by:
- the external news items,
- the inferred fundamental/technical catalysts,
- or explicitly described industry/sector context.

Your task is to classify and enrich catalysts into a structured list.

=== COMPANY NEWS (raw items) ===
{COMPANY_NEWS_JSON}

=== INFERRED FUNDAMENTAL CATALYSTS (from Python metrics) ===
{FUNDAMENTAL_INFERRED_JSON}

=== INFERRED TECHNICAL / MARKET-IMPLIED CATALYSTS (from Python snapshot) ===
{TECHNICAL_INFERRED_JSON}

=== INDUSTRY / ECOSYSTEM CANDIDATES (may be empty in this version) ===
{INDUSTRY_CANDIDATES_JSON}

Return a JSON object with a single field:
{{
  "structured_catalysts": [
    {{
      "category": "company | industry | speculative",
      "description": string,
      "direction": "positive | negative | neutral",
      "source_type": "reported | inferred | market_implied | speculative",
      "time_horizon": "near_term | medium_term | long_term",
      "confidence": "Low | Medium | High",
      "impact_level": "Low | Medium | High",
      "is_already_priced": "Yes | No | Partially",
      "monitoring_trigger": string,
      "evidence_summary": string
    }}
  ]
}}

Be conservative when assigning confidence and impact_level. If evidence is weak or mixed,
you must lower confidence instead of overstating the catalyst.
Use direction "neutral" only when impact has no clear positive/negative bias or uncertainty is too high
to call; pair with confidence "Low" when the catalyst is highly uncertain.
"""


PROMPT_2 = """Synthesize the fundamental, technical, archetype, and catalyst analyses below for {TICKER}.
Identify where fundamentals and technicals diverge or converge, and produce a unified investment outlook
with a price target matrix, a single, hard rating, structured signals, and structured catalysts.

=== COMPANY ARCHETYPE (from Python rules) ===
{ARCHETYPE_JSON}

=== ANNUAL FUNDAMENTAL ANALYSIS ===
{OUTPUT_FROM_PROMPT_1A}

=== QUARTERLY DEVIATION ANALYSIS ===
{OUTPUT_FROM_PROMPT_1B}

=== TECHNICAL ANALYSIS ===
{OUTPUT_FROM_PROMPT_1C}

=== STRUCTURED CATALYST INPUTS (Prompt 1D output) ===
{OUTPUT_FROM_PROMPT_1D}

You must follow these constraints:
- The "rating" field is MANDATORY and must be exactly one of: Overweight | Equal-weight | Hold | Underweight | Reduce.
- The rating should reflect 6–12 month risk/reward, not just narrative attractiveness.
- If you are uncertain, default to "Hold" rather than omitting or leaving the rating empty.
- Do NOT invent catalysts that are not present in or implied by the upstream evidence.
- Do NOT automatically treat technical pullbacks as buying opportunities; only do so when fundamentals, valuation, and catalysts support it.
- Do NOT infer liquidity distress from working capital alone unless clearly justified by the statements or news.
- Avoid exaggerated or promotional language; favor precise, balanced phrasing.
- If evidence is mixed, reduce confidence instead of forcing a strong directional call.

You MUST always output:
- a "rating"
- a balanced discussion of risks, catalysts, and valuation
- a non-empty "key_catalysts" array (use at least an empty array if no clear catalysts exist)
- a "signals" array covering the main investment dimensions
- a "structured_catalysts" array classifying catalysts by type.

The "signals" array is a structured scoring system. For each signal, you must specify:
- "type": one of ["growth", "profitability", "valuation", "balance_sheet", "momentum", "industry_catalyst", "company_catalyst", "speculative_catalyst", "risk", "management_execution"]
- "strength": one of ["strong_positive", "positive", "cautious", "strong_negative"]
- "reason": a concise natural-language explanation linking back to fundamentals, technicals, or catalysts.

The "structured_catalysts" array is a structured catalyst representation. You MUST use only catalysts
that are supported by upstream evidence (Prompts 1A/1B/1C/1D). This is not a place to speculate freely.
For each catalyst, you must output:
- "category": one of ["company", "industry", "speculative"]
- "description": a concise description of the catalyst
- "direction": "positive", "negative", or "neutral" (use neutral when impact is unclear or evidence insufficient; use confidence Low when highly uncertain)
- "source_type": "reported", "inferred", "market_implied", or "speculative"
- "time_horizon": one of ["near_term", "medium_term", "long_term"]
- "confidence": one of ["Low", "Medium", "High"]
- "impact_level": one of ["Low", "Medium", "High"] indicating impact on the investment case.
- "is_already_priced": "Yes", "No", or "Partially"
- "monitoring_trigger": a concrete future event or datapoint that would confirm/negate the catalyst
- "evidence_summary": 1–2 sentences summarizing the underlying evidence.

The "price_target_matrix" array MUST contain exactly three objects, one for each scenario: Bear, Consensus, Bull (in that order).
Each scenario MUST be analyzed independently, with its own price_target_range and key_assumption; you may NOT merge scenarios into a single combined row.
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
 "rating_dimensions": {{
   "expected_return_6_12m": "strong_upside | moderate_upside | flat | moderate_downside | strong_downside",
   "thesis_conviction": "Low | Medium | High",
   "balance_sheet_risk": "Low | Medium | High",
   "catalyst_quality": "Low | Medium | High",
   "valuation_stretch": "cheap | fair | expensive"
 }},
 "signals": [
   {{
     "type": "growth | profitability | valuation | balance_sheet | momentum | industry_catalyst | company_catalyst | speculative_catalyst | risk | management_execution",
     "strength": "strong_positive | positive | cautious | strong_negative",
     "reason": string
   }}
 ],
 "price_target_matrix": [
   {{
    "scenario": "Bear",
    "timeline": string,
    "price_target_range": {{ "low": number, "high": number }},
    "key_assumption": string
  }},
  {{
    "scenario": "Consensus",
    "timeline": string,
    "price_target_range": {{ "low": number, "high": number }},
    "key_assumption": string
  }},
  {{
    "scenario": "Bull",
     "timeline": string,
     "price_target_range": {{ "low": number, "high": number }},
     "key_assumption": string
   }}
 ],
 "overall_outlook": string,
 "key_risks": [string],
 "key_catalysts": [string],
 "structured_catalysts": [
   {{
     "category": "company | industry | speculative",
     "description": string,
     "direction": "positive | negative | neutral",
     "source_type": "reported | inferred | market_implied | speculative",
     "time_horizon": "short_term | medium_term | long_term",
     "confidence": "Low | Medium | High",
     "impact_level": "Low | Medium | High",
     "is_already_priced": "Yes | No | Partially",
     "monitoring_trigger": string,
     "evidence_summary": string
   }}
 ]
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


def build_unique_output_path(ticker: str, analysis_date: str) -> Path:
    """
    Build an outputs path that does NOT overwrite previous runs.
    We include the current time to the minute in the filename, and if a file
    still exists (multiple runs within the same minute), we append a suffix.

    Examples (for ticker=T, analysis_date=2026-03-09, time ~14:37):
    - T_2026-03-09_1437_report.json
    - T_2026-03-09_1437_report_run2.json
    """
    out_dir = ensure_outputs_dir()
    time_tag = dt.datetime.now().strftime("%H%M")
    base = f"{ticker}_{analysis_date}_{time_tag}_report"
    candidate = out_dir / f"{base}.json"
    run_idx = 2
    while candidate.exists():
        candidate = out_dir / f"{base}_run{run_idx}.json"
        run_idx += 1
    return candidate


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
    """Enforce rating to a canonical English value; set rationale if missing.

    If the raw rating is missing or cannot be mapped, fall back to a conservative
    default of 'Hold'.
    """
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


def rating_from_dimensions(report: Dict[str, Any]) -> str:
    """
    Map structured rating dimensions into a canonical rating.
    Used as a stabilizing layer so the final rating does not rely solely
    on free-form judgment.
    """
    dims = report.get("rating_dimensions") or {}
    expected = dims.get("expected_return_6_12m")
    conviction = dims.get("thesis_conviction")
    bs_risk = dims.get("balance_sheet_risk")
    catalyst_q = dims.get("catalyst_quality")
    valuation = dims.get("valuation_stretch")

    # Base score from expected return
    score = 0
    if expected == "strong_upside":
        score += 2
    elif expected == "moderate_upside":
        score += 1
    elif expected == "flat":
        score += 0
    elif expected == "moderate_downside":
        score -= 1
    elif expected == "strong_downside":
        score -= 2

    # Adjust for conviction
    if conviction == "High":
        score *= 1.2
    elif conviction == "Low":
        score *= 0.8

    # Penalize balance sheet risk and expensive valuation
    if bs_risk == "High":
        score -= 1.0
    if valuation == "expensive":
        score -= 0.5
    elif valuation == "cheap":
        score += 0.5

    # Boost for strong catalyst quality
    if catalyst_q == "High":
        score += 0.5
    elif catalyst_q == "Low":
        score -= 0.5

    # Map numeric score to discrete rating
    if score >= 1.5:
        return "Overweight"
    if 0.5 <= score < 1.5:
        return "Equal-weight"
    if -0.5 < score < 0.5:
        return "Hold"
    if -1.5 < score <= -0.5:
        return "Underweight"
    return "Reduce"


def normalize_price_target_matrix(report: Dict[str, Any]) -> None:
    """
    Light normalization for price_target_matrix:
    - Keep at most one entry per scenario (Bear / Consensus / Bull)
    - Order them as [Bear, Consensus, Bull] if present

    It does NOT fabricate new scenarios or copy assumptions; each regime
    must be analyzed separately by the LLM.
    """
    matrix = report.get("price_target_matrix")
    if not isinstance(matrix, list) or not matrix:
        return

    scenarios = ["Bear", "Consensus", "Bull"]
    by_scenario: Dict[str, Dict[str, Any]] = {}

    for row in matrix:
        if not isinstance(row, dict):
            continue
        scen = str(row.get("scenario", ""))
        if scen in scenarios and scen not in by_scenario:
            by_scenario[scen] = row

    if not by_scenario:
        return

    # Rebuild in canonical order, keeping only scenarios the model actually provided.
    report["price_target_matrix"] = [
        by_scenario[s] for s in scenarios if s in by_scenario
    ]


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

    # ---------- Archetype (Prompt 0 equivalent, rule-based) ----------
    meta_df = annual_out.get("meta", pd.DataFrame())
    archetype_obj = classify_archetype(meta_df, annual_metrics)
    archetype_json = safe_json_dumps(archetype_to_dict(archetype_obj))

    # ---------- Catalyst inputs (for Prompt 1D) ----------
    catalyst_inputs = build_catalyst_inputs(
        ticker=ticker,
        analysis_date=analysis_date,
        fundamentals=annual_out,
        tech_snapshot=tech_snap,
    )
    catalyst_inputs_json = safe_json_dumps(catalyst_inputs)

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
    p1d = PROMPT_1D_CATALYSTS.format(
        TICKER=ticker,
        COMPANY_NEWS_JSON=safe_json_dumps(catalyst_inputs.get("company_news", [])),
        FUNDAMENTAL_INFERRED_JSON=safe_json_dumps(catalyst_inputs.get("fundamental_inferred", [])),
        TECHNICAL_INFERRED_JSON=safe_json_dumps(catalyst_inputs.get("technical_inferred", [])),
        INDUSTRY_CANDIDATES_JSON=safe_json_dumps(catalyst_inputs.get("industry_candidates", [])),
    )

    # ---------- LLM client ----------
    cfg = load_llm_config()

    if cfg.backend == "openai":
        client = OpenAI(base_url=cfg.openai_base_url, api_key=cfg.openai_api_key)
        runner = OpenAIRunner(model=cfg.openai_model, client=client)
    elif cfg.backend == "gemini-vertex":
        # Vertex AI in your GCP project (e.g. to consume education credits).
        # Authentication is via ADC, configured outside this script:
        #   gcloud auth application-default login
        if not cfg.gcp_project:
            raise ValueError("GOOGLE_CLOUD_PROJECT must be set when LLM_BACKEND=gemini-vertex")
        client = genai.Client(vertexai=True, project=cfg.gcp_project, location=cfg.gcp_location)
        runner = GeminiRunner(model=cfg.gemini_model, client=client)
    else:
        # Default: Gemini API via API key (GEMINI_API_KEY)
        client = genai.Client()
        runner = GeminiRunner(model=cfg.gemini_model, client=client)

    # ---------- 1A/1B/1C/1D parallel ----------
    results: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {
            ex.submit(runner.run_json, p1a, 0.2): "1A",
            ex.submit(runner.run_json, p1b, 0.2): "1B",
            ex.submit(runner.run_json, p1c, 0.2): "1C",
            ex.submit(runner.run_json, p1d, 0.2): "1D",
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
        ARCHETYPE_JSON=archetype_json,
        OUTPUT_FROM_PROMPT_1A=safe_json_dumps(results.get("1A", {})),
        OUTPUT_FROM_PROMPT_1B=safe_json_dumps(results.get("1B", {})),
        OUTPUT_FROM_PROMPT_1C=safe_json_dumps(results.get("1C", {})),
        OUTPUT_FROM_PROMPT_1D=safe_json_dumps(results.get("1D", {})),
    )
    final_report = runner.run_json(p2, temperature=0.2)

    # Use structured dimensions (if present) to derive a stable rating,
    # then normalize aliases / fallbacks.
    try:
        if isinstance(final_report, dict) and final_report.get("rating_dimensions"):
            final_report.setdefault("rating_raw_model", final_report.get("rating"))
            final_report["rating"] = rating_from_dimensions(final_report)
    except Exception:
        # If anything goes wrong, fall back to whatever the model provided.
        pass

    normalize_rating(final_report)
    normalize_price_target_matrix(final_report)

    # Basic schema validation (non-fatal, but surfaces issues early)
    validation_errors: list = []
    try:
        validation_errors = validate_report(final_report)
    except Exception:
        validation_errors = ["validate_report raised an exception"]

    # Attach archetype and catalyst inputs for auditability
    final_report.setdefault("archetype", {})
    final_report["archetype"] = json.loads(archetype_json)
    final_report.setdefault("catalyst_inputs_debug", catalyst_inputs)
    # Always write validation so the report structure is consistent (errors may be empty)
    final_report.setdefault("validation", {})
    final_report["validation"]["errors"] = validation_errors

    # Annotate report with the LLM backend/model used for full transparency
    meta = final_report.get("report_metadata") or {}
    meta["llm_backend"] = cfg.backend
    if cfg.backend == "openai":
        meta["llm_model"] = cfg.openai_model
    else:
        meta["llm_model"] = cfg.gemini_model
    final_report["report_metadata"] = meta

    # ---------- Save ----------
    out_path = build_unique_output_path(ticker, analysis_date)
    out_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== FINAL REPORT (JSON) ===")
    print(json.dumps(final_report, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
