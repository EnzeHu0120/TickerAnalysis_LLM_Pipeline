# Ticker Analysis LLM Pipeline

A pipeline that pulls fundamental and price data for a given ticker, engineers features, and uses an LLM (Gemini) to produce a structured **fundamental + technical synthesis report** in JSON.

**End goal:** A fully automated workflow to analyze and record a ticker’s fundamentals and technicals, with reports that can be versioned and compared over time. The design is intended to evolve with LLM capabilities (better models, prompts, and tool use) so the pipeline stays current.

Example reports are written to the `outputs/` directory. In this repo, a reference example is:
- `(NEWEST_OUTPUT_EXAMPLE) NVDA_2026-03-13_2231_report.json`

---

## In progress (roadmap)

- **Industry / ecosystem catalysts (`industry_candidates`)**  
  - Currently always empty. Planned: populate via sector/industry news or peer-event feeds (e.g. FMP/Finnhub sector APIs) so Prompt 1D and the final report can include industry-level catalyst evidence.
- **Signal aggregation framework**  
  - Structured `signals` JSON from the LLM (growth / profitability / valuation / balance_sheet / momentum / catalysts / risk / management_execution).  
  - Python-side signal scoring with a simple scale (e.g. strong_positive = +2, positive = +1, cautious = -1, strong_negative = -2).  
- **Signal aggregation ranking**  
  - Compute a per-ticker overall score from signals and catalyst quality.  
  - Support ranking across 100+ tickers and exporting top-ideas / high-risk-high-upside lists.  
- **Backtest**  
  - Design a backtest framework to evaluate whether higher-scored names outperform (hit rate, forward returns, rank IC, drawdown, Sharpe).  
- **UI / visualization**  
  - JSON report visualization (single-ticker detail, multi-ticker comparison, ranking dashboards).  
- **Cost control**  
  - Continue to refine prompt structure (including rating and catalyst logic in `llm_pipeline.py`), model choice, and batching so that large universes (100+ tickers) remain affordable under typical LLM credit limits.

---

## What it does

1. **Data** — Fetches from Yahoo Finance:
   - Annual financials: income statement, balance sheet, cash flow (last 5 fiscal years)
   - Quarterly financials (last 5 quarters)
   - Price history for technicals

2. **Features** — Computes:
   - Fundamental metrics: margins, leverage, growth rates, etc.
   - Technical snapshot (via [ta](https://github.com/bukosabino/ta)): see **Technical indicators** below. A high-info-density **technical summary** is then produced for the LLM (not the full indicator table).

3. **LLM analysis** — Sends the above to the LLM in four parallel prompts, then one synthesis:
   - **1A** — Annual fundamental signals (top 5)
   - **1B** — Quarterly deviation / acceleration / reversal signals
   - **1C** — Technical signals from the computed snapshot
   - **1D** — Catalyst classification from upstream evidence (company news, inferred fundamental/technical catalysts; industry/ecosystem candidates are planned, see **In progress**)
   - **2** — Synthesis: archetype-aware outlook, combined signals, price target matrix (Bear / Consensus / Bull), structured catalysts (direction: positive / negative / neutral), rating, risks

4. **Output** — Writes a single JSON report to `outputs/{TICKER}_{date}_report.json`, including a **hard rating** (Overweight | Equal-weight | Hold | Underweight | Reduce) with rationale.

   **Rating (post-processing):** The report’s `rating` is not the LLM’s raw output. The pipeline saves the model’s original rating as `rating_raw_model`, then sets `rating` from the structured **rating_dimensions** (expected return, thesis conviction, balance-sheet risk, catalyst quality, valuation) via a fixed mapping. That keeps the final rating aligned with the dimensions and comparable across runs.

---

## Setup

- **Python:** 3.10+ (tested on 3.13).

- **Install dependencies:** (includes [ta](https://github.com/bukosabino/ta) for technical indicators)
  ```bash
  pip install -r requirements.txt
  ```

- **Secrets / LLM config (never commit `.env`):**  
  Copy `.env.example` to `.env` and choose your LLM backend:
  ```bash
  cp .env.example .env
  # Edit .env and set:
  #   LLM_BACKEND=gemini        # or: gemini-vertex, openai
  #   GEMINI_MODEL=...          # e.g. gemini-3.1-pro-preview
  #   GEMINI_API_KEY=...        # if using Gemini API
  #   GOOGLE_CLOUD_PROJECT=...  # if using Vertex AI
  #   OPENAI_API_KEY=...        # if using OpenAI gateway
  ```
  `.env` is in `.gitignore`; only `.env.example` (no real keys) is in the repo.

---

## Run

```bash
python llm_pipeline.py AAPL
```
Or run without arguments to be prompted for a ticker.

Reports are saved under `outputs/` (also gitignored so local runs don’t pollute the repo).

---

## Technical indicators (technical analysis)

The pipeline computes the following indicators from OHLCV price history via the [ta](https://github.com/bukosabino/ta) library. Only a compact **summary** of these (plus key levels) is sent to the LLM.

| Category   | Indicator        | Symbol / name   | Parameters / notes                    |
|-----------|-------------------|-----------------|----------------------------------------|
| **Trend** | Simple moving avg | `sma_20`, `sma_50`, `sma_200` | 20-, 50-, 200-day SMA of close        |
|           | Exponential MA   | `ema_20`        | 20-day EMA of close                    |
|           | MACD             | `macd`, `macd_signal`, `macd_diff` | 12/26/9; diff = histogram             |
| **Momentum** | RSI            | `rsi_14`        | 14-period RSI                          |
|           | Stochastic       | `stoch_k`, `stoch_d` | 14-period, 3-period smooth            |
|           | Williams %R      | `williams_r`    | 14-period (lbp=14)                     |
| **Volatility** | ATR           | `atr_14`        | 14-period Average True Range           |
|           | Bollinger Bands   | `bb_mid`, `bb_high`, `bb_low`, `bb_width`, `bb_pos` | 20-period, 2 std dev; width and % position |
| **Volume** | On-Balance Volume | `obv`          | Cumulative volume by close direction   |
|           | Chaikin Money Flow | `cmf`        | 20-period CMF                          |
|           | Money Flow Index  | `mfi_14`      | 14-period MFI                          |
|           | VWAP (rolling)   | `vwap_14`     | 14-period volume-weighted average price |
| **Other** | Support / resistance | —           | Min/max close over 60-day and full lookback |
|           | Returns           | `return_1m`, `return_3m` | 21- and 63-day price return          |

---

## Repo layout

| Path | Purpose |
|------|--------|
| **`fundamental_pipeline.py`** | **Fundamental data source** — Yahoo Finance annual/quarterly fetch, anchor-based history, investment-oriented metrics (growth, margins, leverage, FCF proxies, capex/R&D intensity) |
| **`technical_pipeline.py`** | **Technical data source** — ta-based indicator computation, technical summary for LLM, plus 52w context and trend/volatility regimes |
| **`catalyst_pipeline.py`** | **Catalyst layer** — Aggregates external news (FMP/Finnhub) + inferred fundamental/technical catalysts into structured inputs for the LLM. `industry_candidates` is currently a placeholder (empty); planned to be filled via sector/industry or peer feeds. |
| **`archetype.py`** | **Archetype layer** — Classifies the company into coarse archetypes (e.g. pre-profit deep tech, cyclical industrial, regulated utility) to guide the research lens |
| **`llm_pipeline.py`** | **LLM layer** — Pulls fundamental + technical + archetype + catalyst data, runs prompts 1A/1B/1C/1D + final synthesis, writes JSON report with rating and structured signals/catalysts |
| **`report_validation.py`** | **Schema validation** — Validates final report rating, price_target_matrix, signals, and structured_catalysts, surfacing issues for auditing |
| **`llm_config.py`** | Central LLM configuration (backend/model/project); loaded by `llm_pipeline.py` |
| `.env.example` | Template for local `.env` (copy to `.env`, add `GEMINI_API_KEY`) |
| `requirements.txt` | Python dependencies |
| `outputs/` | Generated reports (gitignored) |

---

## Keeping it current

- The pipeline is built to **stay updated as LLMs improve**: swap models via `GEMINI_MODEL` (or code), refine prompts, and later add tool use / agents if useful.
- Extend with more data sources, metrics, or prompt stages as needed; the goal is a single, repeatable flow from ticker → stored analysis, with auditable inputs (fundamentals, technicals, catalysts, archetype) behind each final rating.
