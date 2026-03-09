# Ticker Analysis LLM Pipeline

A pipeline that pulls fundamental and price data for a given ticker, engineers features, and uses an LLM (Gemini) to produce a structured **fundamental + technical synthesis report** in JSON.

**End goal:** A fully automated workflow to analyze and record a ticker’s fundamentals and technicals, with reports that can be versioned and compared over time. The design is intended to evolve with LLM capabilities (better models, prompts, and tool use) so the pipeline stays current.

---

## What it does

1. **Data** — Fetches from Yahoo Finance:
   - Annual financials: income statement, balance sheet, cash flow (last 5 fiscal years)
   - Quarterly financials (last 5 quarters)
   - Price history for technicals

2. **Features** — Computes:
   - Fundamental metrics: margins, leverage, growth rates, etc.
   - Technical snapshot (via [ta](https://github.com/bukosabino/ta)): see **Technical indicators** below. A high-info-density **technical summary** is then produced for the LLM (not the full indicator table).

3. **LLM analysis** — Sends the above to Gemini in three parallel prompts, then one synthesis:
   - **1A** — Annual fundamental signals (top 5)
   - **1B** — Quarterly deviation / acceleration / reversal signals
   - **1C** — Technical signals from the computed snapshot
   - **2** — Synthesis: fundamental vs technical stance, combined signals, price target matrix (Bear / Consensus / Bull), risks, catalysts

4. **Output** — Writes a single JSON report to `outputs/{TICKER}_{date}_report.json`, including a **hard rating** (Overweight | Equal-weight | Hold | Underweight | Reduce) with rationale.

---

## Setup

- **Python:** 3.10+ (tested on 3.13).

- **Install dependencies:** (includes [ta](https://github.com/bukosabino/ta) for technical indicators)
  ```bash
  pip install -r requirements.txt
  ```

- **Secrets (never commit):**  
  Copy `.env.example` to `.env` and set your Gemini API key:
  ```bash
  cp .env.example .env
  # Edit .env: set GEMINI_API_KEY=your_key
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
| **`fundamental_pipeline.py`** | **Fundamental data source** — Yahoo Finance annual/quarterly fetch, statement merge, anchor_dates, derived metrics |
| **`technical_pipeline.py`** | **Technical data source** — ta-based indicator computation, technical summary for LLM |
| **`llm_pipeline.py`** | **LLM layer** — Pulls fundamental + technical data, builds prompts, runs 1A/1B/1C + synthesis, writes JSON report |
| `.env.example` | Template for local `.env` (copy to `.env`, add `GEMINI_API_KEY`) |
| `requirements.txt` | Python dependencies |
| `outputs/` | Generated reports (gitignored) |

---

## Keeping it current

- The pipeline is built to **stay updated as LLMs improve**: swap models via `GEMINI_MODEL` (or code), refine prompts, and later add tool use / agents if useful.
- Extend with more data sources, metrics, or prompt stages as needed; the goal is a single, repeatable flow from ticker → stored analysis.
