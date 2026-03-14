"""
fundamental_pipeline.py — Fundamental data source.

Fetches from Yahoo Finance: annual/quarterly financials (income statement,
balance sheet, cash flow), aligns to anchor_dates, computes derived metrics
(leverage, growth, margins, etc.). Used by llm_pipeline to feed fundamental
data into the LLM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 1) Math / safe numeric handling
# ============================================================

def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Safe division: 0/NaN or inf in denominator -> NaN."""
    n = pd.to_numeric(numer, errors="coerce")
    d = pd.to_numeric(denom, errors="coerce").replace(0, np.nan)
    out = n / d
    return out.where(np.isfinite(out), np.nan)


def yoy_pct_change(series: pd.Series) -> pd.Series:
    """YoY pct change; no implicit fill (missing stays missing)."""
    s = pd.to_numeric(series, errors="coerce")
    return s.pct_change(fill_method=None)


def pick_first_available(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Return first column present in df.columns; else NaN Series."""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(index=df.index, data=np.nan, dtype="float64")


# ============================================================
# 2) yfinance: fetch and merge multiple sources per statement
# ============================================================

def safe_get_df(tk: yf.Ticker, name: str) -> pd.DataFrame | None:
    """Safely get a single DataFrame attribute from the ticker."""
    val = getattr(tk, name, None)
    if isinstance(val, pd.DataFrame) and not val.empty:
        return val
    return None


def merge_statement_sources(tk: yf.Ticker, candidates: list[str]) -> pd.DataFrame | None:
    """
    For each statement type, try multiple yfinance sources and merge with
    combine_first (fill missing from left with right). Returns items x dates.
    """
    dfs: list[pd.DataFrame] = []
    for name in candidates:
        df = safe_get_df(tk, name)
        if df is not None:
            dfs.append(df)

    if not dfs:
        return None

    # Align to union of index and columns for combine_first
    all_index = dfs[0].index
    all_cols = dfs[0].columns
    for d in dfs[1:]:
        all_index = all_index.union(d.index)
        all_cols = all_cols.union(d.columns)

    aligned = []
    for d in dfs:
        aligned.append(d.reindex(index=all_index, columns=all_cols))

    merged = aligned[0]
    for d in aligned[1:]:
        merged = merged.combine_first(d)

    # Drop all-NaN rows/cols
    merged = merged.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return merged if not merged.empty else None


def statement_to_rows(raw: pd.DataFrame | None) -> pd.DataFrame:
    """
    Convert raw (items x dates) to rows (dates x items).
    Index = fiscal period end dates; columns = line items.
    """
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df.columns = pd.to_datetime(df.columns, errors="coerce")
    df = df.loc[:, df.columns.notna()]
    if df.empty:
        return pd.DataFrame()

    df = df.T.sort_index()
    df.insert(0, "FiscalYear", df.index.year.astype(int))
    return df


# ============================================================
# 3) anchor_dates: use actual returned dates; backfill if < periods
# ============================================================

def build_anchor_dates(
    income_idx: pd.DatetimeIndex,
    balance_idx: pd.DatetimeIndex,
    cash_idx: pd.DatetimeIndex,
    periods: int = 5,
) -> pd.DatetimeIndex:
    """
    anchor_dates = latest `periods` from union of the three statements.
    If union has fewer than periods, backfill by year (same month/day) before earliest.
    """
    union_dates = sorted(set(income_idx).union(set(balance_idx)).union(set(cash_idx)))
    if not union_dates:
        return pd.DatetimeIndex([])

    anchor = list(union_dates[-periods:])

    if len(anchor) < periods:
        oldest = anchor[0]
        mm, dd = oldest.month, oldest.day
        y = oldest.year
        while len(anchor) < periods:
            y -= 1
            anchor.insert(0, pd.Timestamp(year=y, month=mm, day=dd))

    return pd.DatetimeIndex(anchor)


def coverage_report(raw_rows: pd.DataFrame, anchor_dates: pd.DatetimeIndex, label: str) -> pd.DataFrame:
    """Report which anchor_dates are missing (no row) for this statement."""
    have = set(raw_rows.index) if not raw_rows.empty else set()
    missing = [d for d in anchor_dates if d not in have]
    return pd.DataFrame([{
        "Statement": label,
        "AvailablePeriods": 0 if raw_rows.empty else len(raw_rows.index),
        "AnchorPeriods": len(anchor_dates),
        "MissingAnchorDates": ", ".join([d.strftime("%Y-%m-%d") for d in missing]) if missing else "",
    }])


# ============================================================
# 4) Main: annual only, fixed N periods, compute metrics
# ============================================================

def fetch_annual_5_periods_with_metrics(ticker: str, periods: int = 5) -> dict[str, pd.DataFrame]:
    ticker = ticker.strip().upper()
    tk = yf.Ticker(ticker)

    # ---- (1) Fetch and merge each statement from all candidate sources ----
    income_raw = merge_statement_sources(tk, ["income_stmt", "financials"])
    balance_raw = merge_statement_sources(tk, ["balance_sheet", "balancesheet"])
    cash_raw = merge_statement_sources(tk, ["cashflow", "cash_flow"])

    # ---- (2) Convert to rows = dates ----
    income_rows_raw = statement_to_rows(income_raw)
    balance_rows_raw = statement_to_rows(balance_raw)
    cash_rows_raw = statement_to_rows(cash_raw)

    if income_rows_raw.empty and balance_rows_raw.empty and cash_rows_raw.empty:
        meta = pd.DataFrame([{"Ticker": ticker, "Note": "No annual statements returned by yfinance."}])
        return {"meta": meta, "income_statement": pd.DataFrame(), "balance_sheet": pd.DataFrame(),
                "cash_flow": pd.DataFrame(), "metrics": pd.DataFrame(), "coverage": pd.DataFrame()}

    # ---- (3) anchor_dates from actual returned dates only ----
    anchor_dates = build_anchor_dates(
        income_rows_raw.index if not income_rows_raw.empty else pd.DatetimeIndex([]),
        balance_rows_raw.index if not balance_rows_raw.empty else pd.DatetimeIndex([]),
        cash_rows_raw.index if not cash_rows_raw.empty else pd.DatetimeIndex([]),
        periods=periods,
    )

    # ---- (4) Reindex to anchor_dates (fixed N rows; missing -> NaN) ----
    income = income_rows_raw.reindex(anchor_dates)
    balance = balance_rows_raw.reindex(anchor_dates)
    cash = cash_rows_raw.reindex(anchor_dates)

    # Ensure FiscalYear on padded rows
    for df in (income, balance, cash):
        if "FiscalYear" in df.columns:
            df["FiscalYear"] = df.index.year.astype(int)
        else:
            df.insert(0, "FiscalYear", df.index.year.astype(int))

    # ---- (5) Pick line items and compute metrics (division -> NaN where invalid) ----
    # Balance inputs
    total_assets = pick_first_available(balance, ["Total Assets", "TotalAssets"])
    total_equity = pick_first_available(
        balance,
        ["Total Stockholder Equity", "TotalStockholderEquity", "Stockholders Equity", "StockholdersEquity",
         "Total Equity Gross Minority Interest"],
    )
    total_debt = pick_first_available(balance, ["Total Debt", "TotalDebt"])
    if total_debt.isna().all():
        ltd = pick_first_available(balance, ["Long Term Debt", "LongTermDebt"])
        std = pick_first_available(balance, ["Short Long Term Debt", "ShortLongTermDebt", "Current Debt", "CurrentDebt"])
        total_debt = pd.to_numeric(ltd, errors="coerce") + pd.to_numeric(std, errors="coerce")

    # Income inputs
    total_revenue = pick_first_available(income, ["Total Revenue", "TotalRevenue", "Revenue"])
    operating_income = pick_first_available(income, ["Operating Income", "OperatingIncome"])
    net_income = pick_first_available(income, ["Net Income", "NetIncome", "Net Income Common Stockholders", "NetIncomeCommonStockholders"])
    pretax_income = pick_first_available(income, ["Pretax Income", "PretaxIncome", "Income Before Tax", "IncomeBeforeTax"])
    tax_provision = pick_first_available(income, ["Tax Provision", "TaxProvision", "Income Tax Expense", "IncomeTaxExpense"])

    # Cash-flow inputs for investment-oriented metrics
    cfo = pick_first_available(
        cash,
        [
            "Total Cash From Operating Activities",
            "TotalCashFromOperatingActivities",
            "Net Cash Provided by Operating Activities",
            "NetCashProvidedByOperatingActivities",
        ],
    )
    capex = pick_first_available(
        cash,
        [
            "Capital Expenditures",
            "CapitalExpenditures",
            "Purchase Of Property Plant And Equipment",
            "PurchaseOfPropertyPlantAndEquipment",
        ],
    )
    rd_expense = pick_first_available(
        income,
        ["Research Development", "ResearchAndDevelopment"],
    )

    total_revenue_num = pd.to_numeric(total_revenue, errors="coerce")
    operating_income_num = pd.to_numeric(operating_income, errors="coerce")
    net_income_num = pd.to_numeric(net_income, errors="coerce")
    pretax_income_num = pd.to_numeric(pretax_income, errors="coerce")
    tax_provision_num = pd.to_numeric(tax_provision, errors="coerce")
    total_assets_num = pd.to_numeric(total_assets, errors="coerce")
    total_debt_num = pd.to_numeric(total_debt, errors="coerce")
    total_equity_num = pd.to_numeric(total_equity, errors="coerce")
    cfo_num = pd.to_numeric(cfo, errors="coerce")
    capex_num = pd.to_numeric(capex, errors="coerce")
    rd_num = pd.to_numeric(rd_expense, errors="coerce")

    free_cash_flow = cfo_num - capex_num
    revenue_growth_yoy = yoy_pct_change(total_revenue_num)
    net_income_growth_yoy = yoy_pct_change(net_income_num)

    capex_intensity = safe_div(-capex_num, total_revenue_num)
    rd_intensity = safe_div(rd_num, total_revenue_num)

    net_debt = total_debt_num  # we do not always have reliable cash balance history

    metrics = pd.DataFrame(
        {
            "FiscalYear": anchor_dates.year.astype(int),
            "TotalRevenue": total_revenue_num,
            "OperatingIncome": operating_income_num,
            "NetIncome": net_income_num,
            "PretaxIncome": pretax_income_num,
            "TaxProvision": tax_provision_num,
            "TotalAssets": total_assets_num,
            "TotalDebt": total_debt_num,
            "TotalEquity": total_equity_num,
            "Debt_to_Equity": safe_div(total_debt_num, total_equity_num),
            "Asset_Growth_YoY": yoy_pct_change(total_assets_num),
            "Debt_Growth_YoY": yoy_pct_change(total_debt_num),
            "Operating_Margin": safe_div(operating_income_num, total_revenue_num),
            "Net_Margin": safe_div(net_income_num, total_revenue_num),
            "Effective_Tax_Rate": safe_div(tax_provision_num, pretax_income_num),
            # Investment-oriented metrics
            "Revenue_Growth_YoY": revenue_growth_yoy,
            "NetIncome_Growth_YoY": net_income_growth_yoy,
            "Free_Cash_Flow": free_cash_flow,
            "Capex_Intensity": capex_intensity,
            "RD_Intensity": rd_intensity,
            "Net_Debt_Proxy": net_debt,
        },
        index=anchor_dates,
    )

    # ---- (6) Coverage: which anchor rows are entirely missing ----
    cov = pd.concat(
        [
            coverage_report(income_rows_raw, anchor_dates, "IncomeStatement"),
            coverage_report(balance_rows_raw, anchor_dates, "BalanceSheet"),
            coverage_report(cash_rows_raw, anchor_dates, "CashFlow"),
        ],
        ignore_index=True,
    )

    # Key-field missing counts (explains NaN in metrics)
    key_field_missing = pd.DataFrame([{
        "AnchorDates": ", ".join([d.strftime("%Y-%m-%d") for d in anchor_dates]),
        "TotalRevenue_isna": int(pd.to_numeric(total_revenue, errors="coerce").isna().sum()),
        "TotalAssets_isna": int(pd.to_numeric(total_assets, errors="coerce").isna().sum()),
        "NetIncome_isna": int(pd.to_numeric(net_income, errors="coerce").isna().sum()),
    }])

    info = getattr(tk, "info", {}) or {}
    meta = pd.DataFrame([{
        "Ticker": ticker,
        "ShortName": info.get("shortName"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Currency": info.get("currency"),
        "MarketCap": info.get("marketCap"),
        "Exchange": info.get("exchange"),
        "Country": info.get("country"),
        "PeriodsRequested": periods,
        "AnchorDates": ", ".join([d.strftime("%Y-%m-%d") for d in anchor_dates]),
        "IncomeRawPeriods": 0 if income_raw is None else income_raw.shape[1],
        "BalanceRawPeriods": 0 if balance_raw is None else balance_raw.shape[1],
        "CashRawPeriods": 0 if cash_raw is None else cash_raw.shape[1],
    }])

    return {
        "meta": meta,
        "coverage": cov,
        "key_field_missing": key_field_missing,
        "income_statement": income,
        "balance_sheet": balance,
        "cash_flow": cash,
        "metrics": metrics,
    }


# ============================================================
# 5) Run
# ============================================================

if __name__ == "__main__":
    ticker_in = input("Enter ticker (e.g., ORCL, AAPL, MSFT): ").strip().upper()
    if not ticker_in:
        raise ValueError("Ticker cannot be empty.")

    out = fetch_annual_5_periods_with_metrics(ticker_in, periods=5)

    print("\n=== META ===")
    print(out["meta"].to_string(index=False))

    print("\n=== COVERAGE (missing entire rows for anchor dates) ===")
    print(out["coverage"].to_string(index=False))

    print("\n=== KEY FIELD MISSING COUNTS (explains why NaN can still appear) ===")
    print(out["key_field_missing"].to_string(index=False))

    print("\n=== INCOME STATEMENT (annual; anchor-based 5 periods; missing -> NaN) ===")
    print(out["income_statement"])

    print("\n=== BALANCE SHEET (annual; anchor-based 5 periods; missing -> NaN) ===")
    print(out["balance_sheet"])

    print("\n=== CASH FLOW (annual; anchor-based 5 periods; missing -> NaN) ===")
    print(out["cash_flow"])

    print("\n=== METRICS (annual; anchor-based 5 periods; missing -> NaN) ===")
    print(out["metrics"].round(6))
