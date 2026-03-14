from __future__ import annotations

"""
news_clients.py

Thin HTTP clients for external news / event feeds used by the catalyst pipeline.
All secrets are read from environment variables; this module never hardcodes keys.

Supported sources in this milestone:
- Financial Modeling Prep (FMP): company news / press releases
- Finnhub: company news

If API keys are missing or a request fails, the functions degrade gracefully and
return an empty list so the rest of the pipeline can still run.
"""

from dataclasses import dataclass
import datetime as dt
import os
from typing import Any, Dict, List

import requests


@dataclass
class NewsItem:
    source: str
    ticker: str
    published_at: str  # ISO date or datetime string
    headline: str
    summary: str
    url: str | None


def _iso_date(date: dt.date | str) -> str:
    if isinstance(date, str):
        return date
    return date.isoformat()


def fetch_fmp_company_news(
    ticker: str,
    from_date: str,
    to_date: str,
    limit: int = 50,
) -> List[NewsItem]:
    """
    Fetch company news / press releases from FMP.
    API docs: https://site.financialmodelingprep.com/developer/docs
    """
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return []

    base_url = "https://financialmodelingprep.com/api/v3/stock_news"
    params = {
        "tickers": ticker.upper(),
        "from": from_date,
        "to": to_date,
        "limit": limit,
        "apikey": api_key,
    }
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    items: List[NewsItem] = []
    for row in data or []:
        items.append(
            NewsItem(
                source="FMP",
                ticker=ticker.upper(),
                published_at=str(row.get("publishedDate") or ""),
                headline=str(row.get("title") or ""),
                summary=str(row.get("text") or ""),
                url=row.get("url"),
            )
        )
    return items


def fetch_finnhub_company_news(
    ticker: str,
    from_date: str,
    to_date: str,
    limit: int = 50,
) -> List[NewsItem]:
    """
    Fetch company news from Finnhub.
    API docs: https://finnhub.io/docs/api/company-news
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return []

    base_url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker.upper(),
        "from": from_date,
        "to": to_date,
        "token": api_key,
    }
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    items: List[NewsItem] = []
    for row in data or []:
        if len(items) >= limit:
            break
        items.append(
            NewsItem(
                source="Finnhub",
                ticker=ticker.upper(),
                published_at=dt.datetime.utcfromtimestamp(
                    row.get("datetime", 0)
                ).isoformat()
                if row.get("datetime")
                else "",
                headline=str(row.get("headline") or ""),
                summary=str(row.get("summary") or ""),
                url=row.get("url"),
            )
        )
    return items


def news_items_to_dicts(items: List[NewsItem]) -> List[Dict[str, Any]]:
    """Serialize NewsItem objects into JSON-safe dicts."""
    out: List[Dict[str, Any]] = []
    for it in items:
        out.append(
            {
                "source": it.source,
                "ticker": it.ticker,
                "published_at": it.published_at,
                "headline": it.headline,
                "summary": it.summary,
                "url": it.url,
            }
        )
    return out

