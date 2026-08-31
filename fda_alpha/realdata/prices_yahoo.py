"""
Yahoo Finance price adapter.

Emits the same panel shape the synthetic path does — ``{ticker: DataFrame}``
with a tz-aware UTC index and columns ``[open, high, low, close, volume,
spread_bps]`` — so `backtest.engine.PriceBook` consumes it unchanged.

Two granularities, and the gap between them is the whole QT story:

* **daily** (``daily_bars``): ``period1``/``period2`` over arbitrary history,
  years deep. Enough to align real catalysts to real names and to run the cost
  engine, but far too coarse to see the intraday peer reaction-lag that IS the
  edge. A 90-minute hold on daily bars collapses to "next close".

* **intraday 1-minute incl. pre/post** (``intraday_bars``): the QT-relevant
  granularity — but Yahoo only serves it for a trailing ~30-day window. You can
  *measure* the extended-hours microstructure on recent catalysts; you cannot
  *backtest* historical ones. That trailing-window wall is not a bug in this
  adapter, it is the reason the real trade needs paid TAQ history.

Yahoo returns HTTP 429 to a bare client; a browser ``User-Agent`` fixes it.
We retry with host fallback (query1 -> query2) and exponential backoff.
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from . import cache

_HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}
_COLUMNS = ["open", "high", "low", "close", "volume", "spread_bps"]


class YahooError(RuntimeError):
    pass


def _spread_proxy_bps(market_cap_bn: float) -> float:
    """Same cap->spread proxy the engine uses, so the panel is self-consistent."""
    return float(np.clip(120.0 / max(market_cap_bn, 0.05) ** 0.5, 4.0, 400.0))


def _get(url: str, params: dict, *, max_tries: int = 4) -> dict:
    """GET a Yahoo chart URL with host fallback and 429 backoff."""
    last_exc: Exception | None = None
    for attempt in range(max_tries):
        host = _HOSTS[attempt % len(_HOSTS)]
        try:
            r = requests.get(
                url.format(host=host), params=params, headers=_HEADERS, timeout=20
            )
            if r.status_code == 429:
                _time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:  # network / HTTP error
            last_exc = exc
            _time.sleep(1.0 * (attempt + 1))
    raise YahooError(f"Yahoo request failed after {max_tries} tries: {last_exc}")


def _parse_chart(payload: dict, market_cap_bn: float) -> pd.DataFrame:
    """Yahoo chart JSON -> OHLCV DataFrame, tz-aware UTC index, sorted, deduped."""
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise YahooError(str(chart["error"]))
    results = chart.get("result") or []
    if not results:
        return pd.DataFrame(columns=_COLUMNS)
    res = results[0]
    ts = res.get("timestamp") or []
    quote = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    if not ts:
        return pd.DataFrame(columns=_COLUMNS)

    idx = pd.to_datetime(ts, unit="s", utc=True)
    df = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=idx,
    )
    # Yahoo emits null OHLC for empty bars; drop them and forward-fill nothing.
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["volume"] = df["volume"].fillna(0.0)
    df["spread_bps"] = _spread_proxy_bps(market_cap_bn)
    return df[_COLUMNS]


def daily_bars(
    symbol: str,
    start: datetime,
    end: datetime,
    market_cap_bn: float = 1.0,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Daily OHLCV over [start, end]. Deep history; coarse for intraday holds."""
    p1, p2 = int(start.timestamp()), int(end.timestamp())
    key = f"yahoo|daily|{symbol}|{p1}|{p2}"
    if use_cache:
        cached = cache.get_frame(key)
        if cached is not None:
            return cached
    payload = _get(
        "https://{host}/v8/finance/chart/" + symbol,
        {"period1": p1, "period2": p2, "interval": "1d", "events": "div,split"},
    )
    df = _parse_chart(payload, market_cap_bn)
    if use_cache and not df.empty:
        cache.put_frame(key, df)
    return df


def intraday_bars(
    symbol: str,
    range_: str = "5d",
    interval: str = "1m",
    prepost: bool = True,
    market_cap_bn: float = 1.0,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Minute bars including pre/post market for a trailing window.

    ``range_`` is capped by Yahoo: 1-minute data is only available roughly 30
    days back, 7 days per request in practice. This is the extended-hours
    microstructure you can measure but not historically backtest.
    """
    key = f"yahoo|intraday|{symbol}|{range_}|{interval}|{int(prepost)}"
    if use_cache:
        cached = cache.get_frame(key)
        if cached is not None:
            return cached
    payload = _get(
        "https://{host}/v8/finance/chart/" + symbol,
        {
            "range": range_,
            "interval": interval,
            "includePrePost": "true" if prepost else "false",
        },
    )
    df = _parse_chart(payload, market_cap_bn)
    if use_cache and not df.empty:
        cache.put_frame(key, df)
    return df


def daily_panel(
    symbols_caps: dict[str, float],
    start: datetime,
    end: datetime,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch daily bars for many tickers. Skips names Yahoo has no data for."""
    out: dict[str, pd.DataFrame] = {}
    for sym, cap in symbols_caps.items():
        try:
            df = daily_bars(sym, start, end, market_cap_bn=cap, use_cache=use_cache)
        except YahooError:
            df = pd.DataFrame(columns=_COLUMNS)
        if not df.empty:
            out[sym] = df
    return out
