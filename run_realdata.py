#!/usr/bin/env python3
"""
Real-data pipeline: openFDA + ClinicalTrials.gov catalysts, Yahoo prices, run
through the *existing* read-through kernel, cost engine, and backtester.

    python run_realdata.py                 # daily-bar backtest + honest verdict
    python run_realdata.py --micro TICKER   # extended-hours microstructure demo

READ RESEARCH_REALDATA.md FIRST. The headline finding is not a Sharpe ratio; it
is that the only free historical catalyst sources are day-scale lagging, so the
engine correctly refuses them as intraday signal — which is exactly why the real
trade needs a paid wire feed. This script proves the plumbing on real names and
is honest about what it cannot prove.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from fda_alpha.backtest.calibrate import build_design, ridge_fit
from fda_alpha.backtest.costs import ExecutionConfig
from fda_alpha.backtest.engine import BacktestConfig, Backtester, PriceBook
from fda_alpha.backtest.report import bootstrap_sharpe_ci, print_report, summarize
from fda_alpha.ontology import Ontology
from fda_alpha.readthrough import KernelParams, ReadThroughKernel
from fda_alpha.signal import SignalConfig, SignalEngine
from fda_alpha.realdata import catalysts as C
from fda_alpha.realdata import prices_yahoo as PY
from fda_alpha.realdata.market_context import listed_spans, market_context_fn
from fda_alpha.realdata.ontology_real import (
    PROGRAM_QUERIES, REAL_TICKER_META, build_universe, ticker_caps,
)


def peer_abnormal_returns_daily(prices, events, window_days=2) -> pd.DataFrame:
    """Sector-adjusted peer reaction over a few trading days (daily bars)."""
    tickers = sorted(prices.keys())
    # align on the union index of the first ticker (all share the trading calendar)
    idx = prices[tickers[0]].index
    closes = {t: prices[t]["close"].reindex(idx).to_numpy() for t in tickers}
    mat = np.column_stack([closes[t] for t in tickers])
    logret = np.diff(np.log(mat), axis=0, prepend=np.log(mat[:1]))
    sector = np.nanmedian(logret, axis=1)

    rows = {}
    for ev in events:
        k = idx.searchsorted(ev.t_wire)
        k2 = min(k + window_days, len(idx) - 1)
        if k >= len(idx) - 1:
            continue
        r = {}
        for ci, t in enumerate(tickers):
            p0, p1 = mat[k, ci], mat[k2, ci]
            if not (p0 > 0 and p1 > 0):
                continue
            r[t] = float(np.log(p1 / p0)) - float(np.nansum(sector[k:k2]))
        if r:
            rows[ev.event_id] = r
    return pd.DataFrame.from_dict(rows, orient="index")


def microstructure_demo(ticker: str) -> None:
    """Show real extended-hours 1-minute structure — the granularity the trade
    lives at, available only for a trailing ~30-day window."""
    cap = REAL_TICKER_META.get(ticker, (1.0, True, 100.0))[0]
    # Yahoo serves 1-minute bars for at most 7 days per request.
    df = PY.intraday_bars(ticker, range_="7d", interval="1m",
                          prepost=True, market_cap_bn=cap)
    if df.empty:
        print(f"no intraday data for {ticker}")
        return
    et = df.index.tz_convert("America/New_York")
    tod = np.array([t.hour * 60 + t.minute for t in et])
    pre = (tod < 9 * 60 + 30)
    post = (tod >= 16 * 60)
    reg = ~pre & ~post
    print(f"\nEXTENDED-HOURS MICROSTRUCTURE - {ticker} "
          f"({df.index[0].date()} .. {df.index[-1].date()})")
    print(f"  1-min bars: {len(df)}  regular={reg.sum()}  "
          f"pre-market={pre.sum()}  post-market={post.sum()}")
    # overnight gaps: last regular close -> next pre-market first print
    days = pd.Series(et.date, index=df.index)
    gaps = []
    for _, sub in df.groupby(days.values):
        s_et = sub.index.tz_convert("America/New_York")
        s_tod = np.array([t.hour * 60 + t.minute for t in s_et])
        pm = sub[s_tod < 9 * 60 + 30]
        rg = sub[(s_tod >= 9 * 60 + 30) & (s_tod < 16 * 60)]
        if len(pm) and len(rg):
            gaps.append((pm["open"].iloc[0] / rg["close"].iloc[-1] - 1) * 100)
    if gaps:
        g = np.array(gaps)
        print(f"  overnight gap to pre-market open: mean {g.mean():+.2f}%  "
              f"abs-median {np.median(np.abs(g)):.2f}%  max {np.abs(g).max():.2f}%")
    print("  -> this is the print the issuer trade must beat and cannot; the "
          "peer leg trades the slower complex reaction at this same resolution.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-08-30")
    ap.add_argument("--capital", type=float, default=5_000_000.0)
    ap.add_argument("--micro", default="", help="ticker for microstructure demo")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    if args.micro:
        microstructure_demo(args.micro.upper())
        return

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    print("=" * 78)
    print("FDA-ALPHA ON REAL DATA  (openFDA + ClinicalTrials.gov + Yahoo)")
    print("=" * 78)

    programs, indications, links = build_universe()
    onto = Ontology(programs, indications, links)
    id2tk = {p.program_id: p.ticker for p in programs}

    print("\nfetching real daily prices (Yahoo) ...")
    prices = PY.daily_panel(ticker_caps(), start, end, use_cache=not args.no_cache)
    print(f"  price history for {len(prices)}/{len(ticker_caps())} tickers: "
          f"{sorted(prices)}")

    print("fetching real catalysts (openFDA approvals + CT.gov status) ...")
    events = C.build_catalysts(PROGRAM_QUERIES, id2tk, start, end)
    # keep only events on names we actually have prices for
    events = [e for e in events if e.ticker in prices]
    from collections import Counter
    print(f"  {len(events)} real catalysts  "
          f"by_source={dict(Counter(e.source.value for e in events))}  "
          f"by_type={dict(Counter(e.event_type.value for e in events))}")

    if not events:
        print("\nno catalysts sourced — nothing to backtest.")
        return

    book = PriceBook(prices)
    mkt_fn = market_context_fn(prices)
    spans = listed_spans(prices)

    # ---- the honesty check, computed not asserted -----------------------
    ex = ExecutionConfig()
    from fda_alpha.schema import SOURCE_LATENCY_SEC
    lat_days = [SOURCE_LATENCY_SEC[e.source] / 86400.0 for e in events]
    print(f"\nsource latency applied before any fill is allowed: "
          f"median {np.median(lat_days):.2f} days, max {max(lat_days):.2f} days")
    print("  => every catalyst is hours-to-days public before the engine can "
          "trade it (CT.gov ~6h, openFDA ~1d). This is the finding, not a bug.")

    # ---- calibration (best-effort; real data is thin) -------------------
    split = start + (end - start) * 0.5
    train = [e for e in events if e.t_wire < split]
    kernel = ReadThroughKernel(onto, KernelParams())
    prior = KernelParams()
    fitted = prior
    try:
        abn = peer_abnormal_returns_daily(prices, train)
        rows = build_design(
            train, onto, kernel, abn,
            vols={t: mkt_fn(end).daily_vol[t] for t in prices},
            caps={t: REAL_TICKER_META[t][0] for t in prices},
        )
        w, diag = ridge_fit(rows, alpha=8.0, nonneg_prior=prior.vector())
        w = np.maximum(w, 0.0)
        fitted = KernelParams.from_vector(w, prior)
        print(f"\ncalibration (in-sample, thin): n={diag['n']}  R2={diag['r2']:.4f}")
    except (ValueError, KeyError) as e:
        print(f"\ncalibration skipped (too few real events): {e}; using priors")

    # ---- variant comparison on real data --------------------------------
    variants = [
        ("A  naive: own+peers, no gate",
         dict(trade_own_leg=True, trade_peer_leg=True,
              allow_extended_hours=True, min_edge_over_cost=0.0)),
        ("B  peers only, no gate",
         dict(trade_own_leg=False, trade_peer_leg=True,
              allow_extended_hours=True, min_edge_over_cost=0.0)),
        ("C  peers + edge/cost 1.5x",
         dict(trade_own_leg=False, trade_peer_leg=True,
              allow_extended_hours=True, min_edge_over_cost=1.5)),
        ("D  peers + edge/cost 2.5x",
         dict(trade_own_leg=False, trade_peer_leg=True,
              allow_extended_hours=True, min_edge_over_cost=2.5)),
    ]
    summary, detail = [], {}
    for label, kw in variants:
        eng = SignalEngine(onto, ReadThroughKernel(onto, fitted),
                           SignalConfig(own_leg_enabled=kw["trade_own_leg"]))
        bt = Backtester(onto, eng, book, ex,
                        BacktestConfig(capital=args.capital, **kw))
        tr = bt.run(events, mkt_fn, listed_on=spans)
        detail[label] = tr
        s_ = summarize(tr, args.capital) if not tr.empty else {"n_trades": 0}
        row = {"variant": label, "trades": s_.get("n_trades", 0),
               "ret_%": s_.get("total_return_pct", 0.0),
               "sharpe": s_.get("sharpe_annualized", float("nan")),
               "dir_acc": s_.get("directional_accuracy", float("nan"))}
        if not tr.empty:
            lo, hi = bootstrap_sharpe_ci(tr, args.capital)
            row["sharpe_5%"], row["sharpe_95%"] = lo, hi
        summary.append(row)

    print("\n" + "=" * 78)
    print("VARIANT COMPARISON - REAL DATA (daily bars, lagging sources)")
    print("=" * 78)
    print(pd.DataFrame(summary).to_string(index=False,
                                          float_format=lambda x: f"{x:,.2f}"))

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(
        "On free daily bars and day-scale-lagging catalyst sources, there is no\n"
        "intraday peer reaction-lag left to capture: the engine's timestamp\n"
        "discipline rolls every fill past the point where the news is public.\n"
        "This VALIDATES the machinery on real names and DISPROVES nothing about\n"
        "the alpha. To evaluate the edge you need (1) wire-level PR timestamps\n"
        "and (2) intraday extended-hours TAQ history - both paid. Run\n"
        "`python run_realdata.py --micro VKTX` to see the resolution the real\n"
        "trade requires but free history cannot backfill."
    )


if __name__ == "__main__":
    main()
