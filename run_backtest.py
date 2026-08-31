#!/usr/bin/env python3
"""
End-to-end demo: universe -> events -> calibration -> backtest -> report.

    python run_backtest.py

Everything runs on synthetic data (see fda_alpha/data/synth.py for why the
resulting numbers validate the machinery and not the alpha).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from fda_alpha.backtest.calibrate import build_design, ridge_fit, walk_forward_weights
from fda_alpha.backtest.costs import ExecutionConfig
from fda_alpha.backtest.engine import BacktestConfig, Backtester, PriceBook
from fda_alpha.backtest.report import bootstrap_sharpe_ci, print_report, summarize
from fda_alpha.data.synth import (
    TICKER_META, build_universe, generate_events, listed_spans,
    market_context_fn, simulate_prices,
)
from fda_alpha.ontology import Ontology
from fda_alpha.readthrough import KernelParams, ReadThroughKernel
from fda_alpha.signal import SignalConfig, SignalEngine
from fda_alpha.surprise import prior_pos, signed_surprise


def peer_abnormal_returns(prices, onto, events, window_min=90) -> pd.DataFrame:
    """
    Realized peer reaction, sector-adjusted. This is the regression target for
    calibration. Sector adjustment matters: without it you fit beta, not
    read-through.
    """
    tickers = sorted(prices.keys())
    idx = prices[tickers[0]].index
    mat = np.column_stack([prices[t]["close"].to_numpy() for t in tickers])
    logret = np.diff(np.log(mat), axis=0, prepend=np.log(mat[:1]))
    sector = np.median(logret, axis=1)

    rows = {}
    bars = int(window_min / 5)
    for ev in events:
        k = idx.searchsorted(ev.t_wire)
        k2 = min(k + bars, len(idx) - 1)
        if k >= len(idx) - 1:
            continue
        r = {}
        for ci, t in enumerate(tickers):
            raw = float(np.log(mat[k2, ci] / mat[k, ci]))
            r[t] = raw - float(sector[k:k2].sum())
        rows[ev.event_id] = r
    return pd.DataFrame.from_dict(rows, orient="index")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=6.0)
    ap.add_argument("--events", type=int, default=850)
    ap.add_argument("--capital", type=float, default=5_000_000.0)
    ap.add_argument("--no-own-leg", action="store_true",
                    help="disable the issuing-name leg (recommended: see README)")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    end = datetime(2025, 12, 31, tzinfo=timezone.utc)
    start = end - timedelta(days=int(365 * args.years))

    print("building universe and simulating market ...")
    programs, indications, links = build_universe()
    onto = Ontology(programs, indications, links)
    events = generate_events(programs, start, end, n_events=args.events, seed=args.seed)
    prices = simulate_prices(programs, indications, links, events, start, end)
    book = PriceBook(prices)
    mkt_fn = market_context_fn(prices)

    print(f"  {len(programs)} programs, {len(onto.universe())} tickers, "
          f"{len(events)} events, {len(prices[onto.universe()[0]])} bars/ticker")

    # ---------------- calibration on the first half ----------------------
    split = start + (end - start) * 0.5
    train_events = [e for e in events if e.t_wire < split]
    test_events = [e for e in events if e.t_wire >= split]

    kernel = ReadThroughKernel(onto, KernelParams())
    abn = peer_abnormal_returns(prices, onto, train_events)
    rows = build_design(
        train_events, onto, kernel, abn,
        vols={t: m[1] for t, m in TICKER_META.items()},
        caps={t: m[0] for t, m in TICKER_META.items()},
    )
    prior = KernelParams()
    try:
        w, diag = ridge_fit(rows, alpha=4.0, nonneg_prior=prior.vector())
        w = np.maximum(w, 0.0)
        print("\nCALIBRATED CHANNEL WEIGHTS (in-sample, first half)")
        print(f"  n={diag['n']}  R2={diag['r2']:.4f}")
        for c in diag["weights"]:
            print(f"    {c:14s} w={diag['weights'][c]:+.3f}  t={diag['tstats'][c]:+.2f}"
                  f"   (prior {getattr(prior, 'w_' + c):.2f})")
        fitted = KernelParams.from_vector(w, prior)
    except ValueError as e:
        print(f"  calibration skipped: {e}")
        fitted = prior

    # ---------------- out-of-sample backtest ------------------------------
    variants = [
        ("A  naive: own leg + peers, any session, no liquidity gate",
         dict(trade_own_leg=True, trade_peer_leg=True,
              allow_extended_hours=True, min_adv_usd=0.0)),
        ("B  peers only, any session, no liquidity gate",
         dict(trade_own_leg=False, trade_peer_leg=True,
              allow_extended_hours=True, min_adv_usd=0.0)),
        ("C  peers only + edge-over-cost gate (1.5x)",
         dict(trade_own_leg=False, trade_peer_leg=True,
              allow_extended_hours=True, min_edge_over_cost=1.5)),
        ("D  peers only + edge-over-cost gate (2.5x)",
         dict(trade_own_leg=False, trade_peer_leg=True,
              allow_extended_hours=True, min_edge_over_cost=2.5)),
        ("E  own leg too, edge-over-cost 2.5x",
         dict(trade_own_leg=True, trade_peer_leg=True,
              allow_extended_hours=True, min_edge_over_cost=2.5)),
    ]

    spans = listed_spans(start, end)
    rows, detail = [], {}
    for label, kw in variants:
        eng = SignalEngine(onto, ReadThroughKernel(onto, fitted),
                           SignalConfig(own_leg_enabled=kw["trade_own_leg"]))
        bt = Backtester(onto, eng, book, ExecutionConfig(),
                        BacktestConfig(capital=args.capital, seed=args.seed, **kw))
        tr = bt.run(test_events, mkt_fn, listed_on=spans)
        detail[label] = tr
        s_ = summarize(tr, args.capital) if not tr.empty else {"n_trades": 0}
        rows.append({
            "variant": label,
            "trades": s_.get("n_trades", 0),
            "net_pnl": s_.get("net_pnl", 0.0),
            "ret_%": s_.get("total_return_pct", 0.0),
            "sharpe": s_.get("sharpe_annualized", float("nan")),
            "max_dd_%": s_.get("max_drawdown_pct", float("nan")),
            "dir_acc": s_.get("directional_accuracy", float("nan")),
            "IC": s_.get("information_coefficient", float("nan")),
            "avg_slip_bps": s_.get("avg_slippage_bps", float("nan")),
        })
        if not tr.empty:
            lo, hi = bootstrap_sharpe_ci(tr, args.capital)
            rows[-1]["sharpe_5%"], rows[-1]["sharpe_95%"] = lo, hi

    print("\n" + "=" * 100)
    print("OUT-OF-SAMPLE VARIANT COMPARISON")
    print("=" * 100)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    best_label = "D  peers only + edge-over-cost gate (2.5x)"
    trades = detail[best_label]
    # Derive the header from best_label so the two cannot drift apart again.
    print(f"\n\n### FULL REPORT FOR VARIANT {best_label.split()[0]} ###")
    if trades.empty:
        print("no trades")
        return
    print_report(trades, args.capital)

    trades.to_csv("trades.csv", index=False)
    print("\nwrote trades.csv")


if __name__ == "__main__":
    main()
