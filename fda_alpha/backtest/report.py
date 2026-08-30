"""
Reporting. Deliberately reports the things that make the strategy look worse,
because those are the ones that predict live performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(trades: pd.DataFrame, capital: float) -> dict:
    if trades.empty:
        return {"n_trades": 0}

    live = trades[trades["rejected"] == ""].copy()
    if live.empty:
        return {"n_trades": 0, "n_rejected": len(trades)}

    live["date"] = pd.to_datetime(live["t_exec"]).dt.tz_convert("UTC").dt.date
    daily = live.groupby("date")["pnl"].sum()
    ret = daily / capital

    ann_factor = np.sqrt(252)
    sharpe = float(ret.mean() / ret.std() * ann_factor) if ret.std() > 0 else float("nan")

    equity = (1 + ret).cumprod()
    dd = float((equity / equity.cummax() - 1).min())

    hit = float((live["pnl"] > 0).mean())
    gross_pnl = float((live["side"] * live["realized_move"] * live["notional"]).sum())
    net_pnl = float(live["pnl"].sum())
    cost_drag = gross_pnl - net_pnl

    # Correlation between predicted and realized — the honest test of the model
    ic = float(np.corrcoef(live["expected_move"], live["side"] * live["realized_move"])[0, 1]) \
        if len(live) > 3 else float("nan")
    dir_acc = float((np.sign(live["expected_move"]) ==
                     np.sign(live["realized_move"])).mean())

    return {
        "n_trades": int(len(live)),
        "n_rejected": int((trades["rejected"] != "").sum()),
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "cost_drag": cost_drag,
        "cost_drag_pct_of_gross": float(cost_drag / gross_pnl * 100) if gross_pnl else float("nan"),
        "total_return_pct": float(net_pnl / capital * 100),
        "sharpe_annualized": sharpe,
        "max_drawdown_pct": dd * 100,
        "hit_rate": hit,
        "directional_accuracy": dir_acc,
        "information_coefficient": ic,
        "avg_pnl_bps_of_notional": float(live["pnl_bps_of_notional"].mean()),
        "avg_slippage_bps": float(live["slippage_bps"].mean()),
        "trading_days": int(len(daily)),
    }


def bootstrap_sharpe_ci(
    trades: pd.DataFrame, capital: float, n_boot: int = 2000,
    block_days: int = 5, seed: int = 3,
) -> tuple[float, float]:
    """
    Stationary block bootstrap CI on the Sharpe ratio.

    Event strategies have very few independent bets — a hundred trades can be
    twenty events. A point Sharpe from that sample means nothing without an
    interval, and the interval is almost always embarrassingly wide. Report it
    anyway; the alternative is believing a number you should not.
    """
    live = trades[trades["rejected"] == ""]
    if live.empty:
        return float("nan"), float("nan")
    d = live.copy()
    d["date"] = pd.to_datetime(d["t_exec"]).dt.tz_convert("UTC").dt.date
    daily = d.groupby("date")["pnl"].sum() / capital
    x = daily.to_numpy()
    if len(x) < 10:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(x) / block_days))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, len(x), n_blocks)
        samp = np.concatenate([
            np.take(x, range(s, s + block_days), mode="wrap") for s in starts
        ])[: len(x)]
        sd = samp.std()
        out[b] = samp.mean() / sd * np.sqrt(252) if sd > 0 else 0.0
    return float(np.percentile(out, 5)), float(np.percentile(out, 95))


def by_leg(trades: pd.DataFrame) -> pd.DataFrame:
    """
    The single most informative table in the report.

    If own-leg P&L dwarfs peer-leg P&L, the backtest is capturing gaps it could
    not have captured live and you should distrust the whole result. A healthy
    version of this strategy has most of its net P&L in the peer leg.
    """
    live = trades[trades["rejected"] == ""]
    if live.empty:
        return pd.DataFrame()
    g = live.groupby("leg").agg(
        n=("pnl", "size"),
        net_pnl=("pnl", "sum"),
        avg_bps=("pnl_bps_of_notional", "mean"),
        hit_rate=("pnl", lambda s: float((s > 0).mean())),
        avg_slip_bps=("slippage_bps", "mean"),
        avg_notional=("notional", "mean"),
    )
    return g.round(2)


def by_session(trades: pd.DataFrame) -> pd.DataFrame:
    live = trades[trades["rejected"] == ""]
    if live.empty:
        return pd.DataFrame()
    return live.groupby("session").agg(
        n=("pnl", "size"),
        net_pnl=("pnl", "sum"),
        avg_bps=("pnl_bps_of_notional", "mean"),
        avg_slip_bps=("slippage_bps", "mean"),
    ).round(2)


def channel_attribution(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Which read-through channel actually earned the money.

    Attributes each peer trade's P&L across channels in proportion to each
    channel's share of the predicted move. Tells you whether the competitive-
    displacement logic is real or whether you are just being paid for sector
    beta.
    """
    live = trades[(trades["rejected"] == "") & (trades["leg"] == "peer")].copy()
    if live.empty:
        return pd.DataFrame()

    ch_cols = [c for c in live.columns if c.startswith("ch_")]
    if not ch_cols:
        return pd.DataFrame()

    W = live[ch_cols].fillna(0.0).to_numpy()
    denom = np.abs(W).sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    share = np.abs(W) / denom

    attributed = share * live["pnl"].to_numpy()[:, None]
    out = pd.DataFrame({
        "channel": [c[3:] for c in ch_cols],
        "attributed_pnl": attributed.sum(axis=0),
        "avg_abs_contribution": np.abs(W).mean(axis=0),
        "n_active": (np.abs(W) > 1e-9).sum(axis=0),
    })
    return out.sort_values("attributed_pnl", ascending=False).reset_index(drop=True)


def rejection_reasons(trades: pd.DataFrame) -> pd.Series:
    r = trades[trades["rejected"] != ""]["rejected"]
    return r.value_counts()


def print_report(trades: pd.DataFrame, capital: float) -> None:
    s = summarize(trades, capital)
    lo, hi = bootstrap_sharpe_ci(trades, capital)
    s["sharpe_ci90_low"] = lo
    s["sharpe_ci90_high"] = hi
    print("=" * 68)
    print("BACKTEST SUMMARY")
    print("=" * 68)
    for k, v in s.items():
        print(f"  {k:32s} {v:>14,.4f}" if isinstance(v, float) else f"  {k:32s} {v:>14}")

    print("\n" + "-" * 68)
    print("P&L BY LEG  (own = issuing name, peer = read-through complex)")
    print("-" * 68)
    print(by_leg(trades).to_string())

    print("\n" + "-" * 68)
    print("P&L BY EXECUTION SESSION")
    print("-" * 68)
    print(by_session(trades).to_string())

    ca = channel_attribution(trades)
    if not ca.empty:
        print("\n" + "-" * 68)
        print("READ-THROUGH CHANNEL ATTRIBUTION (peer leg only)")
        print("-" * 68)
        print(ca.to_string(index=False))

    rr = rejection_reasons(trades)
    if len(rr):
        print("\n" + "-" * 68)
        print("REJECTED ORDERS")
        print("-" * 68)
        print(rr.to_string())
