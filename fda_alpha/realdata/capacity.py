"""
Capacity study: how much capital the peer complex actually absorbs.

The strategy's binding constraint is not signal — it is extended-hours
liquidity in mid-cap biotech, which is thin. This module quantifies that
using the *existing* cost engine (`backtest.costs`), so the numbers are
consistent with the backtest rather than a separate hand-wave.

Two views a quant trader is expected to produce:

* **Per-name pre-market capacity** — the notional you can push into each peer's
  pre-market book before the cost engine's participation cap bites, and the
  slippage you eat at that size.
* **The impact curve / ceiling** — as you scale notional on a single name,
  round-trip slippage grows like sqrt(participation); the capacity ceiling is
  the notional at which that cost swallows a typical peer read-through move.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from ..backtest.costs import (
    ExecutionConfig, Session, max_tradeable_notional, slippage_bps,
)
from .ontology_real import REAL_TICKER_META


def _spread_bps(cap_bn: float) -> float:
    """Cap-based spread proxy, identical to engine._spread_proxy."""
    return float(np.clip(120.0 / max(cap_bn, 0.05) ** 0.5, 4.0, 400.0))


def per_name_capacity(
    adv_usd: dict[str, float],
    cfg: ExecutionConfig | None = None,
    session: str = Session.PRE,
) -> list[dict]:
    """Pre-market max tradeable notional and its slippage, per ticker."""
    cfg = cfg or ExecutionConfig()
    rows = []
    for tk, adv in sorted(adv_usd.items(), key=lambda kv: -kv[1]):
        cap = REAL_TICKER_META.get(tk, (1.0, True, 100.0))[0]
        max_notional = max_tradeable_notional(adv, session, cfg)
        slip = slippage_bps(
            max_notional, adv, quoted_spread_bps=_spread_bps(cap),
            session=session, cfg=cfg, gap_pct=0.0, is_gapping_name=False,
        )
        rows.append({
            "ticker": tk, "cap_bn": cap, "adv_usd": adv,
            "premkt_capacity_usd": max_notional,
            "roundtrip_slip_bps": 2.0 * slip,
        })
    return rows


def impact_curve(
    ticker: str,
    adv_usd: float,
    notionals: list[float] | None = None,
    cfg: ExecutionConfig | None = None,
    session: str = Session.PRE,
) -> list[dict]:
    """Round-trip slippage vs order size on one name (the ceiling shape)."""
    cfg = cfg or ExecutionConfig()
    cap = REAL_TICKER_META.get(ticker, (1.0, True, 100.0))[0]
    if notionals is None:
        notionals = [1e4, 2.5e4, 5e4, 1e5, 2.5e5, 5e5, 1e6]
    out = []
    for n in notionals:
        slip = slippage_bps(
            n, adv_usd, quoted_spread_bps=_spread_bps(cap),
            session=session, cfg=cfg, gap_pct=0.0, is_gapping_name=False,
        )
        out.append({
            "notional_usd": n,
            "participation": min(n / max(adv_usd * cfg.premarket_liquidity_frac, 1.0), 1.0),
            "roundtrip_slip_bps": 2.0 * slip,
        })
    return out


def print_capacity(
    adv_usd: dict[str, float],
    typical_edge_bps: float = 200.0,
    cfg: ExecutionConfig | None = None,
) -> None:
    """Human-readable capacity report for the whole peer complex."""
    cfg = cfg or ExecutionConfig()
    rows = per_name_capacity(adv_usd, cfg)
    total = sum(r["premkt_capacity_usd"] for r in rows)

    print("=" * 74)
    print("PRE-MARKET CAPACITY BY NAME  (cost engine: 6% ADV pre-mkt, 5% cap)")
    print("=" * 74)
    print(f"{'ticker':<7}{'cap$bn':>8}{'ADV $m':>10}"
          f"{'premkt cap $k':>15}{'RT slip bps':>13}")
    for r in rows:
        print(f"{r['ticker']:<7}{r['cap_bn']:>8.0f}{r['adv_usd']/1e6:>10.1f}"
              f"{r['premkt_capacity_usd']/1e3:>15.1f}{r['roundtrip_slip_bps']:>13.0f}")
    print("-" * 74)
    print(f"complex pre-market capacity per event: ${total/1e6:.2f}M "
          f"(sum of per-name caps; you rarely trade all legs)")

    # Ceiling on the thinnest liquid small-cap peer.
    small = min(adv_usd, key=lambda t: adv_usd[t])
    print(f"\nIMPACT CURVE on the thinnest name ({small}, "
          f"ADV ${adv_usd[small]/1e6:.1f}m):")
    print(f"  a {typical_edge_bps:.0f}bps expected read-through is eaten when "
          f"round-trip slippage crosses it")
    for pt in impact_curve(small, adv_usd[small], cfg=cfg):
        flag = "  <- edge gone" if pt["roundtrip_slip_bps"] >= typical_edge_bps else ""
        print(f"  ${pt['notional_usd']/1e3:>6.0f}k  "
              f"participation {pt['participation']*100:>5.1f}%  "
              f"RT slip {pt['roundtrip_slip_bps']:>5.0f}bps{flag}")
    print("\n-> Capacity is bounded by pre-market ADV, not by signal. The peer "
          "reaction-lag\n   also compresses as more desks automate it. Position "
          "limits and the kill\n   switch are load-bearing, not decoration.")
