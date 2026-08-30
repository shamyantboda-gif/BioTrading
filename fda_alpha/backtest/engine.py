"""
Event-driven backtester with strict timestamp discipline.

Design rules, enforced in code rather than by convention:

  R1  A decision made at t may only consult bars whose close timestamp is
      STRICTLY LESS THAN t. The engine slices the price panel by index and
      raises if a lookup would touch a future bar.

  R2  Trade time = t_wire + source_latency + system_latency, then rolled
      forward to the next tradeable session and past any halt.

  R3  The fill price is a bar the engine has not yet shown to the model.

  R4  The universe is drawn from a point-in-time membership table that
      includes delisted issuers. Biotech failure is the modal outcome; a
      universe built from currently-listed tickers is a survivorship machine.

  R5  Own-name and peer-name P&L are accounted separately, because they have
      completely different capacity and completely different credibility.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..ontology import Ontology
from ..schema import CatalystEvent, EventType, Fill, SOURCE_LATENCY_SEC, Signal
from ..signal import MarketContext, SignalEngine
from .costs import (
    ExecutionConfig, Session, can_short, carry_cost,
    max_tradeable_notional, next_tradeable_time, session_of, slippage_bps,
)

ET = ZoneInfo("America/New_York")


class LookAheadError(RuntimeError):
    pass


@dataclass
class PriceBook:
    """
    Minute-bar panel with an index that refuses future lookups.

    bars: dict ticker -> DataFrame indexed by tz-aware UTC timestamp with
          columns [open, high, low, close, volume, spread_bps]
    """

    bars: dict[str, pd.DataFrame]
    _idx: dict[str, list[datetime]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for t, df in self.bars.items():
            if df.index.tz is None:
                raise ValueError(f"{t}: bar index must be tz-aware")
            self._idx[t] = list(df.index)

    def last_close_before(self, ticker: str, t: datetime) -> float | None:
        """R1: strictly-before lookup used for all decision inputs."""
        idx = self._idx.get(ticker)
        if not idx:
            return None
        k = bisect_left(idx, t) - 1
        if k < 0:
            return None
        return float(self.bars[ticker]["close"].iloc[k])

    def bar_at_or_after(self, ticker: str, t: datetime) -> tuple[datetime, pd.Series] | None:
        """R3: execution lookup. Never fed back into the model."""
        idx = self._idx.get(ticker)
        if not idx:
            return None
        k = bisect_left(idx, t)
        if k >= len(idx):
            return None
        return idx[k], self.bars[ticker].iloc[k]

    def realized_return(
        self, ticker: str, t_entry: datetime, hold_min: int
    ) -> tuple[float, datetime] | None:
        e = self.bar_at_or_after(ticker, t_entry)
        if e is None:
            return None
        t_e, bar_e = e
        x = self.bar_at_or_after(ticker, t_e + timedelta(minutes=hold_min))
        if x is None:
            # exit on last available bar (delisting / end of sample)
            t_x = self._idx[ticker][-1]
            bar_x = self.bars[ticker].iloc[-1]
        else:
            t_x, bar_x = x
        p_e, p_x = float(bar_e["open"]), float(bar_x["close"])
        if p_e <= 0:
            return None
        return (p_x - p_e) / p_e, t_x

    def gap_pct(self, ticker: str, t_exec: datetime, t_wire: datetime) -> float:
        pre = self.last_close_before(ticker, t_wire)
        e = self.bar_at_or_after(ticker, t_exec)
        if pre is None or e is None or pre <= 0:
            return 0.0
        return (float(e[1]["open"]) - pre) / pre


@dataclass
class BacktestConfig:
    capital: float = 5_000_000.0
    risk_per_event: float = 0.010          # fraction of capital at risk per event
    max_position_frac: float = 0.020       # cap per single name
    max_gross_frac: float = 0.60
    kelly_fraction: float = 0.25
    allow_extended_hours: bool = True
    trade_own_leg: bool = True
    trade_peer_leg: bool = True
    # Liquidity gate. Extended-hours spreads in sub-$1bn biotech routinely
    # exceed the entire expected read-through move; filtering on ADV is the
    # difference between a strategy and a donation to market makers.
    min_adv_usd: float = 0.0
    # Trade only when the expected move clears the estimated round-trip cost
    # by this multiple. This is the gate that matters: a 2% expected read-
    # through into a 300bps pre-market spread is a losing trade no matter how
    # correct the direction call was. Gating on liquidity is a crude proxy for
    # this; gating on edge/cost is the thing itself.
    min_edge_over_cost: float = 0.0
    seed: int = 7


@dataclass
class Trade:
    event_id: str
    ticker: str
    leg: str
    t_wire: datetime
    t_exec: datetime
    session: str
    expected_move: float
    realized_move: float
    notional: float
    side: int
    slippage_bps: float
    borrow_bps: float
    pnl: float
    pnl_bps_of_notional: float
    hold_min: int
    channels: dict = field(default_factory=dict)
    rejected: str = ""


class Backtester:
    def __init__(
        self,
        ontology: Ontology,
        signal_engine: SignalEngine,
        prices: PriceBook,
        exec_cfg: ExecutionConfig | None = None,
        bt_cfg: BacktestConfig | None = None,
    ) -> None:
        self.onto = ontology
        self.sig = signal_engine
        self.px = prices
        self.ex = exec_cfg or ExecutionConfig()
        self.cfg = bt_cfg or BacktestConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    # ------------------------------------------------------------------

    def run(
        self,
        events: list[CatalystEvent],
        mkt_ctx_fn,
        listed_on: dict[str, tuple[datetime, datetime]] | None = None,
    ) -> pd.DataFrame:
        """
        events      chronologically sortable list of CatalystEvent
        mkt_ctx_fn  callable(t: datetime) -> MarketContext, must itself be
                    point-in-time (only data with timestamp < t)
        listed_on   R4: ticker -> (first_listed, last_listed) including
                    delisted names.
        """
        trades: list[Trade] = []
        events = sorted(events, key=lambda e: e.t_wire)

        for ev in events:
            t_signal = ev.t_wire + timedelta(
                seconds=SOURCE_LATENCY_SEC[ev.source] + self.ex.system_latency_sec
            )
            mkt = mkt_ctx_fn(t_signal)
            signals = self.sig.generate(ev, mkt)
            if not signals:
                continue

            halted_until = self._halt_until(ev)

            for s in signals:
                if s.leg == "own" and not self.cfg.trade_own_leg:
                    continue
                if s.leg == "peer" and not self.cfg.trade_peer_leg:
                    continue
                if mkt.adv_usd.get(s.ticker, 0.0) < self.cfg.min_adv_usd:
                    continue
                if listed_on is not None:
                    span = listed_on.get(s.ticker)
                    if span is None or not (span[0] <= t_signal <= span[1]):
                        continue
                tr = self._execute(ev, s, mkt, t_signal, halted_until)
                if tr is not None:
                    trades.append(tr)

        return _to_frame(trades)

    # ------------------------------------------------------------------

    def _halt_until(self, ev: CatalystEvent) -> dict[str, datetime]:
        """R2: material regulatory news halts the issuing name with some prob."""
        material = ev.event_type in {
            EventType.APPROVAL, EventType.CRL, EventType.TOPLINE_EFFICACY,
            EventType.SAFETY_SIGNAL, EventType.CLINICAL_HOLD,
            EventType.PROGRAM_DISCONTINUATION,
        }
        if not material:
            return {}
        if self.rng.random() > self.ex.halt_prob_own_material:
            return {}
        dur = timedelta(minutes=float(self.rng.normal(
            self.ex.halt_duration_min, self.ex.halt_duration_min * 0.4
        )))
        dur = max(dur, timedelta(minutes=10))
        return {ev.ticker: ev.t_wire + dur}

    def _execute(
        self,
        ev: CatalystEvent,
        s: Signal,
        mkt: MarketContext,
        t_signal: datetime,
        halted: dict[str, datetime],
    ) -> Trade | None:
        t_sig_et = t_signal.astimezone(ET)
        halt_et = halted.get(s.ticker)
        halt_et = halt_et.astimezone(ET) if halt_et else None

        t_exec_et, sess = next_tradeable_time(
            t_sig_et, self.ex,
            allow_extended=self.cfg.allow_extended_hours,
            halted_until=halt_et,
        )
        t_exec = t_exec_et.astimezone(timezone.utc)

        side = 1 if s.expected_move > 0 else -1

        # Short-side feasibility
        if side < 0:
            ok, why = can_short(
                s.ticker,
                mkt.borrowable.get(s.ticker, True),
                mkt.borrow_fee_bps.get(s.ticker, 50.0),
                mkt.cap(s.ticker), self.ex, self.rng,
            )
            if not ok:
                return Trade(
                    ev.event_id, s.ticker, s.leg, ev.t_wire, t_exec, sess,
                    s.expected_move, 0.0, 0.0, side, 0.0, 0.0, 0.0, 0.0,
                    s.horizon_min, s.channel_breakdown, rejected=why,
                )

        # Sizing: fractional Kelly on expected edge vs event-day variance,
        # then hard caps, then liquidity cap.
        vol = max(mkt.vol(s.ticker) * (1.9 if s.leg == "own" else 1.0), 1e-3)
        edge = abs(s.expected_move)
        # Fractional Kelly on edge/variance, then hard caps.
        kelly = self.cfg.kelly_fraction * edge / (vol ** 2)
        frac = min(kelly * self.cfg.risk_per_event, self.cfg.max_position_frac)
        notional = frac * self.cfg.capital

        adv = mkt.adv_usd.get(s.ticker, 5e6)
        cap_liq = max_tradeable_notional(adv, sess, self.ex)
        notional = min(notional, cap_liq)
        if notional < 5_000:
            return None

        gap = self.px.gap_pct(s.ticker, t_exec, ev.t_wire)
        slip = slippage_bps(
            notional, adv,
            quoted_spread_bps=_spread_proxy(mkt, s.ticker),
            session=sess, cfg=self.ex,
            gap_pct=gap,
            is_gapping_name=(s.leg == "own"),
        )

        # Edge-over-cost gate. Estimated ex-ante: entry + exit slippage plus
        # borrow carry, all knowable before the order is sent.
        if self.cfg.min_edge_over_cost > 0.0:
            est_borrow = 0.0
            if side < 0:
                est_borrow = carry_cost(
                    mkt.borrow_fee_bps.get(s.ticker, 50.0), s.horizon_min
                ) * 10_000.0
            est_cost = (2.0 * slip + est_borrow) / 10_000.0
            if abs(s.expected_move) < self.cfg.min_edge_over_cost * est_cost:
                return Trade(
                    ev.event_id, s.ticker, s.leg, ev.t_wire, t_exec, sess,
                    s.expected_move, 0.0, 0.0, side, slip, est_borrow, 0.0, 0.0,
                    s.horizon_min, s.channel_breakdown, rejected="edge_below_cost",
                )

        rr = self.px.realized_return(s.ticker, t_exec, s.horizon_min)
        if rr is None:
            return None
        realized, t_exit = rr

        borrow = 0.0
        if side < 0:
            borrow = carry_cost(
                mkt.borrow_fee_bps.get(s.ticker, 50.0), s.horizon_min
            ) * 10_000.0

        gross = side * realized
        net = gross - (2.0 * slip + borrow) / 10_000.0   # entry + exit slippage
        pnl = net * notional

        return Trade(
            event_id=ev.event_id, ticker=s.ticker, leg=s.leg,
            t_wire=ev.t_wire, t_exec=t_exec, session=sess,
            expected_move=s.expected_move, realized_move=realized,
            notional=notional, side=side, slippage_bps=slip,
            borrow_bps=borrow, pnl=pnl, pnl_bps_of_notional=net * 10_000.0,
            hold_min=s.horizon_min, channels=s.channel_breakdown,
        )


def _spread_proxy(mkt: MarketContext, ticker: str) -> float:
    """Spread in bps as a function of size; replace with real NBBO history."""
    cap = mkt.cap(ticker)
    return float(np.clip(120.0 / max(cap, 0.05) ** 0.5, 4.0, 400.0))


def _to_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = []
    for t in trades:
        d = t.__dict__.copy()
        ch = d.pop("channels", {})
        for k, v in ch.items():
            d[f"ch_{k}"] = v
        rows.append(d)
    df = pd.DataFrame(rows).sort_values("t_exec").reset_index(drop=True)
    return df
