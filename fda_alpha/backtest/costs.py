"""
Execution realism.

If you take one thing from this repo, take this file. A naive event backtest
that marks the fill at the post-news price will show a Sharpe of 4 and will
lose money live, for four reasons that are all modeled here:

1. TIMING. The large majority of FDA and topline-data press releases are
   issued outside 09:30-16:00 ET, deliberately. The headline move happens in
   a pre-market print you cannot participate in at size.

2. HALTS. Material regulatory news on a listed issuer frequently triggers a
   T1 news-pending halt. During the halt there is no market. You resume at the
   post-halt auction price, which already contains the news.

3. GAP SLIPPAGE. On the primary name, the first accessible print is not the
   settled price. Effective slippage scales with the size of the gap itself.

4. BORROW. The short side of this strategy is small-cap biotech. Locate
   availability and fee are binding constraints, not footnotes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone

# US equity session boundaries in ET, expressed as UTC offsets handled by the
# caller's calendar. Kept simple here: the engine passes ET-localized times.
PREMARKET_OPEN = time(4, 0)
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
POSTMARKET_CLOSE = time(20, 0)


class Session(str):
    CLOSED = "closed"
    PRE = "premarket"
    REGULAR = "regular"
    POST = "postmarket"


def session_of(t_et: datetime) -> str:
    if t_et.weekday() >= 5:
        return Session.CLOSED
    tod = t_et.time()
    if REGULAR_OPEN <= tod < REGULAR_CLOSE:
        return Session.REGULAR
    if PREMARKET_OPEN <= tod < REGULAR_OPEN:
        return Session.PRE
    if REGULAR_CLOSE <= tod < POSTMARKET_CLOSE:
        return Session.POST
    return Session.CLOSED


@dataclass
class ExecutionConfig:
    # System latency: wire -> parse -> score -> order ack.
    system_latency_sec: float = 1.8

    # Probability the primary name is halted on a material regulatory event,
    # and how long the halt lasts.
    halt_prob_own_material: float = 0.55
    halt_duration_min: float = 55.0

    # Slippage model coefficients (basis points).
    spread_capture: float = 0.60        # fraction of quoted spread paid
    impact_coef_bps: float = 42.0       # * sqrt(participation)
    gap_slippage_coef: float = 0.16     # * |gap| ; only on the gapping name

    # Extended-hours penalty multipliers on spread and impact.
    premarket_spread_mult: float = 3.2
    postmarket_spread_mult: float = 3.8
    premarket_liquidity_frac: float = 0.06   # ADV available in pre-market

    # Borrow
    max_borrow_fee_bps_annual: float = 3000.0
    short_locate_prob_smallcap: float = 0.55

    # Risk-side caps
    max_participation: float = 0.05     # of interval volume
    commission_bps: float = 0.5


def next_tradeable_time(
    t_signal_et: datetime,
    cfg: ExecutionConfig,
    allow_extended: bool = True,
    halted_until: datetime | None = None,
) -> tuple[datetime, str]:
    """
    First instant we could actually get a fill, and in which session.

    This is the function that kills the fantasy P&L. An event at 16:05 ET on a
    Friday is not tradeable until Monday pre-market at the earliest, by which
    point the information is fully public and reflected.
    """
    t = t_signal_et
    if halted_until is not None and halted_until > t:
        t = halted_until

    for _ in range(8):  # at most a long weekend + holiday
        sess = session_of(t)
        if sess == Session.REGULAR:
            return t, sess
        if sess in (Session.PRE, Session.POST) and allow_extended:
            return t, sess
        if sess == Session.PRE:
            # today's regular open is still ahead of us
            t = t.replace(hour=REGULAR_OPEN.hour, minute=REGULAR_OPEN.minute,
                          second=0, microsecond=0)
            return t, Session.REGULAR
        if sess == Session.POST:
            # today's open is behind us; roll to the NEXT session day. Getting
            # this wrong resolves execution backwards in time, which is the
            # most dangerous class of bug in an event backtester.
            t = (t + timedelta(days=1)).replace(
                hour=PREMARKET_OPEN.hour, minute=0, second=0, microsecond=0
            )
            continue
        # closed: roll forward to next weekday pre-market open
        t = (t + timedelta(days=1)).replace(
            hour=PREMARKET_OPEN.hour, minute=0, second=0, microsecond=0
        )
    raise RuntimeError("could not find a tradeable session")


def slippage_bps(
    notional_usd: float,
    adv_usd: float,
    quoted_spread_bps: float,
    session: str,
    cfg: ExecutionConfig,
    gap_pct: float = 0.0,
    is_gapping_name: bool = False,
) -> float:
    """
    Round-trip-agnostic one-way slippage in bps.

    The `gap_pct` term is the piece most models omit: when a name is printing
    a 45% gap, the effective cost of getting filled is a meaningful fraction
    of the gap itself, because you are crossing into a book that is being
    repriced faster than you can quote.
    """
    spread_mult = {
        Session.REGULAR: 1.0,
        Session.PRE: cfg.premarket_spread_mult,
        Session.POST: cfg.postmarket_spread_mult,
    }.get(session, 1.0)

    liq_frac = cfg.premarket_liquidity_frac if session != Session.REGULAR else 1.0
    effective_adv = max(adv_usd * liq_frac, 1.0)
    participation = min(notional_usd / effective_adv, 1.0)

    bps = cfg.spread_capture * quoted_spread_bps * spread_mult
    bps += cfg.impact_coef_bps * (participation ** 0.5)
    if is_gapping_name:
        bps += cfg.gap_slippage_coef * abs(gap_pct) * 10_000.0
    bps += cfg.commission_bps
    return float(bps)


def max_tradeable_notional(
    adv_usd: float, session: str, cfg: ExecutionConfig
) -> float:
    liq_frac = cfg.premarket_liquidity_frac if session != Session.REGULAR else 1.0
    return adv_usd * liq_frac * cfg.max_participation


def can_short(
    ticker: str,
    borrowable: bool,
    borrow_fee_bps: float,
    market_cap_bn: float,
    cfg: ExecutionConfig,
    rng=None,
) -> tuple[bool, str]:
    if not borrowable:
        return False, "no_borrow"
    if borrow_fee_bps > cfg.max_borrow_fee_bps_annual:
        return False, "borrow_too_expensive"
    if market_cap_bn < 0.5 and rng is not None:
        if rng.random() > cfg.short_locate_prob_smallcap:
            return False, "locate_failed"
    return True, "ok"


def carry_cost(borrow_fee_bps_annual: float, hold_minutes: float) -> float:
    """Fractional borrow cost over the holding period."""
    years = hold_minutes / (60 * 24 * 365)
    return (borrow_fee_bps_annual / 10_000.0) * years
