"""
Surprise: how much of this outcome was already in the price?

A phase 3 success in a program the market handicapped at 90% is worth almost
nothing. The same success at 25% is a repricing event. Trading raw outcomes
instead of surprises is the second most common way this strategy loses money
(the first is assuming you can fill on the gap).

signed_surprise ranges roughly [-1.6, +1.6] and is the input to every channel
in the read-through kernel.
"""

from __future__ import annotations

import math

from .schema import CatalystEvent, EventType, Phase

# Historical probability of technical and regulatory success by phase.
# These are broad literature base rates (BIO/Informa-style transition
# probabilities); replace with your own indication-conditional table.
BASE_POS: dict[Phase, float] = {
    Phase.PRECLIN: 0.06,
    Phase.P1: 0.10,
    Phase.P1_2: 0.14,
    Phase.P2: 0.28,
    Phase.P2_3: 0.42,
    Phase.P3: 0.55,
    Phase.REGISTRATIONAL: 0.85,
    Phase.MARKETED: 0.95,
}

# Multiplicative adjustments to the phase base rate.
INDICATION_ADJ: dict[str, float] = {
    "oncology": 0.75,
    "neurology": 0.70,
    "rare_disease": 1.30,
    "infectious_disease": 1.10,
    "cardiometabolic": 0.95,
    "immunology": 1.05,
}


def prior_pos(
    phase: Phase,
    therapeutic_area: str = "",
    has_precedented_target: bool = False,
    has_positive_ph2: bool = False,
    market_implied: float | None = None,
) -> float:
    """
    Prior probability of success.

    If `market_implied` is available — backed out from the options-implied
    straddle move around the catalyst date, or from a prediction market — use
    it and ignore the base-rate stack. It is a strictly better estimator
    because it contains everything the base rate contains plus the market's
    read on the specific program.
    """
    if market_implied is not None:
        return float(min(max(market_implied, 0.02), 0.98))

    p = BASE_POS[phase]
    p *= INDICATION_ADJ.get(therapeutic_area, 1.0)
    if has_precedented_target:
        p *= 1.35
    if has_positive_ph2:
        p *= 1.25
    return float(min(max(p, 0.02), 0.98))


def signed_surprise(event: CatalystEvent, pos: float) -> float:
    """
    Convert (outcome, prior) into signed information content.

    Uses a log-odds update rather than a linear (outcome - prior). Log-odds
    matches how prices actually behave at the tails: going from 90% to 99%
    confidence is a small move, going from 10% to 40% is a large one.
    """
    pos = min(max(pos, 0.02), 0.98)

    if event.polarity > 0:
        # Realized success: information = -log(prior)  (surprise of the event)
        info = -math.log(pos)
        # Normalize so a coin-flip prior gives ~1.0
        s = info / math.log(2.0)
    elif event.polarity < 0:
        info = -math.log(1.0 - pos)
        s = -info / math.log(2.0)
    else:
        return 0.0

    # Strength scales magnitude conditional on direction; confidence gates it.
    s *= (0.35 + 0.65 * event.strength) * event.confidence
    # Compress the tail — a 2% prior does not produce a 5.6x move.
    s = math.copysign(1.6 * math.tanh(abs(s) / 1.6), s)
    return float(s)


def event_day_vol(
    ticker_daily_vol: float,
    market_cap_usd_bn: float,
    is_binary_catalyst: bool = True,
) -> float:
    """
    Typical magnitude of a *peer's* move on an event day, used to convert a
    dimensionless channel score into a return. Not the primary name's gap.

    Small caps move multiples of their normal vol on sector catalysts; large
    caps barely move. The size term captures that.
    """
    size_mult = 1.0 + 1.6 * math.exp(-market_cap_usd_bn / 3.0)
    binary_mult = 1.9 if is_binary_catalyst else 1.0
    return float(ticker_daily_vol * size_mult * binary_mult)
