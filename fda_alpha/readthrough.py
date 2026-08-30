"""
The cross-effect (read-through) kernel.

This answers the question the whole strategy is built around: a competitor
posts good data — does *my* stock go up or down?

There is no single answer, because a rival's success travels through channels
that point in opposite directions. The kernel decomposes the reaction into
five signed channels and lets the data decide their relative weights.

    VALIDATION      same biology, different patients   -> +
    DISPLACEMENT    same patients                      -> -
    CLASS_SAFETY    same biology, any patients         -> - (safety events only)
    PRECEDENT       regulatory pathway / endpoint      -> +
    ECONOMIC        royalty, milestone, equity stake   -> sign of the contract

The classic case in the docstring of this module — "competitor's drug excels,
my drug also excels, but in a different program" — resolves as:

    mechanism_sim high, indication_overlap low
    -> VALIDATION dominates, DISPLACEMENT ~ 0
    -> the stock goes UP

Flip the indication to the same one and the sign inverts, but *how far* it
inverts depends on market headroom: in a saturated market a rival's win is
close to a zero-sum transfer, while in a large undertreated market two winners
can both be right. That conditional is the `headroom` term below, and it is
the reason a constant "competitor good = me bad" rule loses money.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from .ontology import Ontology, PairRelation
from .schema import CatalystEvent, EventType, Program

CHANNELS = ("validation", "displacement", "class_safety", "precedent", "economic")


@dataclass
class KernelParams:
    """
    Channel weights. Initialized to hand-set priors, then shrunk toward
    fitted values by ridge regression on realized peer reactions
    (see backtest/calibrate.py).

    Units: each weight maps a dimensionless [0,1] channel activation to a
    fraction-of-event-day-vol move on the peer.
    """

    w_validation: float = 0.55
    w_displacement: float = 0.85
    w_class_safety: float = 1.15
    w_precedent: float = 0.35
    w_economic: float = 0.70

    # Curvature of the "am I ahead or behind?" penalty. A rival winning hurts
    # far more when they will reach market before you.
    lead_gap_scale_months: float = 18.0

    # How much of the displacement hit survives in an untapped market.
    # 0 -> headroom fully neutralizes competition; 1 -> headroom irrelevant.
    headroom_floor: float = 0.20

    # Sector beta: any large biotech catalyst nudges the whole complex.
    w_sector_beta: float = 0.06

    def vector(self) -> np.ndarray:
        return np.array([
            self.w_validation, self.w_displacement, self.w_class_safety,
            self.w_precedent, self.w_economic,
        ])

    @classmethod
    def from_vector(cls, v: np.ndarray, template: "KernelParams") -> "KernelParams":
        d = asdict(template)
        d.update(dict(zip(
            ["w_validation", "w_displacement", "w_class_safety",
             "w_precedent", "w_economic"],
            [float(x) for x in v],
        )))
        return cls(**d)


# Which event types transmit through which channels, and how strongly.
# Rows are event types; values scale the channel activation before weighting.
_TRANSMISSION: dict[EventType, dict[str, float]] = {
    EventType.TOPLINE_EFFICACY:      {"validation": 1.00, "displacement": 1.00, "precedent": 0.15},
    EventType.INTERIM_ANALYSIS:      {"validation": 0.70, "displacement": 0.70, "precedent": 0.10},
    EventType.APPROVAL:              {"validation": 0.45, "displacement": 1.00, "precedent": 0.80},
    EventType.CRL:                   {"validation": 0.30, "displacement": 0.85, "precedent": 0.90},
    EventType.ADCOM:                 {"validation": 0.35, "displacement": 0.70, "precedent": 0.85},
    EventType.ADCOM_BRIEFING:        {"validation": 0.25, "displacement": 0.45, "precedent": 0.70},
    EventType.SAFETY_SIGNAL:         {"class_safety": 1.00, "displacement": 0.80},
    EventType.CLINICAL_HOLD:         {"class_safety": 0.85, "displacement": 0.75},
    EventType.PROGRAM_DISCONTINUATION:{"validation": 0.60, "displacement": 0.90},
    EventType.LABEL_CHANGE:          {"class_safety": 0.60, "displacement": 0.35, "precedent": 0.30},
    EventType.DESIGNATION:           {"validation": 0.20, "precedent": 0.40, "displacement": 0.30},
    EventType.PDUFA_SHIFT:           {"displacement": 0.40, "precedent": 0.25},
    EventType.PARTNERSHIP:           {"validation": 0.45, "economic": 1.00},
    EventType.TRIAL_STATUS:          {"validation": 0.20, "displacement": 0.20},
}


@dataclass
class ChannelActivations:
    validation: float = 0.0
    displacement: float = 0.0
    class_safety: float = 0.0
    precedent: float = 0.0
    economic: float = 0.0

    def vector(self) -> np.ndarray:
        return np.array([
            self.validation, self.displacement, self.class_safety,
            self.precedent, self.economic,
        ])

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class ReadThroughKernel:
    def __init__(self, ontology: Ontology, params: KernelParams | None = None) -> None:
        self.onto = ontology
        self.p = params or KernelParams()

    # ------------------------------------------------------------------
    # Channel activations
    # ------------------------------------------------------------------

    def activations(
        self,
        pid_i: str,
        event: CatalystEvent,
        signed_surprise: float,
    ) -> ChannelActivations:
        """
        Signed activation of each channel for peer program i, given an event on
        program j. `signed_surprise` carries the direction and the magnitude of
        the news relative to what was already priced (see surprise.py).
        """
        rel = self.onto.relation(pid_i, event.program_id)
        i = self.onto.programs[pid_i]
        j = self.onto.programs[event.program_id]
        trans = _TRANSMISSION.get(event.event_type, {})
        ind = self.onto.indication(i.indication)

        a = ChannelActivations()

        # -- VALIDATION -------------------------------------------------
        # Shared biology read across to the peer's own program.
        #
        # Overlap suppresses validation only to the extent the market is
        # zero-sum. In a saturated indication (headroom ~0.1) a rival proving
        # the mechanism mostly just tells you they will take your patients, so
        # validation is nearly extinguished. In a large undertreated market
        # (headroom ~0.9) the same result is close to pure good news for the
        # category even though you compete head-on: both drugs can win.
        #
        # This single term is what makes the model give different answers to
        # "rival succeeds in obesity" and "rival succeeds in 2L NSCLC".
        overlap_discount = 1.0 - rel.indication_overlap * (1.0 - ind.headroom)
        mech_read = rel.mechanism_sim * overlap_discount
        platform_read = rel.platform_sim * 0.8
        a.validation = (
            trans.get("validation", 0.0)
            * (mech_read + platform_read)
            * signed_surprise
        )

        # -- DISPLACEMENT -----------------------------------------------
        # Competition for the same patients. Attenuated by market headroom
        # (untreated patients mean both drugs can win) and amplified by
        # winner-take-most dynamics and by the rival's time-to-market lead.
        contested = rel.indication_overlap * (
            self.p.headroom_floor + (1.0 - self.p.headroom_floor) * (1.0 - ind.headroom)
        )
        contested *= (0.5 + 0.5 * ind.winner_take_most)
        a.displacement = (
            -trans.get("displacement", 0.0)
            * contested
            * self._lead_penalty(rel)
            * signed_surprise
        )

        # -- CLASS SAFETY -----------------------------------------------
        # A tox signal travels along mechanism regardless of indication, and
        # only in one direction: down. A rival's hepatotoxicity is never good
        # news for you if you inhibit the same target. It IS good news if you
        # inhibit a different target in the same indication — that part is
        # picked up by the displacement channel, which flips positive because
        # signed_surprise is negative.
        if trans.get("class_safety", 0.0) > 0.0:
            same_class = rel.safety_class_sim
            a.class_safety = (
                -trans["class_safety"] * same_class * abs(signed_surprise)
            )

        # -- PRECEDENT --------------------------------------------------
        # FDA behavior is information about FDA behavior. An accelerated
        # approval on a surrogate endpoint reprices everyone planning to use
        # that endpoint, whatever their target. Keyed on shared indication
        # family and shared development stage, not on biology.
        stage_prox = self._stage_proximity(i, j)
        a.precedent = (
            trans.get("precedent", 0.0)
            * max(rel.indication_overlap, 0.4 * rel.mechanism_sim)
            * stage_prox
            * signed_surprise
        )

        # -- ECONOMIC ---------------------------------------------------
        # Explicit contractual exposure dominates everything else when present.
        a.economic = (
            trans.get("economic", 0.0)
            * rel.economic_link
            * signed_surprise
        )
        if rel.economic_link != 0.0:
            # A royalty stake makes a partner's win unambiguously good even in
            # a shared indication: suppress the displacement channel.
            a.displacement *= max(0.0, 1.0 - abs(rel.economic_link))

        return a

    def _lead_penalty(self, rel: PairRelation) -> float:
        """
        Ranges ~0.45 (i is far ahead — a laggard rival's win barely dents you)
        to ~1.55 (i is far behind — the rival will take the market first).
        """
        gap = rel.lead_gap_months  # >0 means i reaches market first
        return 1.0 - 0.55 * math.tanh(gap / self.p.lead_gap_scale_months)

    @staticmethod
    def _stage_proximity(i: Program, j: Program) -> float:
        from .ontology import PHASE_ORDER
        d = abs(PHASE_ORDER[i.phase] - PHASE_ORDER[j.phase])
        return math.exp(-d / 2.0)

    # ------------------------------------------------------------------
    # Expected move
    # ------------------------------------------------------------------

    def peer_move(
        self,
        pid_i: str,
        event: CatalystEvent,
        signed_surprise: float,
        event_day_vol: float,
    ) -> tuple[float, dict[str, float]]:
        """
        Expected fractional return on peer i.

        Scaled by two things beyond the channels:
          - the peer's own EV concentration in the affected program (a 2%-of-EV
            program cannot move the stock 15% no matter how strong the signal)
          - the peer's typical event-day volatility (converts a dimensionless
            channel score into a return)
        """
        a = self.activations(pid_i, event, signed_surprise)
        raw = float(np.dot(a.vector(), self.p.vector()))

        i = self.onto.programs[pid_i]
        # Concentration enters sub-linearly: even diversified names react.
        conc = i.ev_share ** 0.6

        sector = self.p.w_sector_beta * signed_surprise * min(
            1.0, self.onto.programs[event.program_id].ev_share
        )

        move = (raw * conc + sector) * event_day_vol
        move = float(np.clip(move, -0.60, 0.60))

        breakdown = {k: v * w * conc * event_day_vol for (k, v), w
                     in zip(a.as_dict().items(), self.p.vector())}
        breakdown["sector_beta"] = sector * event_day_vol
        return move, breakdown

    def own_move(
        self,
        event: CatalystEvent,
        signed_surprise: float,
        event_day_vol: float,
    ) -> float:
        """
        Expected move on the company that issued the news. Included for
        completeness and for the pre-positioning sleeve; note that in live
        trading this move is almost never capturable — see backtest/costs.py.
        """
        j = self.onto.programs[event.program_id]
        kappa = {
            EventType.APPROVAL: 1.6, EventType.CRL: 2.2,
            EventType.TOPLINE_EFFICACY: 2.4, EventType.SAFETY_SIGNAL: 2.0,
            EventType.ADCOM: 1.5, EventType.PROGRAM_DISCONTINUATION: 1.8,
        }.get(event.event_type, 1.0)
        move = kappa * signed_surprise * (j.ev_share ** 0.75) * event_day_vol
        return float(np.clip(move, -0.85, 1.50))
