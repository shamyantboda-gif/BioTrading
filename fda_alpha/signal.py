"""
Signal layer: event -> list of Signals across the own name and its peer complex.

Deliberately produces the peer legs even when the own name is untradeable
(halted, gapped, no borrow). The peer complex is where the latency edge lives:
the issuing stock reprices in the pre-market print, while a rival three
mechanisms away can take minutes to hours to be repriced by humans reading the
release.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ontology import Ontology
from .readthrough import ReadThroughKernel
from .schema import CatalystEvent, Signal
from .surprise import event_day_vol, prior_pos, signed_surprise


@dataclass
class MarketContext:
    """Point-in-time market state, as of an instant strictly before t_signal."""

    daily_vol: dict[str, float]            # 20d realized vol, per ticker
    market_cap_bn: dict[str, float]
    adv_usd: dict[str, float]              # 20d average dollar volume
    borrowable: dict[str, bool]
    borrow_fee_bps: dict[str, float]
    implied_pos: dict[str, float] | None = None   # program_id -> options-implied POS

    def vol(self, t: str) -> float:
        return self.daily_vol.get(t, 0.045)

    def cap(self, t: str) -> float:
        return self.market_cap_bn.get(t, 1.0)


@dataclass
class SignalConfig:
    min_confidence: float = 0.65           # parser confidence floor
    min_abs_move: float = 0.010            # ignore anything under 1.2% expected
    min_conviction: float = 0.10
    max_peer_legs: int = 8
    own_leg_enabled: bool = True
    peer_horizon_min: int = 90
    own_horizon_min: int = 240


class SignalEngine:
    def __init__(
        self,
        ontology: Ontology,
        kernel: ReadThroughKernel,
        cfg: SignalConfig | None = None,
    ) -> None:
        self.onto = ontology
        self.kernel = kernel
        self.cfg = cfg or SignalConfig()

    def generate(self, event: CatalystEvent, mkt: MarketContext) -> list[Signal]:
        if event.confidence < self.cfg.min_confidence:
            return []
        if event.program_id not in self.onto.programs:
            return []

        j = self.onto.programs[event.program_id]
        implied = (mkt.implied_pos or {}).get(event.program_id)
        pos = prior_pos(
            j.phase,
            therapeutic_area=_area_of(j.indication),
            market_implied=implied if implied is not None else j.pos_prior,
        )
        s = signed_surprise(event, pos)
        if s == 0.0:
            return []

        out: list[Signal] = []

        # -- own leg -----------------------------------------------------
        if self.cfg.own_leg_enabled:
            v_own = event_day_vol(mkt.vol(j.ticker), mkt.cap(j.ticker))
            move = self.kernel.own_move(event, s, v_own)
            if abs(move) >= self.cfg.min_abs_move:
                out.append(Signal(
                    event_id=event.event_id, ticker=j.ticker, leg="own",
                    expected_move=move,
                    conviction=_conviction(move, v_own),
                    horizon_min=self.cfg.own_horizon_min,
                    channel_breakdown={"own_binary": move},
                ))

        # -- peer legs ---------------------------------------------------
        candidates: list[tuple[float, Signal]] = []
        for pid_i in self.onto.peers_of(event.program_id):
            i = self.onto.programs[pid_i]
            if i.ticker == j.ticker:
                continue
            v_i = event_day_vol(mkt.vol(i.ticker), mkt.cap(i.ticker),
                                is_binary_catalyst=False)
            move, breakdown = self.kernel.peer_move(pid_i, event, s, v_i)
            if abs(move) < self.cfg.min_abs_move:
                continue
            conv = _conviction(move, v_i)
            if conv < self.cfg.min_conviction:
                continue
            candidates.append((conv, Signal(
                event_id=event.event_id, ticker=i.ticker, leg="peer",
                expected_move=move, conviction=conv,
                horizon_min=self.cfg.peer_horizon_min,
                channel_breakdown=breakdown,
            )))

        candidates.sort(key=lambda x: -x[0])
        # Collapse duplicate tickers (a company can own several peer programs):
        # keep the strongest-conviction leg per ticker rather than summing,
        # which would double-count a single stock's reaction.
        seen: set[str] = set()
        for _, sig in candidates:
            if sig.ticker in seen:
                continue
            seen.add(sig.ticker)
            out.append(sig)
            if len(seen) >= self.cfg.max_peer_legs:
                break

        return out


def _conviction(expected_move: float, vol: float) -> float:
    """Squash expected_move / vol into [0,1]."""
    import math
    z = abs(expected_move) / max(vol, 1e-4)
    return float(math.tanh(z / 1.5))


def _area_of(indication: str) -> str:
    from .ontology import INDICATION_FAMILY
    fam = INDICATION_FAMILY.get(indication, "")
    return {
        "lung": "oncology", "b_cell_malignancy": "oncology",
        "plasma_cell_malignancy": "oncology",
        "neurodeg": "neurology", "attr": "rare_disease",
        "cardiometabolic": "cardiometabolic",
        "ibd": "immunology", "derm_immuno": "immunology",
    }.get(fam, "")
