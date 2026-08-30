"""
Normalized data contracts.

Everything downstream (model, backtest, live execution) speaks these types only.
The single most important field in the whole system is `t_wire`: the UTC instant
at which the information first became public. If that field is wrong, every
backtest number produced by this repo is fiction.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------

class EventType(str, Enum):
    """What kind of information arrived."""

    # Clinical readouts
    TOPLINE_EFFICACY = "topline_efficacy"      # primary endpoint hit / missed
    INTERIM_ANALYSIS = "interim_analysis"      # DSMB interim (stop for efficacy/futility)
    SAFETY_SIGNAL = "safety_signal"            # SAE, death, hepatotox, clinical hold
    TRIAL_STATUS = "trial_status"              # enrollment complete, terminated, suspended

    # Regulatory
    APPROVAL = "approval"                      # NDA/BLA approval
    CRL = "crl"                                # Complete Response Letter (rejection)
    ADCOM = "adcom"                            # advisory committee vote
    ADCOM_BRIEFING = "adcom_briefing"          # FDA briefing docs, ~2 days pre-AdCom
    PDUFA_SHIFT = "pdufa_shift"                # extension / priority review granted
    LABEL_CHANGE = "label_change"              # boxed warning added/removed
    DESIGNATION = "designation"                # breakthrough / fast track / orphan
    CLINICAL_HOLD = "clinical_hold"

    # Corporate, but catalyst-shaped
    PARTNERSHIP = "partnership"
    PROGRAM_DISCONTINUATION = "discontinuation"


class Phase(str, Enum):
    PRECLIN = "preclinical"
    P1 = "phase1"
    P1_2 = "phase1_2"
    P2 = "phase2"
    P2_3 = "phase2_3"
    P3 = "phase3"
    REGISTRATIONAL = "registrational"
    MARKETED = "marketed"


class SourceKind(str, Enum):
    PR_WIRE = "pr_wire"                # Businesswire / GlobeNewswire / PRNewswire
    SEC_8K = "sec_8k"                  # EDGAR full-text
    FDA_PRESS = "fda_press"            # fda.gov newsroom
    OPENFDA = "openfda"                # api.fda.gov (LAGGING — see note below)
    CTGOV = "ctgov"                    # clinicaltrials.gov v2
    FDA_CALENDAR = "fda_calendar"      # AdCom calendar, PDUFA trackers
    MANUAL = "manual"


# Latency profile of each source, in seconds, relative to true public
# dissemination. Used by the backtest to refuse to trade on a source that
# could not physically have been seen at that time.
#
# openFDA is a batch ETL of Drugs@FDA and lags the actual approval press
# release by hours to days. It is a *reconciliation* source, not a signal
# source. CT.gov record edits similarly post after the press release.
SOURCE_LATENCY_SEC: dict[SourceKind, float] = {
    SourceKind.PR_WIRE: 1.0,
    SourceKind.SEC_8K: 45.0,
    SourceKind.FDA_PRESS: 20.0,
    SourceKind.OPENFDA: 86_400.0,
    SourceKind.CTGOV: 21_600.0,
    SourceKind.FDA_CALENDAR: 0.0,
    SourceKind.MANUAL: 0.0,
}


# --------------------------------------------------------------------------
# Core records
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Program:
    """
    A (company, drug, mechanism, indication) tuple. This is the atom of the
    ontology — read-through is computed between Programs, not between tickers.
    """

    program_id: str
    ticker: str
    drug: str
    target: str                  # molecular target, e.g. "PD-1", "TTR", "LRRK2"
    modality: str                # "small_molecule" | "mab" | "adc" | "sirna" | "car_t" | ...
    indication: str              # normalized, e.g. "nsclc_2l" (line of therapy matters)
    phase: Phase

    # Fraction of the company's enterprise value attributable to this program.
    # Drives how far the stock moves on its own news. A one-asset microcap is
    # ~0.85; a large-cap's fourth-line asset is ~0.02.
    ev_share: float = 0.30

    # Months until this program could reach market. Used to compute who is
    # ahead of whom when two companies chase the same indication.
    months_to_market: float = 48.0

    # Prior probability of success. Defaults should be overridden with
    # phase/indication base rates (see surprise.py).
    pos_prior: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 <= self.ev_share <= 1.0:
            raise ValueError(f"ev_share out of range for {self.program_id}")
        if not 0.0 < self.pos_prior < 1.0:
            raise ValueError(f"pos_prior out of range for {self.program_id}")


@dataclass(frozen=True)
class Indication:
    """Market structure of a disease area. Determines whether a rival's win hurts."""

    indication_id: str
    peak_sales_usd_bn: float

    # headroom in [0, 1]: share of addressable patients currently untreated.
    #   ~0.05  mature, saturated (e.g. 2L NSCLC IO) -> zero-sum, rival win hurts
    #   ~0.85  large undertreated (e.g. early Alzheimer's) -> rival win validates
    headroom: float = 0.4

    # winner_take_most in [0,1]: how concentrated share ends up. Rare disease
    # with a first-mover lock is ~0.9; primary care chronic is ~0.2.
    winner_take_most: float = 0.5


@dataclass(frozen=True)
class CatalystEvent:
    """
    One piece of market-moving regulatory/clinical information, normalized.
    """

    event_id: str
    t_wire: datetime                 # UTC, tz-aware. First public dissemination.
    t_ingest: datetime               # UTC. When *our* system saw it.
    source: SourceKind
    event_type: EventType
    program_id: str
    ticker: str

    # +1 good, -1 bad, 0 ambiguous/mixed. For AdCom, use vote margin scaled.
    polarity: float = 0.0

    # [0, 1] how strong the result is *conditional on direction*. A p<0.0001 hit
    # on OS with a hazard ratio of 0.55 is 1.0; a barely-significant PFS-only
    # win with no OS trend is 0.3.
    strength: float = 0.5

    # [0, 1] parser confidence. Gates whether we trade at all.
    confidence: float = 1.0

    headline: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("t_wire", "t_ingest"):
            ts = getattr(self, name)
            if ts.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware UTC")

    @property
    def detection_latency_sec(self) -> float:
        return (self.t_ingest - self.t_wire).total_seconds()

    @staticmethod
    def make_id(source: SourceKind, t_wire: datetime, headline: str) -> str:
        """Deterministic id so the same wire item ingested twice dedupes."""
        key = f"{source.value}|{t_wire.isoformat()}|{headline.strip().lower()}"
        return hashlib.sha1(key.encode()).hexdigest()[:16]


@dataclass
class Signal:
    """Model output for one (event, ticker) pair."""

    event_id: str
    ticker: str
    leg: str                      # "own" | "peer"
    expected_move: float          # signed fractional return, e.g. -0.12
    conviction: float             # [0,1]; expected_move / event-day vol, squashed
    horizon_min: int              # holding period in minutes
    channel_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class Fill:
    ticker: str
    t: datetime
    qty: float                    # signed; negative = short
    price: float
    slippage_bps: float
    notional: float
    reason: str = ""


def utc(*args, **kwargs) -> datetime:
    """Shorthand for tz-aware UTC datetimes in configs and tests."""
    return datetime(*args, **kwargs, tzinfo=timezone.utc)
