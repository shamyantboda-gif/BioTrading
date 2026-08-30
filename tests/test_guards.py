"""
Guard tests. These assert the properties that, if violated, silently turn the
backtest into fiction.

    python -m pytest tests/ -q      (or: python tests/test_guards.py)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from fda_alpha.backtest.costs import ExecutionConfig, Session, next_tradeable_time, session_of
from fda_alpha.backtest.engine import ET, PriceBook
from fda_alpha.data.synth import build_universe
from fda_alpha.ontology import Ontology
from fda_alpha.readthrough import KernelParams, ReadThroughKernel
from fda_alpha.schema import CatalystEvent, EventType, SourceKind, SOURCE_LATENCY_SEC
from fda_alpha.surprise import prior_pos, signed_surprise


def _book():
    idx = pd.date_range("2025-06-02 13:30", periods=100, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": np.linspace(10, 20, 100), "high": np.linspace(10, 20, 100) + 0.1,
        "low": np.linspace(10, 20, 100) - 0.1, "close": np.linspace(10, 20, 100),
        "volume": np.full(100, 1e5), "spread_bps": np.full(100, 20.0),
    }, index=idx)
    return PriceBook({"TEST": df}), idx


def test_no_lookahead_on_decision_path():
    book, idx = _book()
    t = idx[50]
    # A decision made at t must not see the bar at t.
    assert book.last_close_before("TEST", t) == book.bars["TEST"]["close"].iloc[49]
    assert book.last_close_before("TEST", idx[0]) is None
    print("ok  decision path is strictly backward-looking")


def test_execution_path_is_forward():
    book, idx = _book()
    t_bar, _ = book.bar_at_or_after("TEST", idx[50])
    assert t_bar == idx[50]
    t_bar, _ = book.bar_at_or_after("TEST", idx[50] + timedelta(seconds=1))
    assert t_bar == idx[51]
    print("ok  execution path resolves forward, never backward")


def test_after_hours_event_rolls_to_next_session():
    cfg = ExecutionConfig()
    # Friday 16:20 ET -> post-market Friday if extended allowed
    t = datetime(2025, 6, 6, 16, 20, tzinfo=ET)
    t_x, sess = next_tradeable_time(t, cfg, allow_extended=True)
    assert sess == Session.POST and t_x.date() == t.date()
    # ...and Monday if not
    t_x, sess = next_tradeable_time(t, cfg, allow_extended=False)
    assert sess == Session.REGULAR and t_x.weekday() == 0, (t_x, sess)
    # Saturday news is not tradeable until Monday under any setting
    t_sat = datetime(2025, 6, 7, 9, 0, tzinfo=ET)
    t_x, _ = next_tradeable_time(t_sat, cfg, allow_extended=True)
    assert t_x.weekday() == 0
    print("ok  weekend and after-hours events roll forward correctly")


def test_halt_blocks_execution():
    cfg = ExecutionConfig()
    t = datetime(2025, 6, 4, 10, 0, tzinfo=ET)
    halt_until = datetime(2025, 6, 4, 11, 5, tzinfo=ET)
    t_x, sess = next_tradeable_time(t, cfg, halted_until=halt_until)
    assert t_x >= halt_until and sess == Session.REGULAR
    print("ok  halts push execution past the resume time")


def test_lagging_sources_cannot_generate_fast_signal():
    # openFDA and CT.gov must be modeled as day-scale lagging reconciliation
    # sources, never as signal sources.
    assert SOURCE_LATENCY_SEC[SourceKind.OPENFDA] >= 3600
    assert SOURCE_LATENCY_SEC[SourceKind.CTGOV] >= 3600
    assert SOURCE_LATENCY_SEC[SourceKind.PR_WIRE] < 5
    print("ok  lagging sources carry day-scale latency")


def test_readthrough_sign_flips_with_market_headroom():
    """
    The central claim of the model: the SAME event shape produces opposite
    signs depending on whether the contested market is saturated.
    """
    programs, indications, links = build_universe()
    onto = Ontology(programs, indications, links)
    k = ReadThroughKernel(onto, KernelParams())
    t = datetime(2025, 6, 3, 14, 0, tzinfo=timezone.utc)

    def move(event_pid, peer_pid):
        p = onto.programs[event_pid]
        ev = CatalystEvent("t", t, t, SourceKind.PR_WIRE,
                           EventType.TOPLINE_EFFICACY, event_pid, p.ticker,
                           polarity=1.0, strength=0.85, confidence=1.0)
        s = signed_surprise(ev, prior_pos(p.phase, market_implied=p.pos_prior))
        return k.peer_move(peer_pid, ev, s, 0.10)[0]

    # obesity: headroom 0.88 -> rival's win LIFTS a head-on competitor
    up = move("METB-gipglp-obes", "AMYL-amylin-obes")
    # 2L NSCLC: headroom 0.10 -> rival's win SINKS a head-on competitor
    down = move("ADCO-adc-nsclc2", "KRSX-g12c-nsclc2")
    assert up > 0 > down, (up, down)
    print(f"ok  headroom flips the sign: obesity {up:+.3%} vs 2L NSCLC {down:+.3%}")


def test_safety_class_does_not_travel_by_modality_alone():
    """Two unrelated antibodies are not in the same safety class."""
    programs, indications, links = build_universe()
    onto = Ontology(programs, indications, links)
    rel = onto.relation("SYNU-asyn-pd", "CNXS-abeta-ad")   # both mabs, unrelated targets
    same = onto.relation("GENE-crispr-attr", "SILN-sirna-attr")  # same target
    assert rel.safety_class_sim < 0.2 < same.safety_class_sim
    print("ok  safety class follows target/pathway, not modality family")


def test_validation_beats_displacement_only_where_it_should():
    programs, indications, links = build_universe()
    onto = Ontology(programs, indications, links)
    assert onto.indication("obesity").headroom > onto.indication("nsclc_2l").headroom
    # ATTR-PN and ATTR-CM are different diseases, not lines of therapy
    r = onto.relation("GENE-crispr-attr", "SILN-sirna-attr")
    assert r.indication_overlap < 0.5, r.indication_overlap
    print("ok  indication overlap distinguishes disease from line of therapy")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
    print("\nall guard tests passed")
