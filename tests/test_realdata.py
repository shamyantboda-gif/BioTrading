"""
Guards for the real-data adapters. Network-free: catalyst parsing is exercised
by injecting canned API payloads through the on-disk cache, so this runs in CI
without hitting openFDA / CT.gov / Yahoo.

Run: python tests/test_realdata.py
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fda_alpha.ontology import (
    INDICATION_FAMILY, MODALITY_FAMILY, PATHWAY, Ontology,
)
from fda_alpha.schema import SOURCE_LATENCY_SEC, SourceKind
from fda_alpha.realdata import cache, catalysts as C
from fda_alpha.realdata.market_context import listed_spans, market_context_fn
from fda_alpha.realdata.ontology_real import (
    PROGRAM_QUERIES, build_universe,
)


def _ok(msg: str) -> None:
    print(f"ok  {msg}")


def test_ontology_wellformed() -> None:
    programs, indications, _ = build_universe()
    ind_ids = {i.indication_id for i in indications}
    for p in programs:
        assert p.target in PATHWAY, f"{p.program_id}: target {p.target} not in PATHWAY"
        assert p.modality in MODALITY_FAMILY, f"{p.program_id}: modality {p.modality}"
        assert p.indication in INDICATION_FAMILY, f"{p.program_id}: indication"
        assert p.indication in ind_ids, f"{p.program_id}: indication not defined"
        # schema invariants (would raise in __post_init__, assert intent anyway)
        assert 0.0 <= p.ev_share <= 1.0 and 0.0 < p.pos_prior < 1.0
    # every program must have at least one peer, or the ontology slice is useless
    onto = Ontology(programs, indications, {})
    for p in programs:
        assert onto.peers_of(p.program_id), f"{p.program_id} has no peers"
    _ok(f"real ontology well-formed: {len(programs)} programs, all targets/"
        f"indications/modalities known, every program has peers")


def test_query_coverage() -> None:
    programs, _, _ = build_universe()
    for p in programs:
        assert p.program_id in PROGRAM_QUERIES, f"no query for {p.program_id}"
        assert PROGRAM_QUERIES[p.program_id].get("generic"), "empty generic"
    _ok("every program has a catalyst search query")


def test_lagging_sources_are_lagging() -> None:
    # The whole honesty of the real path rests on these being day-scale.
    assert SOURCE_LATENCY_SEC[SourceKind.OPENFDA] >= 86_400.0
    assert SOURCE_LATENCY_SEC[SourceKind.CTGOV] >= 3_600.0
    assert SOURCE_LATENCY_SEC[SourceKind.PR_WIRE] <= 5.0
    _ok("openFDA/CT.gov tagged day-scale lagging; PR wire sub-5s")


def test_openfda_parse_and_collapse() -> None:
    # Canned openFDA payload with an ORIG approval and two same-day supplements.
    payload = {"results": [{
        "application_number": "NDA999999",
        "sponsor_name": "TEST CO",
        "openfda": {"generic_name": ["testdrug"]},
        "submissions": [
            {"submission_type": "ORIG", "submission_number": "1",
             "submission_status": "AP", "submission_status_date": "20230115"},
            {"submission_type": "SUPPL", "submission_number": "2",
             "submission_status": "AP", "submission_status_date": "20230620"},
            {"submission_type": "SUPPL", "submission_number": "3",
             "submission_status": "AP", "submission_status_date": "20230620"},
            {"submission_type": "ORIG", "submission_number": "9",
             "submission_status": "TA", "submission_status_date": "20230101"},  # not AP
        ],
    }]}
    cache.put_json("openfda|testdrug", payload)
    evs = C.fetch_openfda_approvals("TEST-pid", "TEST", "testdrug")
    approvals = [e for e in evs if e.event_type.value == "approval"]
    suppls = [e for e in evs if e.event_type.value == "label_change"]
    assert len(approvals) == 1, "exactly one ORIG-AP approval expected"
    assert len(suppls) == 2, "two supplements before collapse"
    assert all(e.source is SourceKind.OPENFDA for e in evs)
    assert all(e.polarity > 0 for e in evs)

    # build_catalysts default drops supplements and collapses same-day.
    cache.put_json("ctgov|testdrug", {})  # no trials
    q = {"TEST-pid": {"generic": "testdrug"}}
    built = C.build_catalysts(
        q, {"TEST-pid": "TEST"},
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    assert len(built) == 1 and built[0].event_type.value == "approval", \
        "default build keeps only the original approval"
    _ok("openFDA parse: ORIG->approval, SUPPL->label_change, non-AP skipped, "
        "supplements dropped + same-day collapsed by default")


def test_ctgov_only_negative_directional() -> None:
    payload = {"studies": [
        {"protocolSection": {
            "identificationModule": {"nctId": "NCT1"},
            "statusModule": {"overallStatus": "TERMINATED",
                             "lastUpdatePostDateStruct": {"date": "2022-05-10"}}}},
        {"protocolSection": {
            "identificationModule": {"nctId": "NCT2"},
            "statusModule": {"overallStatus": "COMPLETED",
                             "lastUpdatePostDateStruct": {"date": "2022-06-10"}}}},
    ]}
    cache.put_json("ctgov|testdrug2", payload)
    evs = C.fetch_ctgov_status("P", "T", "testdrug2")
    assert len(evs) == 1, "only TERMINATED is directional; COMPLETED dropped"
    assert evs[0].polarity < 0 and evs[0].source is SourceKind.CTGOV
    _ok("CT.gov: terminations are negative catalysts; completions ignored")


def test_market_context_point_in_time() -> None:
    # Build a tiny 2-ticker daily panel; the last bar has an extreme return.
    idx = pd.date_range("2023-01-02", periods=40, freq="B", tz="UTC")
    base = np.linspace(100, 110, 40)
    spike = base.copy()
    spike[-1] = 400.0  # a future blow-up that must NOT leak into vol before it
    df_a = pd.DataFrame({"open": base, "high": base, "low": base,
                         "close": base, "volume": 1e6, "spread_bps": 10.0}, index=idx)
    df_b = pd.DataFrame({"open": spike, "high": spike, "low": spike,
                         "close": spike, "volume": 1e6, "spread_bps": 10.0}, index=idx)
    prices = {"AAA": df_a, "BBB": df_b}
    fn = market_context_fn(prices)

    # As of the timestamp of the last bar, vol must exclude that bar (strictly
    # before). Vol computed at t = last-bar time should equal vol excluding it.
    t_last = idx[-1].to_pydatetime()
    mc = fn(t_last)
    # BBB's pre-spike series is smooth, so realized vol must be small, not huge.
    assert mc.daily_vol["BBB"] < 0.05, \
        f"future spike leaked into point-in-time vol: {mc.daily_vol['BBB']}"
    # A timestamp strictly after the spike bar DOES see it.
    mc_after = fn(idx[-1].to_pydatetime().replace(microsecond=1) +
                  (idx[-1] - idx[-2]).to_pytimedelta())
    assert mc_after.daily_vol["BBB"] > mc.daily_vol["BBB"], \
        "vol after the spike bar should rise"
    _ok("market_context is point-in-time: future bars never leak into vol/ADV")


def test_listed_spans_from_panel() -> None:
    idx = pd.date_range("2021-03-01", periods=10, freq="B", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "volume": 1.0, "spread_bps": 10.0}, index=idx)
    spans = listed_spans({"XYZ": df})
    assert spans["XYZ"][0] == idx[0].to_pydatetime()
    assert spans["XYZ"][1] == idx[-1].to_pydatetime()
    _ok("listing spans derived from real first/last available bar")


def main() -> None:
    test_ontology_wellformed()
    test_query_coverage()
    test_lagging_sources_are_lagging()
    test_openfda_parse_and_collapse()
    test_ctgov_only_negative_directional()
    test_market_context_point_in_time()
    test_listed_spans_from_panel()
    print("\nall real-data guard tests passed")


if __name__ == "__main__":
    main()
