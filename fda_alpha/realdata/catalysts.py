"""
Real catalyst adapters: openFDA Drugs@FDA + ClinicalTrials.gov v2.

These are the only *free, historical, structured* catalyst sources, and that is
precisely the problem the honesty in this repo is built around:

* openFDA gives an approval **date** (no wire time) and is a batch ETL of
  Drugs@FDA that lags the approval press release by a day or more.
* CT.gov status edits post **after** the sponsor's own release.

`schema.SOURCE_LATENCY_SEC` already encodes this (OPENFDA=86400s, CTGOV=21600s),
so the backtester adds a day-plus to ``t_wire`` before it is allowed to trade —
by which point the information is fully public. The result: these sources are
faithful for *reconciliation and ontology maintenance* and structurally
useless as an intraday signal. Building the pipeline and watching the engine
refuse them is the lesson, not a defect to engineer around. The real trade
needs a licensed wire feed with sub-second timestamps; that is the binding
constraint, and it is not free.

Attribution is by construction: we query each source once per program using the
program's own generic name, so every returned filing/trial already belongs to a
known ``program_id``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests

from . import cache
from ..schema import CatalystEvent, EventType, SourceKind, SOURCE_LATENCY_SEC

_UA = {"User-Agent": "fda-alpha-research/0.1 (educational; contact: local)"}

_OPENFDA_URL = "https://api.fda.gov/drug/drugsfda.json"
_CTGOV_URL = "https://clinicaltrials.gov/api/v2/studies"

# CT.gov statuses that are genuinely directional (negative). COMPLETED is
# deliberately excluded: a trial finishing is not itself a directional readout,
# so it maps to polarity 0 and the signal engine drops it.
_CTGOV_NEGATIVE = {"TERMINATED", "SUSPENDED", "WITHDRAWN"}


def _get_json(url: str, params: dict, key: str) -> dict | None:
    cached = cache.get_json(key)
    if cached is not None:
        return cached
    try:
        r = requests.get(url, params=params, headers=_UA, timeout=25)
    except requests.RequestException:
        return None
    if r.status_code == 404:  # openFDA returns 404 for "no matches"
        cache.put_json(key, {})
        return {}
    if r.status_code != 200:
        return None
    payload = r.json()
    cache.put_json(key, payload)
    return payload


def _parse_yyyymmdd(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y%m%d").replace(
            hour=16, minute=30, tzinfo=timezone.utc  # date-only; noon-ish ET
        )
    except (ValueError, TypeError):
        return None


def _parse_iso_date(s: str) -> datetime | None:
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(s, fmt).replace(
                hour=16, minute=30, tzinfo=timezone.utc
            )
        except (ValueError, TypeError):
            continue
    return None


def fetch_openfda_approvals(
    program_id: str, ticker: str, generic: str, limit: int = 100
) -> list[CatalystEvent]:
    """Approval (ORIG) and supplemental-approval (SUPPL) events for one drug."""
    key = f"openfda|{generic.lower()}"
    payload = _get_json(
        _OPENFDA_URL,
        {"search": f'openfda.generic_name:"{generic}"', "limit": limit},
        key,
    )
    if not payload or "results" not in payload:
        return []

    events: list[CatalystEvent] = []
    for res in payload["results"]:
        for sub in res.get("submissions", []):
            if sub.get("submission_status") != "AP":
                continue
            t_wire = _parse_yyyymmdd(sub.get("submission_status_date", ""))
            if t_wire is None:
                continue
            is_orig = sub.get("submission_type") == "ORIG"
            etype = EventType.APPROVAL if is_orig else EventType.LABEL_CHANGE
            headline = (
                f"{generic} {'approval' if is_orig else 'supplemental approval'} "
                f"({res.get('application_number', '')} "
                f"{sub.get('submission_type')}-{sub.get('submission_number')})"
            )
            events.append(_make_event(
                SourceKind.OPENFDA, t_wire, etype, program_id, ticker,
                polarity=1.0,                       # an approval is positive
                strength=0.75 if is_orig else 0.35,
                headline=headline,
                raw={"application_number": res.get("application_number"),
                     "submission": sub},
            ))
    return events


def fetch_ctgov_status(
    program_id: str, ticker: str, generic: str, page_size: int = 50
) -> list[CatalystEvent]:
    """Directional trial-status changes (terminations/suspensions) for one drug."""
    key = f"ctgov|{generic.lower()}"
    payload = _get_json(
        _CTGOV_URL,
        {
            "query.intr": generic,
            "pageSize": page_size,
            "fields": ",".join([
                "protocolSection.identificationModule.nctId",
                "protocolSection.statusModule.overallStatus",
                "protocolSection.statusModule.lastUpdatePostDateStruct",
                "protocolSection.designModule.phases",
            ]),
        },
        key,
    )
    if not payload or "studies" not in payload:
        return []

    events: list[CatalystEvent] = []
    for st in payload["studies"]:
        ps = st.get("protocolSection", {})
        sm = ps.get("statusModule", {})
        status = sm.get("overallStatus", "")
        if status not in _CTGOV_NEGATIVE:
            continue
        posted = (sm.get("lastUpdatePostDateStruct") or {}).get("date")
        t_wire = _parse_iso_date(posted)
        if t_wire is None:
            continue
        nct = ps.get("identificationModule", {}).get("nctId", "")
        events.append(_make_event(
            SourceKind.CTGOV, t_wire, EventType.TRIAL_STATUS, program_id, ticker,
            polarity=-1.0, strength=0.6,
            headline=f"{generic} trial {status.lower()} ({nct})",
            raw={"nct_id": nct, "status": status},
        ))
    return events


def _make_event(
    source: SourceKind, t_wire: datetime, etype: EventType,
    program_id: str, ticker: str, *, polarity: float, strength: float,
    headline: str, raw: dict,
) -> CatalystEvent:
    # t_ingest reflects a batch/reconciliation source: we "see" it only after
    # the source's own latency. It does not drive backtest timing (the engine
    # uses SOURCE_LATENCY_SEC for that) but keeps detection_latency honest.
    t_ingest = t_wire + timedelta(seconds=SOURCE_LATENCY_SEC[source])
    return CatalystEvent(
        event_id=CatalystEvent.make_id(source, t_wire, headline),
        t_wire=t_wire, t_ingest=t_ingest, source=source, event_type=etype,
        program_id=program_id, ticker=ticker, polarity=polarity,
        strength=strength, confidence=1.0, headline=headline, raw=raw,
    )


def build_catalysts(
    program_queries: dict[str, dict[str, str]],
    id_to_ticker: dict[str, str],
    start: datetime,
    end: datetime,
    include_supplements: bool = False,
) -> list[CatalystEvent]:
    """
    Real catalysts for the given programs, within [start, end], cleaned.

    ``program_queries`` maps program_id -> {"generic": ..., "sponsor": ...}
    (see ontology_real.PROGRAM_QUERIES). ``id_to_ticker`` maps program_id ->
    ticker.

    Two data-quality steps, both defensible in an interview:

    * ``include_supplements=False`` (default) drops SUPPL "label_change" events.
      openFDA batch-dates old supplements — e.g. eight Keytruda supplements
      stamped the same day — so their ``submission_status_date`` is an ETL
      artifact, not a real approval instant. Original approvals (ORIG) and trial
      status changes carry trustworthy dates.
    * Same ``(ticker, date, event_type)`` is collapsed to one event, so a batch
      of same-day filings cannot masquerade as many independent catalysts.
    """
    collected: list[CatalystEvent] = []
    for pid, q in program_queries.items():
        generic = q.get("generic", "")
        ticker = id_to_ticker.get(pid, "")
        if not generic or not ticker:
            continue
        for ev in (fetch_openfda_approvals(pid, ticker, generic)
                   + fetch_ctgov_status(pid, ticker, generic)):
            if not (start <= ev.t_wire <= end):
                continue
            if ev.event_type == EventType.LABEL_CHANGE and not include_supplements:
                continue
            collected.append(ev)

    # Collapse same-day same-type events on a name to a single catalyst.
    collapsed: dict[tuple, CatalystEvent] = {}
    for ev in collected:
        key = (ev.ticker, ev.t_wire.date(), ev.event_type)
        collapsed.setdefault(key, ev)
    return sorted(collapsed.values(), key=lambda e: e.t_wire)
