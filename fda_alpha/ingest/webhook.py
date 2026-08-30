"""
Ingestion: webhook receiver + pollers.

Timestamp integrity is the whole job. Three rules:

  1. `t_wire` is the WIRE's timestamp, not ours. Businesswire/GlobeNewswire
     items carry an issuance timestamp; use it. Do not use the time your HTTP
     client finished reading the body, or your backtest and your live system
     will disagree by seconds that matter.

  2. Record `t_ingest` separately and monitor the delta. If your median
     detection latency drifts from 1.5s to 9s, your live fills will diverge
     from every backtest you have run, silently.

  3. Dedupe on content hash, not URL. The same release appears on the wire,
     on the company IR page, in an 8-K, and in openFDA hours later. Trading
     the same information four times is a good way to quadruple a loss.

Sources, in descending order of usefulness for signal:
  * PR wire push feeds (paid; Businesswire NX, GlobeNewswire, PRNewswire)
  * SEC EDGAR full-text / RSS (free, ~40s lag)
  * FDA newsroom RSS (free, minutes)
  * ClinicalTrials.gov v2 (free; status edits post AFTER the press release)
  * openFDA Drugs@FDA (free; batch ETL, lags by a day or more)

The last two are reconciliation and universe-construction sources. They do not
generate tradeable signal — SOURCE_LATENCY_SEC encodes that so the backtester
cannot accidentally treat them as fast.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from queue import Queue
from typing import Callable, Iterable

import urllib.parse
import urllib.request

from ..schema import CatalystEvent, EventType, SourceKind

log = logging.getLogger("fda_alpha.ingest")

CTGOV_BASE = "https://clinicaltrials.gov/api/v2"
OPENFDA_BASE = "https://api.fda.gov"


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------

class Deduper:
    def __init__(self, ttl_sec: float = 86_400.0) -> None:
        self._seen: dict[str, float] = {}
        self._ttl = ttl_sec
        self._lock = threading.Lock()

    @staticmethod
    def content_key(headline: str, ticker: str, body: str = "") -> str:
        norm = " ".join((headline + " " + body[:400]).lower().split())
        return hashlib.sha1(f"{ticker}|{norm}".encode()).hexdigest()

    def is_new(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            self._seen = {k: v for k, v in self._seen.items() if now - v < self._ttl}
            if key in self._seen:
                return False
            self._seen[key] = now
            return True


# ---------------------------------------------------------------------------
# Webhook server
# ---------------------------------------------------------------------------

def verify_signature(secret: str, body: bytes, provided: str) -> bool:
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, provided or "")


class EventBus:
    """Thread-safe fan-out from ingestion threads to the strategy loop."""

    def __init__(self) -> None:
        self.q: "Queue[CatalystEvent]" = Queue()
        self.dedupe = Deduper()

    def publish(self, ev: CatalystEvent) -> bool:
        key = Deduper.content_key(ev.headline, ev.ticker)
        if not self.dedupe.is_new(key):
            log.debug("dedup drop %s", ev.headline[:60])
            return False
        self.q.put(ev)
        return True

    def consume(self, handler: Callable[[CatalystEvent], None]) -> None:
        while True:
            ev = self.q.get()
            try:
                handler(ev)
            except Exception:
                log.exception("handler failed for %s", ev.event_id)


def make_app(bus: EventBus, secret: str, parser: Callable[[dict], CatalystEvent | None]):
    """
    FastAPI app if available. Falls back to a stdlib handler so this module is
    importable (and the repo runnable) without extra dependencies.
    """
    try:
        from fastapi import FastAPI, Header, HTTPException, Request
    except ImportError:
        return _stdlib_server(bus, secret, parser)

    app = FastAPI(title="fda-alpha ingest")

    @app.post("/hook/{source}")
    async def hook(source: str, request: Request,
                   x_signature: str = Header(default="")):
        body = await request.body()
        if secret and not verify_signature(secret, body, x_signature):
            raise HTTPException(status_code=401, detail="bad signature")
        payload = json.loads(body)
        payload["_source"] = source
        ev = parser(payload)
        if ev is None:
            return {"status": "ignored"}
        accepted = bus.publish(ev)
        return {
            "status": "accepted" if accepted else "duplicate",
            "event_id": ev.event_id,
            "detection_latency_sec": ev.detection_latency_sec,
        }

    @app.get("/health")
    async def health():
        return {"ok": True, "queue_depth": bus.q.qsize()}

    return app


def _stdlib_server(bus: EventBus, secret: str, parser):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            sig = self.headers.get("X-Signature", "")
            if secret and not verify_signature(secret, body, sig):
                self.send_response(401); self.end_headers(); return
            try:
                payload = json.loads(body)
                payload["_source"] = self.path.rsplit("/", 1)[-1]
                ev = parser(payload)
            except Exception:
                self.send_response(400); self.end_headers(); return
            status = "ignored"
            if ev is not None:
                status = "accepted" if bus.publish(ev) else "duplicate"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": status}).encode())

        def log_message(self, *a):  # silence
            pass

    def serve(host="0.0.0.0", port=8080):
        HTTPServer((host, port), Handler).serve_forever()

    return serve


# ---------------------------------------------------------------------------
# Pollers (free sources)
# ---------------------------------------------------------------------------

def _get_json(url: str, timeout: float = 15.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "fda-alpha/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def poll_ctgov_updates(
    sponsors: Iterable[str],
    since_date: str,
    page_size: int = 200,
) -> list[dict]:
    """
    ClinicalTrials.gov v2. Use the LastUpdatePostDate range filter to pull
    records touched since `since_date` (YYYY-MM-DD).

    Reconciliation only: a sponsor updating a trial record to COMPLETED or
    TERMINATED almost always trails their own press release. Its real value is
    keeping the Program ontology current — new trials, phase transitions,
    enrollment, and competitor programs you did not know existed.
    """
    out: list[dict] = []
    for sponsor in sponsors:
        params = {
            "query.lead": sponsor,
            "query.term": f"AREA[LastUpdatePostDate]RANGE[{since_date},MAX]",
            "pageSize": str(page_size),
            "fields": ",".join([
                "NCTId", "BriefTitle", "OverallStatus", "Phase",
                "LeadSponsorName", "Condition", "InterventionName",
                "LastUpdatePostDate", "StartDate", "PrimaryCompletionDate",
                "WhyStopped",
            ]),
        }
        url = f"{CTGOV_BASE}/studies?{urllib.parse.urlencode(params)}"
        token = None
        while True:
            u = url + (f"&pageToken={token}" if token else "")
            data = _get_json(u)
            out.extend(data.get("studies", []))
            token = data.get("nextPageToken")
            if not token:
                break
            time.sleep(1.2)   # ~50 req/min limit
    return out


def poll_ctgov_history(nct_id: str) -> list[dict]:
    """
    Version history for a single record — the point-in-time reconstruction
    tool. ClinicalTrials.gov records are edited retroactively; if you build
    features from today's record and backtest them against 2019 prices, you
    are using information that did not exist. Reconstruct the record as of the
    decision date instead.
    """
    url = f"{CTGOV_BASE}/studies/{nct_id}?fields=NCTId"
    _ = _get_json(url)   # existence check
    hist_url = (f"https://clinicaltrials.gov/api/int/studies/{nct_id}/history")
    try:
        return _get_json(hist_url).get("changes", [])
    except Exception:
        log.warning("history unavailable for %s", nct_id)
        return []


def poll_openfda_approvals(since: str, limit: int = 100) -> list[dict]:
    """
    Drugs@FDA via openFDA. Batch-refreshed; expect a day or more of lag versus
    the actual approval announcement. Use it to build and audit the approval
    history table, never to trigger a trade.
    """
    q = urllib.parse.quote(f"submissions.submission_status_date:[{since} TO 99991231]")
    url = (f"{OPENFDA_BASE}/drug/drugsfda.json?search={q}"
           f"&limit={limit}")
    key = os.environ.get("OPENFDA_API_KEY")
    if key:
        url += f"&api_key={key}"
    try:
        return _get_json(url).get("results", [])
    except Exception:
        log.exception("openFDA poll failed")
        return []


class Poller(threading.Thread):
    """Generic polling loop that publishes onto the bus."""

    def __init__(self, bus: EventBus, fn: Callable[[], list[CatalystEvent]],
                 interval_sec: float = 60.0, name: str = "poller") -> None:
        super().__init__(daemon=True, name=name)
        self.bus, self.fn, self.interval = bus, fn, interval_sec
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                for ev in self.fn():
                    self.bus.publish(ev)
            except Exception:
                log.exception("%s failed", self.name)
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
