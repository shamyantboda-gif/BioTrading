"""
Live execution: broker abstraction, pre-trade risk, and the run loop.

Defaults are paper-trading and DRY_RUN. Flipping to live is a deliberate
environment-variable act, not a code path you can fall into.

The risk gates below are not decoration. An automated system that trades on
parsed natural language will, eventually, act on a misparse — a headline about
a competitor's approval attributed to the wrong ticker, an 8-K that repeats
last quarter's news, a wire test message. The gates are what stand between a
misparse and an account.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ..schema import CatalystEvent, Fill, Signal

log = logging.getLogger("fda_alpha.exec")


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------

class Broker(ABC):
    @abstractmethod
    def submit(self, ticker: str, qty: float, limit_price: float | None,
               tif: str = "day", extended: bool = False) -> str: ...

    @abstractmethod
    def positions(self) -> dict[str, float]: ...

    @abstractmethod
    def quote(self, ticker: str) -> tuple[float, float]: ...

    @abstractmethod
    def is_halted(self, ticker: str) -> bool: ...

    @abstractmethod
    def flatten_all(self) -> None: ...


class PaperBroker(Broker):
    """In-memory broker for dry runs and integration tests."""

    def __init__(self, quotes: dict[str, tuple[float, float]] | None = None) -> None:
        self._pos: dict[str, float] = {}
        self._quotes = quotes or {}
        self._orders: list[dict] = []

    def submit(self, ticker, qty, limit_price=None, tif="day", extended=False):
        oid = f"paper-{len(self._orders)}"
        bid, ask = self.quote(ticker)
        px = limit_price or (ask if qty > 0 else bid)
        self._pos[ticker] = self._pos.get(ticker, 0.0) + qty
        self._orders.append({"id": oid, "ticker": ticker, "qty": qty, "px": px,
                             "t": datetime.now(timezone.utc), "extended": extended})
        log.info("PAPER %s %s @ %.4f", ticker, qty, px)
        return oid

    def positions(self):
        return dict(self._pos)

    def quote(self, ticker):
        return self._quotes.get(ticker, (10.0, 10.02))

    def is_halted(self, ticker):
        return False

    def flatten_all(self):
        for t, q in list(self._pos.items()):
            if q:
                self.submit(t, -q)
        self._pos.clear()


class AlpacaBroker(Broker):
    """
    Thin REST wrapper. Requires ALPACA_KEY_ID / ALPACA_SECRET_KEY.
    Defaults to the paper endpoint; the live URL must be set explicitly.

    Extended-hours orders must be LIMIT and marked extended_hours=True. A
    market order sent pre-market is silently queued to the open, which will
    fill you into the auction at a price unrelated to your signal.
    """

    def __init__(self, base_url: str | None = None) -> None:
        import urllib.request  # local import keeps the module dependency-free
        self._req = urllib.request
        self.base = base_url or os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )
        self.key = os.environ["ALPACA_KEY_ID"]
        self.secret = os.environ["ALPACA_SECRET_KEY"]

    def _call(self, path: str, method: str = "GET", body: dict | None = None):
        import json
        url = f"{self.base}{path}"
        data = json.dumps(body).encode() if body else None
        req = self._req.Request(url, data=data, method=method, headers={
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
            "Content-Type": "application/json",
        })
        with self._req.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode() or "{}")

    def submit(self, ticker, qty, limit_price=None, tif="day", extended=False):
        if extended and limit_price is None:
            raise ValueError("extended-hours orders must be limit orders")
        order = {
            "symbol": ticker,
            "qty": abs(round(qty)),
            "side": "buy" if qty > 0 else "sell",
            "type": "limit" if limit_price else "market",
            "time_in_force": tif,
            "extended_hours": extended,
        }
        if limit_price:
            order["limit_price"] = round(limit_price, 2)
        return self._call("/v2/orders", "POST", order)["id"]

    def positions(self):
        return {p["symbol"]: float(p["qty"]) for p in self._call("/v2/positions")}

    def quote(self, ticker):
        d = self._call(f"/v2/stocks/{ticker}/quotes/latest")["quote"]
        return float(d["bp"]), float(d["ap"])

    def is_halted(self, ticker):
        try:
            return not self._call(f"/v2/assets/{ticker}")["tradable"]
        except Exception:
            return True   # fail closed

    def flatten_all(self):
        self._call("/v2/positions", "DELETE")


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

@dataclass
class RiskLimits:
    max_notional_per_name: float = 100_000.0
    max_gross_notional: float = 2_000_000.0
    max_orders_per_event: int = 9
    max_orders_per_hour: int = 60
    daily_loss_limit: float = 50_000.0
    min_signal_confidence: float = 0.70
    max_detection_latency_sec: float = 30.0
    require_two_sources: bool = False    # corroboration before size
    max_price_move_since_wire: float = 0.35   # do not chase a done move
    dry_run: bool = True


@dataclass
class RiskState:
    day_pnl: float = 0.0
    orders_this_hour: list[datetime] = field(default_factory=list)
    gross: float = 0.0
    tripped: str = ""


class RiskManager:
    def __init__(self, limits: RiskLimits) -> None:
        self.lim = limits
        self.state = RiskState()
        self._lock = threading.Lock()

    def check(self, ev: CatalystEvent, sig: Signal, notional: float,
              broker: Broker, price_move_since_wire: float) -> tuple[bool, str]:
        with self._lock:
            if self.state.tripped:
                return False, f"kill_switch:{self.state.tripped}"
            if sig.conviction < self.lim.min_signal_confidence:
                return False, "low_conviction"
            if ev.confidence < self.lim.min_signal_confidence:
                return False, "low_parse_confidence"
            if ev.detection_latency_sec > self.lim.max_detection_latency_sec:
                # We are late. The information is public and the move has
                # happened; entering now is buying someone else's exit.
                return False, "stale_detection"
            if abs(price_move_since_wire) > self.lim.max_price_move_since_wire:
                return False, "move_already_done"
            if broker.is_halted(sig.ticker):
                return False, "halted"
            if notional > self.lim.max_notional_per_name:
                return False, "name_limit"
            if self.state.gross + notional > self.lim.max_gross_notional:
                return False, "gross_limit"

            now = datetime.now(timezone.utc)
            self.state.orders_this_hour = [
                t for t in self.state.orders_this_hour if now - t < timedelta(hours=1)
            ]
            if len(self.state.orders_this_hour) >= self.lim.max_orders_per_hour:
                return False, "rate_limit"

            if self.state.day_pnl < -self.lim.daily_loss_limit:
                self.state.tripped = "daily_loss"
                return False, "kill_switch:daily_loss"

            self.state.orders_this_hour.append(now)
            self.state.gross += notional
            return True, "ok"

    def trip(self, reason: str, broker: Broker) -> None:
        with self._lock:
            self.state.tripped = reason
        log.critical("KILL SWITCH: %s — flattening", reason)
        broker.flatten_all()


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

class LiveStrategy:
    def __init__(self, signal_engine, broker: Broker, risk: RiskManager,
                 mkt_ctx_fn, capital: float) -> None:
        self.sig = signal_engine
        self.broker = broker
        self.risk = risk
        self.mkt_ctx_fn = mkt_ctx_fn
        self.capital = capital
        self._exits: list[tuple[datetime, str, float]] = []

    def on_event(self, ev: CatalystEvent) -> list[str]:
        t0 = time.perf_counter()
        mkt = self.mkt_ctx_fn(datetime.now(timezone.utc))
        signals = self.sig.generate(ev, mkt)
        order_ids: list[str] = []

        for s in signals[: self.risk.lim.max_orders_per_event]:
            bid, ask = self.broker.quote(s.ticker)
            mid = (bid + ask) / 2
            ref = mkt.__dict__.get("_ref_price", {}).get(s.ticker, mid)
            move_since = (mid - ref) / ref if ref else 0.0

            notional = min(
                self.risk.lim.max_notional_per_name,
                self.capital * 0.02 * s.conviction,
            )
            ok, why = self.risk.check(ev, s, notional, self.broker, move_since)
            if not ok:
                log.info("blocked %s %s: %s", s.ticker, s.leg, why)
                continue

            side = 1 if s.expected_move > 0 else -1
            # Marketable limit, not market: caps the damage from a misparse or
            # a thin extended-hours book.
            limit = ask * 1.004 if side > 0 else bid * 0.996
            qty = side * notional / max(mid, 0.01)

            if self.risk.lim.dry_run:
                log.info("DRY_RUN %s qty=%.0f limit=%.2f exp=%.3f",
                         s.ticker, qty, limit, s.expected_move)
                continue

            oid = self.broker.submit(s.ticker, qty, limit_price=limit,
                                     extended=True)
            order_ids.append(oid)
            self._exits.append((
                datetime.now(timezone.utc) + timedelta(minutes=s.horizon_min),
                s.ticker, qty,
            ))

        log.info("event %s handled in %.1f ms (%d orders)",
                 ev.event_id, (time.perf_counter() - t0) * 1000, len(order_ids))
        return order_ids

    def poll_exits(self) -> None:
        now = datetime.now(timezone.utc)
        due = [x for x in self._exits if x[0] <= now]
        self._exits = [x for x in self._exits if x[0] > now]
        for _, ticker, qty in due:
            if not self.risk.lim.dry_run:
                self.broker.submit(ticker, -qty)
