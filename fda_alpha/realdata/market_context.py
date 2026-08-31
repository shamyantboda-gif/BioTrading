"""
Point-in-time MarketContext + listing spans from a real daily price panel.

Mirrors ``data.synth.market_context_fn`` but reads realized vol and ADV from
actual Yahoo bars strictly *before* the decision instant, so the same
point-in-time discipline the synthetic path enforces holds on real data.
Market cap and borrow terms come from the static ``REAL_TICKER_META`` (clearly
approximate; see ontology_real).
"""

from __future__ import annotations

from bisect import bisect_left
from datetime import datetime

import numpy as np
import pandas as pd

from ..signal import MarketContext
from .ontology_real import REAL_TICKER_META


def market_context_fn(prices: dict[str, pd.DataFrame]):
    """Return ``fn(t) -> MarketContext`` using only daily bars with index < t."""
    # Precompute a plain datetime index per ticker and bisect it, exactly as
    # PriceBook does. Using DatetimeIndex.searchsorted directly raises when the
    # (parquet-restored) index resolution differs from the sub-second decision
    # timestamp; bisect on comparable datetimes sidesteps the unit clash.
    idx_lists = {tk: list(df.index) for tk, df in prices.items()}

    def fn(t: datetime) -> MarketContext:
        vols, caps, advs, borrow_ok, borrow_fee = {}, {}, {}, {}, {}
        for tk, df in prices.items():
            cap, ok, fee = REAL_TICKER_META.get(tk, (1.0, True, 100.0))
            k = bisect_left(idx_lists[tk], t) - 1  # strictly before t
            dvol, adv = 0.045, 5e6            # fallbacks
            if k >= 20:
                close = df["close"].to_numpy()[max(0, k - 60):k + 1]
                r = np.diff(np.log(close))
                if len(r) > 10:
                    dvol = float(np.std(r))    # daily realized vol
                vol_sh = df["volume"].to_numpy()[max(0, k - 20):k + 1]
                px = df["close"].to_numpy()[max(0, k - 20):k + 1]
                adv = float(np.mean(px * vol_sh))
            vols[tk], caps[tk], advs[tk] = dvol, cap, adv
            borrow_ok[tk], borrow_fee[tk] = ok, float(fee)
        return MarketContext(vols, caps, advs, borrow_ok, borrow_fee)

    return fn


def listed_spans(
    prices: dict[str, pd.DataFrame]
) -> dict[str, tuple[datetime, datetime]]:
    """
    Real listing spans = first/last available bar per ticker.

    This captures genuine point-in-time membership for names that IPO'd or were
    acquired mid-sample (a real, if partial, survivorship control). It is not a
    full point-in-time index table with delisted tickers — that needs a paid
    membership feed, noted in RESEARCH_REALDATA.md.
    """
    spans: dict[str, tuple[datetime, datetime]] = {}
    for tk, df in prices.items():
        if not df.empty:
            spans[tk] = (df.index[0].to_pydatetime(), df.index[-1].to_pydatetime())
    return spans
