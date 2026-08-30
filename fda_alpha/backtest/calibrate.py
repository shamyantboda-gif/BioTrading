"""
Fitting the read-through channel weights.

The hand-set priors in KernelParams are a starting point, not an answer. The
right weights are estimated from history:

    r_i,e / vol_i  =  Σ_c  w_c · a_c(i, e)  +  ε

where a_c are the channel activations and r is the peer's realized abnormal
return over the event window. Ridge, because the channels are correlated by
construction (mechanism_sim shows up in three of them).

Two disciplines matter more than the estimator choice:

  * WALK-FORWARD. Weights used to trade event e are fitted only on events
    strictly before e, with an embargo gap so that overlapping event windows
    do not leak. `walk_forward_weights` enforces this.

  * ABNORMAL RETURNS. Regressing on raw peer returns will just recover the
    peer's beta to XBI on days when biotech happened to rally. Strip the
    sector return first.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..ontology import Ontology
from ..readthrough import CHANNELS, KernelParams, ReadThroughKernel
from ..schema import CatalystEvent
from ..surprise import event_day_vol, prior_pos, signed_surprise


@dataclass
class TrainingRow:
    t: datetime
    event_id: str
    ticker: str
    activations: np.ndarray     # (5,)
    y: float                    # abnormal return / event-day vol


def build_design(
    events: list[CatalystEvent],
    onto: Ontology,
    kernel: ReadThroughKernel,
    peer_returns: pd.DataFrame,     # index=event_id, columns=ticker, values=abnormal ret
    vols: dict[str, float],
    caps: dict[str, float],
) -> list[TrainingRow]:
    rows: list[TrainingRow] = []
    for ev in events:
        if ev.program_id not in onto.programs:
            continue
        if ev.event_id not in peer_returns.index:
            continue
        j = onto.programs[ev.program_id]
        s = signed_surprise(ev, prior_pos(j.phase, market_implied=j.pos_prior))
        if s == 0.0:
            continue
        for pid_i in onto.peers_of(ev.program_id):
            i = onto.programs[pid_i]
            if i.ticker == j.ticker:
                continue
            r = peer_returns.at[ev.event_id, i.ticker] if i.ticker in peer_returns.columns else np.nan
            if not np.isfinite(r):
                continue
            a = kernel.activations(pid_i, ev, s)
            v = event_day_vol(vols.get(i.ticker, 0.045), caps.get(i.ticker, 1.0))
            conc = i.ev_share ** 0.6
            rows.append(TrainingRow(
                t=ev.t_wire, event_id=ev.event_id, ticker=i.ticker,
                activations=a.vector() * conc,
                y=float(r) / max(v, 1e-4),
            ))
    return rows


def ridge_fit(
    rows: list[TrainingRow],
    alpha: float = 3.0,
    nonneg_prior: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Ridge with the prior as the shrinkage target rather than zero.

        w = argmin ||Xw - y||^2 + alpha * ||w - w0||^2

    Shrinking toward the structural prior instead of zero keeps the model
    sensible when a channel has few observations — which is always true for
    class-safety events, since they are rare.
    """
    if len(rows) < 30:
        raise ValueError(f"only {len(rows)} training rows; refusing to fit")

    X = np.vstack([r.activations for r in rows])
    y = np.array([r.y for r in rows])
    w0 = nonneg_prior if nonneg_prior is not None else np.zeros(X.shape[1])

    A = X.T @ X + alpha * np.eye(X.shape[1])
    b = X.T @ y + alpha * w0
    w = np.linalg.solve(A, b)

    resid = y - X @ w
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    # Per-channel t-like statistic using the ridge-adjusted covariance
    sigma2 = float((resid ** 2).sum()) / max(len(y) - X.shape[1], 1)
    cov = sigma2 * np.linalg.inv(A)
    se = np.sqrt(np.diag(cov))
    tstats = w / np.maximum(se, 1e-12)

    diag = {
        "n": len(y),
        "r2": r2,
        "weights": dict(zip(CHANNELS, [float(x) for x in w])),
        "tstats": dict(zip(CHANNELS, [float(x) for x in tstats])),
    }
    return w, diag


def walk_forward_weights(
    rows: list[TrainingRow],
    fit_dates: list[datetime],
    lookback_days: int = 730,
    embargo_days: int = 5,
    alpha: float = 3.0,
    prior: KernelParams | None = None,
) -> dict[datetime, np.ndarray]:
    """
    For each refit date, fit only on rows in
        [date - lookback, date - embargo)
    The embargo prevents a peer window that straddles the refit boundary from
    contributing to the weights used to trade that same window.
    """
    prior = prior or KernelParams()
    w0 = prior.vector()
    rows_sorted = sorted(rows, key=lambda r: r.t)
    ts = np.array([r.t.timestamp() for r in rows_sorted])

    out: dict[datetime, np.ndarray] = {}
    for d in fit_dates:
        hi = (d - timedelta(days=embargo_days)).timestamp()
        lo = (d - timedelta(days=lookback_days)).timestamp()
        sel = [r for r, t in zip(rows_sorted, ts) if lo <= t < hi]
        try:
            w, _ = ridge_fit(sel, alpha=alpha, nonneg_prior=w0)
        except ValueError:
            w = w0.copy()
        # Sign discipline: displacement and class_safety enter the kernel with
        # a negative sign already baked into the activation, so their weights
        # must stay positive. Clip rather than let a noisy sample flip them.
        w = np.maximum(w, 0.0)
        out[d] = w
    return out
