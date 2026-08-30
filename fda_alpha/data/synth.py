"""
Synthetic universe + market simulator.

READ THIS BEFORE INTERPRETING ANY NUMBER THE DEMO PRINTS.

This module generates prices from a data-generating process that *contains*
read-through structure by construction. Running the backtest on it therefore
tests the plumbing — timestamp discipline, session rolling, halt handling,
sizing, cost accounting, attribution — and nothing else. A good Sharpe here
means the code works, not that the strategy works.

The ground-truth DGP deliberately differs from the model in ways real markets
do: the true weights are not the model's priors, roughly 45% of the peer
reaction is unmodelable noise, and a third of events carry a confounding
sector move. So the demo should show a *degraded* version of the structure,
which is the realistic case.

To evaluate the actual edge you need: a licensed timestamped news archive
(Ravenpack / Bloomberg / Dow Jones), minute or tick TAQ history, point-in-time
index membership including delisted issuers, and borrow-fee history.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..schema import (
    CatalystEvent, EventType, Indication, Phase, Program, SourceKind,
)

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def build_universe() -> tuple[list[Program], list[Indication], dict]:
    programs = [
        # --- Obesity / incretin: huge untreated market, high headroom -------
        Program("BIGP-glp1-obes", "BIGP", "bigpatide", "GLP-1R", "peptide",
                "obesity", Phase.MARKETED, ev_share=0.35, months_to_market=0, pos_prior=0.90),
        Program("METB-gipglp-obes", "METB", "metbatide", "GIPR", "peptide",
                "obesity", Phase.P3, ev_share=0.70, months_to_market=20, pos_prior=0.62),
        Program("ORLX-oralglp-obes", "ORLX", "orlixin", "GLP-1R", "small_molecule",
                "obesity", Phase.P2, ev_share=0.80, months_to_market=40, pos_prior=0.34),
        Program("AMYL-amylin-obes", "AMYL", "amylitide", "AMYLIN", "peptide",
                "obesity", Phase.P2, ev_share=0.75, months_to_market=42, pos_prior=0.30),
        Program("METB-gipglp-t2d", "METB", "metbatide", "GIPR", "peptide",
                "t2d", Phase.P3, ev_share=0.20, months_to_market=24, pos_prior=0.68),

        # --- Neurodegeneration: low headroom in some lines, immature bio ----
        Program("NRDG-tau-ad", "NRDG", "taulumab", "TAU", "mab",
                "alzheimers_early", Phase.P2, ev_share=0.85, months_to_market=44, pos_prior=0.22),
        Program("CNXS-abeta-ad", "CNXS", "abetamab", "ABETA", "mab",
                "alzheimers_early", Phase.P3, ev_share=0.55, months_to_market=14, pos_prior=0.48),
        Program("SYNU-asyn-pd", "SYNU", "synuclimab", "ASYN", "mab",
                "parkinsons", Phase.P2, ev_share=0.80, months_to_market=48, pos_prior=0.20),
        Program("LRKX-lrrk2-pd", "LRKX", "lrrkanib", "LRRK2", "small_molecule",
                "parkinsons", Phase.P2, ev_share=0.65, months_to_market=46, pos_prior=0.26),

        # --- Oncology: crowded, saturated 2L; earlier lines contested -------
        Program("ONCV-pd1-nsclc2", "ONCV", "oncavelimab", "PD-1", "mab",
                "nsclc_2l", Phase.P3, ev_share=0.45, months_to_market=16, pos_prior=0.52),
        Program("IMTX-tigit-nsclc1", "IMTX", "tigitumab", "TIGIT", "mab",
                "nsclc_1l", Phase.P3, ev_share=0.72, months_to_market=18, pos_prior=0.36),
        Program("KRSX-g12c-nsclc2", "KRSX", "krasinib", "KRAS_G12C", "small_molecule",
                "nsclc_2l", Phase.P2, ev_share=0.68, months_to_market=34, pos_prior=0.40),
        Program("ADCO-adc-nsclc2", "ADCO", "adcotecan", "TROP2", "adc",
                "nsclc_2l", Phase.P3, ev_share=0.60, months_to_market=15, pos_prior=0.50),

        # --- Genetic medicine: immature platform, big validation channel ----
        Program("GENE-crispr-attr", "GENE", "genecrisp", "TTR", "lnp_crispr",
                "attr_cm", Phase.P2, ev_share=0.88, months_to_market=38, pos_prior=0.35),
        Program("SILN-sirna-attr", "SILN", "silnaran", "TTR", "sirna",
                "attr_pn", Phase.P3, ev_share=0.62, months_to_market=12, pos_prior=0.66),
        Program("VECT-aav-hemb", "VECT", "vectagene", "FIX", "aav",
                "hemophilia_b", Phase.P3, ev_share=0.90, months_to_market=10, pos_prior=0.58),
    ]

    indications = [
        # headroom high -> a rival's win validates more than it displaces
        Indication("obesity", 90.0, headroom=0.88, winner_take_most=0.30),
        Indication("t2d", 60.0, headroom=0.45, winner_take_most=0.25),
        Indication("alzheimers_early", 30.0, headroom=0.92, winner_take_most=0.45),
        Indication("parkinsons", 12.0, headroom=0.95, winner_take_most=0.50),
        # headroom low -> zero-sum; a rival's win is a direct transfer
        Indication("nsclc_2l", 8.0, headroom=0.10, winner_take_most=0.75),
        Indication("nsclc_1l", 22.0, headroom=0.20, winner_take_most=0.80),
        Indication("attr_cm", 14.0, headroom=0.55, winner_take_most=0.65),
        Indication("attr_pn", 5.0, headroom=0.25, winner_take_most=0.70),
        Indication("hemophilia_b", 2.0, headroom=0.40, winner_take_most=0.85),
    ]

    # ORLX licensed its oral GLP-1 to BIGP: BIGP earns royalties, so BIGP
    # should RISE on ORLX good news despite competing in the same indication.
    economic_links = {("BIGP", "ORLX"): 0.55, ("ORLX", "BIGP"): 0.25}

    return programs, indications, economic_links


TICKER_META = {
    # ticker: (market_cap_bn, daily_vol, adv_usd, borrowable, borrow_bps, px0)
    "BIGP": (180.0, 0.016, 900e6, True, 30, 210.0),
    "METB": (12.0, 0.038, 120e6, True, 45, 88.0),
    "ORLX": (0.85, 0.075, 14e6, True, 320, 12.4),
    "AMYL": (0.42, 0.088, 6e6, True, 900, 7.1),
    "NRDG": (0.30, 0.095, 4e6, True, 1400, 4.6),
    "CNXS": (6.5, 0.045, 70e6, True, 60, 54.0),
    "SYNU": (0.55, 0.082, 7e6, True, 700, 9.3),
    "LRKX": (1.4, 0.064, 18e6, True, 210, 21.0),
    "ONCV": (22.0, 0.030, 210e6, True, 35, 76.0),
    "IMTX": (3.1, 0.055, 45e6, True, 120, 33.0),
    "KRSX": (1.1, 0.070, 15e6, True, 260, 16.5),
    "ADCO": (4.8, 0.050, 55e6, True, 90, 41.0),
    "GENE": (0.95, 0.086, 12e6, False, 2500, 13.8),
    "SILN": (7.2, 0.042, 80e6, True, 55, 62.0),
    "VECT": (0.65, 0.090, 8e6, True, 1100, 10.2),
}


# ---------------------------------------------------------------------------
# Ground-truth DGP (differs from the model on purpose)
# ---------------------------------------------------------------------------

TRUE_WEIGHTS = np.array([0.72, 0.61, 1.05, 0.22, 0.88])  # val, disp, safety, prec, econ
NOISE_FRAC = 0.45          # share of peer variance that is unmodelable
SECTOR_CONFOUND_PROB = 0.33


def generate_events(
    programs: list[Program],
    start: datetime,
    end: datetime,
    n_events: int = 260,
    seed: int = 11,
) -> list[CatalystEvent]:
    rng = np.random.default_rng(seed)
    pids = [p.program_id for p in programs]
    by_pid = {p.program_id: p for p in programs}

    etypes = [
        (EventType.TOPLINE_EFFICACY, 0.34),
        (EventType.APPROVAL, 0.10),
        (EventType.CRL, 0.05),
        (EventType.ADCOM, 0.06),
        (EventType.SAFETY_SIGNAL, 0.09),
        (EventType.INTERIM_ANALYSIS, 0.08),
        (EventType.CLINICAL_HOLD, 0.04),
        (EventType.DESIGNATION, 0.10),
        (EventType.PARTNERSHIP, 0.06),
        (EventType.PROGRAM_DISCONTINUATION, 0.04),
        (EventType.TRIAL_STATUS, 0.04),
    ]
    types, probs = zip(*etypes)
    probs = np.array(probs) / sum(probs)

    span_days = (end - start).days
    events = []
    for k in range(n_events):
        pid = pids[rng.integers(len(pids))]
        p = by_pid[pid]
        etype = types[rng.choice(len(types), p=probs)]

        # Realistic release timing: ~72% outside regular US hours, weighted
        # to pre-market 06:00-08:30 ET and post-close 16:05-18:00 ET.
        day = start + timedelta(days=int(rng.integers(span_days)))
        while day.weekday() >= 5:
            day += timedelta(days=1)
        u = rng.random()
        if u < 0.42:                      # pre-market
            hh, mm = 6 + int(rng.integers(3)), int(rng.integers(60))
        elif u < 0.72:                    # post-close
            hh, mm = 16 + int(rng.integers(2)), 5 + int(rng.integers(55))
        else:                             # intraday
            hh, mm = 10 + int(rng.integers(5)), int(rng.integers(60))
        t_wire_et = day.astimezone(ET).replace(hour=hh, minute=mm, second=int(rng.integers(60)),
                                               microsecond=0)
        t_wire = t_wire_et.astimezone(timezone.utc)

        if etype in (EventType.SAFETY_SIGNAL, EventType.CRL,
                     EventType.CLINICAL_HOLD, EventType.PROGRAM_DISCONTINUATION):
            polarity = -1.0
        elif etype == EventType.ADCOM:
            polarity = 1.0 if rng.random() < 0.62 else -1.0
        else:
            polarity = 1.0 if rng.random() < p.pos_prior else -1.0

        events.append(CatalystEvent(
            event_id=f"E{k:04d}",
            t_wire=t_wire,
            t_ingest=t_wire + timedelta(seconds=float(rng.gamma(2.0, 0.8))),
            source=SourceKind.PR_WIRE,
            event_type=etype,
            program_id=pid,
            ticker=p.ticker,
            polarity=polarity,
            strength=float(np.clip(rng.beta(4, 3), 0.05, 1.0)),
            confidence=float(np.clip(rng.beta(9, 1.5), 0.3, 1.0)),
            headline=f"{p.drug} {etype.value} readout",
        ))

    return sorted(events, key=lambda e: e.t_wire)


def simulate_prices(
    programs: list[Program],
    indications: list[Indication],
    economic_links: dict,
    events: list[CatalystEvent],
    start: datetime,
    end: datetime,
    seed: int = 23,
) -> dict[str, pd.DataFrame]:
    """
    Minute bars with catalyst-driven jumps injected at t_wire.

    Bars exist only in 04:00-20:00 ET on weekdays, which is what forces the
    backtester to roll an after-hours event forward to the next session.
    """
    from ..ontology import Ontology
    from ..readthrough import KernelParams, ReadThroughKernel
    from ..surprise import event_day_vol, prior_pos, signed_surprise

    rng = np.random.default_rng(seed)
    onto = Ontology(programs, indications, economic_links)
    truth = ReadThroughKernel(onto, KernelParams())

    # Bar timeline
    idx = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            base = d.astimezone(ET).replace(hour=4, minute=0, second=0, microsecond=0)
            for m in range(0, 16 * 60, 5):        # 5-minute bars, 04:00-20:00 ET
                idx.append((base + timedelta(minutes=m)).astimezone(timezone.utc))
        d += timedelta(days=1)
    idx = pd.DatetimeIndex(sorted(idx))

    tickers = sorted({p.ticker for p in programs})
    n = len(idx)

    # Baseline diffusion + a common sector factor
    sector = rng.normal(0, 0.0035, n).cumsum()
    prices: dict[str, np.ndarray] = {}
    for t in tickers:
        cap, dvol, adv, borrowable, bfee, px0 = TICKER_META[t]
        bar_vol = dvol / np.sqrt(78)
        beta = float(np.clip(0.9 + 0.5 * np.exp(-cap / 5.0), 0.7, 1.8))
        shocks = rng.normal(0, bar_vol, n) + beta * np.diff(sector, prepend=sector[0])
        prices[t] = px0 * np.exp(np.cumsum(shocks))

    # bars in which a ticker's OWN news lands; the open of such a bar must
    # already reflect the jump, otherwise the backtester can "buy the gap"
    # at the stale pre-news price -- the classic look-ahead leak.
    gapped_bars: dict[str, set[int]] = {}

    def nearest_bar(ts):
        k = idx.searchsorted(ts)
        return int(min(k, n - 1))

    for ev in events:
        j = onto.programs[ev.program_id]
        pos = prior_pos(j.phase, market_implied=j.pos_prior)
        s = signed_surprise(ev, pos)
        if s == 0.0:
            continue

        k = nearest_bar(ev.t_wire)

        # Own name jump
        cap_j, dvol_j, *_ = TICKER_META[j.ticker]
        v_own = event_day_vol(dvol_j, cap_j)
        own = truth.own_move(ev, s, v_own) * (1 + rng.normal(0, 0.30))
        own = float(np.clip(own, -0.92, 2.0))
        prices[j.ticker][k:] *= (1 + own)
        gapped_bars.setdefault(j.ticker, set()).add(k)

        # Sector confound: sometimes the whole complex moves for unrelated
        # reasons on the same bar, which is exactly what makes naive event
        # studies overstate read-through.
        if rng.random() < SECTOR_CONFOUND_PROB:
            shock = rng.normal(0, 0.012)
            for t in tickers:
                prices[t][k:] *= (1 + shock)

        # Peer read-through, using TRUE weights + heavy noise
        for pid_i in onto.peers_of(ev.program_id):
            i = onto.programs[pid_i]
            if i.ticker == j.ticker:
                continue
            cap_i, dvol_i, *_ = TICKER_META[i.ticker]
            v_i = event_day_vol(dvol_i, cap_i)
            a = truth.activations(pid_i, ev, s).vector()
            raw = float(np.dot(a, TRUE_WEIGHTS)) * (i.ev_share ** 0.6)
            signal_part = np.clip(raw * v_i, -0.5, 0.5)
            noise = rng.normal(0, abs(signal_part) * NOISE_FRAC + v_i * 0.35)
            move = float(np.clip(signal_part + noise, -0.55, 0.55))

            # Peers reprice over ~20-70 minutes, not instantly. That lag is
            # the only reason this strategy has anywhere to make money.
            lag_bars = int(rng.integers(4, 14))
            for b in range(lag_bars):
                kk = min(k + b, n - 1)
                prices[i.ticker][kk:] *= (1 + move / lag_bars)

    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        c = prices[t]
        o = np.concatenate([[c[0]], c[:-1]])
        for kk in gapped_bars.get(t, ()):        # no pre-news open on a gap bar
            o[kk] = c[kk]
        cap, dvol, adv, *_ = TICKER_META[t]
        spread = float(np.clip(120.0 / max(cap, 0.05) ** 0.5, 4.0, 400.0))
        df = pd.DataFrame({
            "open": o,
            "high": np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.001, n))),
            "low": np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.001, n))),
            "close": c,
            "volume": np.maximum(rng.lognormal(np.log(adv / 78 / np.maximum(c, 1.0)), 0.6), 1.0),
            "spread_bps": np.full(n, spread),
        }, index=idx)
        out[t] = df
    return out


def market_context_fn(prices: dict[str, pd.DataFrame]):
    """
    Point-in-time MarketContext builder. Uses only bars strictly before t.
    """
    from ..signal import MarketContext

    def fn(t: datetime) -> MarketContext:
        vols, caps, advs, borrow_ok, borrow_fee = {}, {}, {}, {}, {}
        for tk, meta in TICKER_META.items():
            cap, dvol, adv, ok, fee, _ = meta
            df = prices[tk]
            k = df.index.searchsorted(t) - 1
            if k > 80:
                r = np.diff(np.log(df["close"].to_numpy()[max(0, k - 780):k + 1]))
                dvol = float(np.std(r) * np.sqrt(78)) if len(r) > 30 else dvol
            vols[tk], caps[tk], advs[tk] = dvol, cap, adv
            borrow_ok[tk], borrow_fee[tk] = ok, float(fee)
        return MarketContext(vols, caps, advs, borrow_ok, borrow_fee)

    return fn


def listed_spans(start: datetime, end: datetime) -> dict[str, tuple[datetime, datetime]]:
    """
    Point-in-time listing table. Two names delist mid-sample so the engine has
    to handle it — in a real universe this is where survivorship bias enters.
    """
    spans = {t: (start, end) for t in TICKER_META}
    spans["NRDG"] = (start, start + (end - start) * 0.72)   # fails, delists
    spans["SYNU"] = (start + (end - start) * 0.15, end)     # IPOs mid-sample
    return spans
