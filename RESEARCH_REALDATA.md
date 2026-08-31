# Real-data evaluation: what free data can and cannot tell you

*A research note on running the fda-alpha read-through strategy against real
catalysts and real prices. Companion to `run_realdata.py`.*

This note is written the way a quant trader should present a negative result:
state the hypothesis, source real data, run it through the same timestamp-strict
engine, and report what the data structurally cannot prove — rather than
tuning until a backtest looks good. The headline is not a Sharpe ratio.

---

## Hypothesis

The strategy's thesis (see `README.md`) is that the issuing stock gaps instantly
on an FDA/clinical catalyst, but its **peer complex** reprices with a lag of
minutes to hours as humans decide what a rival's result means. The tradeable
edge lives entirely in that intraday peer reaction-lag, in extended hours,
after a *wire-level* release timestamp.

**Falsifiable claim for this note:** using only free, public, historical data,
can we reconstruct that trade well enough to measure the edge?

**Answer: no — and the reason is structural, not a matter of effort.**

---

## Data actually sourced (all free, all live)

Run on 2020-01-01 → 2026-08-30, universe = 14 real US-listed biotech names
across three read-through complexes (incretin/obesity, NSCLC checkpoint+KRAS,
TTR amyloidosis; see `fda_alpha/realdata/ontology_real.py`).

| Layer | Source | What came back |
|---|---|---|
| Catalysts | openFDA Drugs@FDA | 10 original approvals (trustworthy dates) |
| Catalysts | ClinicalTrials.gov v2 | 40 directional status changes (terminations/suspensions) |
| Prices | Yahoo daily | 14/14 tickers, multi-year history |
| Prices | Yahoo 1-min + pre/post | trailing ~7 days only (Yahoo hard limit) |

Two data-quality decisions, both defensible in an interview:

1. **Supplemental approvals dropped by default.** openFDA batch-dates old
   supplements — e.g. eight Keytruda `SUPPL` records stamped the same day — so
   `submission_status_date` for supplements is an ETL artifact, not a real
   approval instant. Keeping them would have flooded the sample with 197
   near-duplicate "catalysts". Original approvals and trial-status edits carry
   real dates.
2. **Same-day, same-type events on a name collapse to one.** A batch of filings
   must not masquerade as many independent bets.

## The binding constraint, computed not asserted

`schema.SOURCE_LATENCY_SEC` tags the two free catalyst sources as day-scale
lagging: **openFDA ≈ 86,400 s (1 day), CT.gov ≈ 21,600 s (6 h)**. openFDA is a
batch ETL of Drugs@FDA that trails the approval press release; CT.gov status
edits post *after* the sponsor's own release. The backtester adds that latency
to `t_wire` before it will allow a fill, then rolls to the next tradeable
session. So **every free catalyst is already 6 hours to a day-plus public before
the engine can trade it** — past the entire window in which the peer lag exists.

This is the finding. The only free, historical, structured catalyst sources are
exactly the ones that lag the wire. You cannot backtest a latency edge with data
that is itself latent.

## Results (daily bars, free lagging catalysts)

```
                     variant  trades  ret_%  sharpe  dir_acc   sharpe 90% CI
A  naive: own+peers, no gate      39  -0.42   -4.29     0.51   [-7.8, -1.2]
B  peers only, no gate            17  -0.04   -1.13     0.53   [-4.9,  4.8]
C  peers + edge/cost 1.5x          2   0.10   18.49     1.00     — (n too small)
D  peers + edge/cost 2.5x          1   0.08     —       1.00     — (n too small)
```

Read it honestly:

- **Naive (A)** loses and calls direction at 0.51 — a coin flip. Consistent with
  the synthetic-data finding: without the latency edge, the issuer gap and costs
  dominate.
- **Peers-only (B)** is ~zero with a CI straddling zero. There is no measurable
  read-through edge left once the sources lag and the bars are daily.
- **Gated (C/D)** reject down to 1–2 trades. A Sharpe on one trade is not a
  number; it is a punchline. The gate works — it just has nothing to pass
  because the alpha window is gone before entry.

This **validates the machinery on real names** (fetch, align, point-in-time
vol, session/halt rolling, cost accounting, attribution all run on real data)
and **disproves nothing about the alpha**, because the data cannot address it.

## What we *can* measure: extended-hours microstructure

`python run_realdata.py --micro VKTX` pulls real 1-minute bars including
pre/post market for the trailing week:

```
EXTENDED-HOURS MICROSTRUCTURE - VKTX (7 trading days)
  1-min bars: 2594  regular=2196  pre-market=194  post-market=204
  overnight gap to pre-market open: abs-median 2.36%  max 4.07%
```

A ~2.4% median overnight gap to the pre-market open is the print the issuer
trade must beat and cannot — it is already in the first accessible price. The
peer leg must trade the slower complex reaction at this same one-minute,
extended-hours resolution. We can *see* this resolution for the last week; we
cannot get it for a 2021 catalyst from any free source. That gap is the entire
data-cost argument.

---

## What a real evaluation requires (and costs money)

| Need | Why | Free substitute here |
|---|---|---|
| Wire-level PR timestamps (RavenPack / Bloomberg / DJ) | the edge is sub-hour; scraped/ETL dates are hours-to-days late | openFDA/CT.gov (lagging) |
| Minute/tick TAQ incl. extended hours + NBBO | the trade lives pre-market; need real spreads | Yahoo daily + 7-day 1-min window; cap-based spread proxy |
| Point-in-time index membership w/ delisted names | biotech failure is modal; survivorship inflates everything | first/last available bar per ticker (partial) |
| Borrow-fee & locate history | the short side is small-cap biotech | static `REAL_TICKER_META` approximations |
| Point-in-time CT.gov records | registry entries are edited retroactively | current snapshot (has look-ahead) |

## Capacity (the other thing a QT interviewer will press)

Even with the data, capacity is bounded by **extended-hours liquidity in
mid-cap biotech, which is thin**. The cost engine already models a ~94% haircut
to available ADV pre-market (`premarket_liquidity_frac = 0.06`) and 3–4× spread
multipliers. On names like VKTX/NTLA/GPCR, a few hundred thousand dollars of
pre-market participation is enough to move the print into your own signal. The
strategy is real but small, and the peer reaction-lag has compressed as more
desks automate it. Position limits and the kill switch are load-bearing.

## Bottom line

The free-data pipeline is a faithful *engineering* demonstration on real names
and an honest *negative* research result: the edge is not measurable with data
that lags the wire and lacks intraday extended-hours granularity. The correct
next step is not more parameter tuning — it is one month of licensed wire feed
plus intraday TAQ, paper-traded live, compared against these fill assumptions.
That comparison, not any backtest, is what tells you whether the trade is real.

*None of this is investment advice.*
