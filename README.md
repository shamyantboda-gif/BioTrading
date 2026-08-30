# fda-alpha

Event-driven trading framework for FDA regulatory and clinical-trial catalysts, with a
webhook ingestion path, an automated execution layer, and a timestamp-strict backtester.

```
python run_backtest.py          # full pipeline on synthetic data
python explain_readthrough.py   # worked examples of the cross-effect model
python seed_sweep.py            # robustness across 12 independent histories
python tests/test_guards.py     # point-in-time and sign guards
```

Requires numpy, pandas, scipy. FastAPI is optional (the webhook falls back to stdlib).

---

## The core idea

The stock that issues the news gaps instantly. You will not be in front of that gap —
the release lands at 06:45 or 16:20, the name halts, and the first accessible print
already contains the information.

The *peer complex* is different. When a competitor posts data, a human has to decide
what it means for eleven other companies: whose mechanism was just validated, whose
market just shrank, whose safety profile just got questioned by association. That
decision takes minutes to hours and is frequently wrong, because the naive rule
("rival wins → I lose") is wrong about half the time.

That reaction lag is the only place in this trade where latency buys you anything. So
the system is built to fire on peers, and the backtest is built to prove you cannot
make money on the issuer.

---

## Read-through: does a rival's success help or hurt?

A competitor's win reaches you through five channels with conflicting signs. The kernel
(`fda_alpha/readthrough.py`) scores each separately.

| Channel | Fires when | Sign |
|---|---|---|
| **Validation** | shared target, pathway, or immature platform | **+** |
| **Displacement** | shared patients | **−** |
| **Class safety** | shared target/pathway, safety events only | **−** |
| **Precedent** | FDA accepts an endpoint or pathway you also plan to use | **+** |
| **Economic** | royalty, milestone, or equity exposure to the issuer | contract sign |

Three terms decide which channel wins.

**Market headroom.** This is what the question "will the price go up or down?" actually
turns on. A rival's win suppresses validation only to the extent the market is zero-sum:

```
overlap_discount = 1 − indication_overlap × (1 − headroom)
```

In 2L NSCLC (headroom ≈ 0.10) a rival's approval is nearly a direct transfer of your
revenue — validation is extinguished and displacement dominates. In obesity
(headroom ≈ 0.88) most patients are untreated, two drugs can both win, and a rival
proving the mechanism lifts the whole complex even though you compete head-on.

**Time-to-market gap.** A laggard rival's win barely dents an incumbent; a leader's win
badly damages someone three years behind. Enters as `tanh(gap / 18 months)`, spanning
roughly 0.45× to 1.55× on the displacement channel.

**EV concentration.** A 2%-of-enterprise-value program cannot move a large cap 15% no
matter how strong the read. Scales sub-linearly (`ev_share ** 0.6`) — diversified names
still react, just less.

### Worked output

Same event shape, opposite answers:

```
[1] Rival posts positive Ph3 in OBESITY (headroom 0.88)
                 peer   mech  indic   valid   displ   NET %
ORLX (GLP-1R/obesity)   0.55   1.00   +2.47   -2.02   +1.03    <- head-on rival, goes UP
AMYL (AMYLIN/obesity)   0.55   1.00   +3.02   -2.51   +1.24

[2] Rival posts positive Ph3 in 2L NSCLC (headroom 0.10)
                     peer   mech  indic   valid   displ   NET %
KRSX (KRAS_G12C/nsclc_2l)   0.00  1.00   +0.00   -9.38   -8.76  <- head-on rival, goes DOWN
     ONCV (PD-1/nsclc_2l)   0.20  1.00   +0.12   -1.07   -0.79

[3] Rival succeeds on the SAME target in a DIFFERENT disease
   GENE (TTR/attr_cm)      0.80   0.35   +3.33   -1.84   +1.91  <- validation dominates
```

Case [3] is the scenario in the original brief: same mechanism, different program. The
model says **up**, and says so because the validation channel survives when the
indication overlap is low.

Safety inverts the logic. On a rival's tox signal, class safety travels along the
*target or pathway* — never along modality family, since two unrelated antibodies are
not in the same safety class — while displacement flips positive because the rival
stumbled. A different-mechanism competitor in the same indication is a clean
beneficiary; a same-mechanism competitor is not, regardless of indication.

### The weights are fitted, not asserted

The hand-set priors are a starting point. Actual weights come from ridge regression of
sector-adjusted peer returns on channel activations, walk-forward with an embargo, and
shrunk toward the structural prior rather than toward zero (class-safety events are too
rare to fit from scratch). On the demo data:

```
n=2413  R2=0.0394
  validation     w=+0.505  t=+7.04
  displacement   w=+0.571  t=+6.38
  class_safety   w=+0.899  t=+9.41
  precedent      w=+0.369  t=+3.05
  economic       w=+0.685  t=+3.34
```

R² of 3–4% is what a real cross-sectional event model looks like. Anything much higher
means you have a leak.

### Surprise, not outcome

A phase 3 success in a program the market handicapped at 90% is worth nothing. Every
channel is driven by log-odds surprise relative to a prior probability of success,
taken from the options-implied straddle where available and from phase/indication base
rates otherwise. Trading raw outcomes instead of surprises is the second-fastest way to
lose money with this strategy.

---

## Backtesting: the four things that make naive results fake

`fda_alpha/backtest/` enforces these in code, not by convention.

**1. Timestamps.** `t_wire` is the wire's issuance instant, not when your HTTP client
finished reading the body. The decision path (`last_close_before`) refuses any bar at or
after the decision time; the execution path is a separate call whose output never
re-enters the model.

**2. Session and halt rolling.** Trade time is `t_wire + source_latency + system_latency`,
rolled forward to the next tradeable session and past any halt. Roughly 70% of releases
land outside regular hours by design. A Friday 16:20 release is not tradeable until
Monday if you don't trade extended hours — and the router that got this backwards
(resolving to the *same day's* 09:30 open) is exactly the bug class the guard tests
exist to catch.

**3. Source latency.** openFDA's Drugs@FDA endpoint is a batch ETL that lags the approval
press release by a day or more; ClinicalTrials.gov status edits post *after* the sponsor's
release. Both carry day-scale latency constants so the backtester cannot treat them as
signal. They are reconciliation and ontology-maintenance sources.

**4. Survivorship.** Biotech failure is the modal outcome. The universe comes from a
point-in-time listing table including delisted issuers; two names in the demo delist or
IPO mid-sample.

### What the backtest actually found

Out-of-sample, and stable across 12 independent simulated histories:

| Variant | Trades | Return | Sharpe (90% CI) | Dir. acc | Seeds positive |
|---|---|---|---|---|---|
| A  naive: issuer + peers, no cost gate | 499 | −24.7% | −9.5 [−12.1, −7.4] | 0.55 | **0 / 12** |
| B  peers only | 162 | +2.2% | 1.9 [−0.9, 4.0] | 0.60 | 6 / 12 |
| C  peers + edge-over-cost gate (1.5×) | 35 | +4.0% | 7.9 [4.8, 11.8] | 0.71 | — |
| D  peers + edge-over-cost gate (2.5×) | 19 | +3.4% | 10.8 [8.6, 15.1] | 0.84 | **11 / 12** |
| E  issuer leg added back | 124 | −4.9% | −4.7 [−10.2, −1.6] | 0.56 | — |

Three findings, in order of importance:

- **The issuer leg is not a trade.** Once the fill can't happen before the gap, it loses
  money in every history tested. Any backtest showing otherwise has a look-ahead leak —
  this one did, until the demo's price simulator was fixed to make the news bar open at
  the gapped price.
- **Direction alone doesn't pay.** Variant B calls direction correctly 60% of the time
  and still nets ~zero, because a 2% expected read-through into a 300bps pre-market
  spread is a losing trade however right you were.
- **The gate that matters is edge over estimated cost, not liquidity.** Requiring
  expected move ≥ 2.5× estimated round-trip cost rejects ~89% of signals and is the
  difference between the strategy and a donation to market makers. Liquidity filters are
  a crude proxy for the same thing.

Note the trade counts. Nineteen trades is not a track record; the bootstrap CI is
computed on ~15 distinct trading days and should be read as "not obviously negative"
rather than as a Sharpe estimate. Event strategies have far fewer independent bets than
they have trades, which is why every summary here reports an interval.

---

## Ingestion

```
PR wire push feed  ──┐
SEC EDGAR RSS      ──┤                                    ┌─ risk gates ─┐
FDA newsroom RSS   ──┼─► webhook /hook/{source} ─► parse ─┤              ├─► broker
ClinicalTrials.gov ──┤   (HMAC verified, deduped)  score  └─ kill switch ┘
openFDA Drugs@FDA  ──┘        (reconciliation only)
```

- **Dedupe on content hash, not URL.** The same release appears on the wire, the IR page,
  an 8-K, and openFDA hours later.
- **Record `t_ingest` separately and monitor the delta.** If median detection latency
  drifts from 1.5s to 9s, live fills diverge from every backtest you've run, silently.
- **Two-stage parsing.** The regex layer in `ingest/normalize.py` decides in microseconds
  and emits a confidence score; run an LLM classifier in parallel for the 3–8 second
  confirmation and flatten on disagreement. Negation patterns are tested before positive
  ones — a matcher that finds "met the primary endpoint" inside "did not meet the primary
  endpoint" is expensive.

## Execution

Defaults are paper-trading and `dry_run=True`. Pre-trade gates: parse confidence, minimum
conviction, stale-detection cutoff (if you're 30 seconds late the move is done and you're
buying someone's exit), move-already-happened cutoff, halt check, per-name and gross
notional caps, order rate limit, and a daily-loss kill switch that flattens. Extended-hours
orders must be limit orders — a market order sent pre-market queues to the auction and
fills at a price unrelated to your signal.

---

## Going live: what this repo does not have

The demo runs on synthetic data whose generating process contains read-through structure
by construction. **A good result there validates the plumbing, not the alpha.** Real
evaluation needs:

- **Timestamped news archive** with wire-level granularity (RavenPack, Bloomberg,
  Dow Jones). This is the binding constraint and it is not free. Scraped article
  timestamps are publication times, not dissemination times, and the difference is the
  entire trade.
- **Minute or tick TAQ history** including extended hours, plus NBBO for real spreads
  instead of the market-cap proxy in `_spread_proxy`.
- **Point-in-time index membership** with delisted issuers.
- **Borrow fee and locate history** — modeled here, not sourced.
- **Point-in-time ClinicalTrials.gov records.** Registry entries are edited
  retroactively; `poll_ctgov_history` reconstructs the record as of the decision date.
  Building features from today's record and testing against 2019 prices uses information
  that did not exist.
- **A hand-built ontology.** The `Program` graph — targets, indications with line of
  therapy, EV shares, time-to-market — is the actual moat and cannot be scraped. Roughly
  200 programs is enough to cover the liquid US biotech complex.

Other things worth knowing before committing capital: this is a well-populated trade and
the peer reaction lag has compressed as more participants automate it; capacity is bounded
by extended-hours liquidity in mid-cap biotech, which is thin; and the strategy's return
profile is short-gamma-like, with frequent small wins and occasional large losses when a
misparse or an unmodeled linkage fires. Position limits and the kill switch are load-bearing.

None of this is investment advice, and I'm not a financial advisor — it's a modeling and
engineering framework. Paper-trade it against live wire feeds for a few months before
considering anything else; that comparison, not the backtest, is what tells you whether
the detection latency and fill assumptions hold.

---

## Layout

```
fda_alpha/
  schema.py            normalized event/program contracts, source latency constants
  ontology.py          program graph; mechanism, indication, safety-class similarity
  readthrough.py       the five-channel cross-effect kernel
  surprise.py          log-odds surprise, POS priors, event-day vol
  signal.py            event -> sized, gated trade intents
  ingest/
    webhook.py         HMAC webhook, dedupe, CT.gov + openFDA pollers
    normalize.py       press release -> typed CatalystEvent
  execution/
    live.py            broker abstraction, risk gates, kill switch, run loop
  backtest/
    costs.py           sessions, halts, gap slippage, borrow
    engine.py          point-in-time event loop
    calibrate.py       walk-forward ridge fit of channel weights
    report.py          leg/session/channel attribution, bootstrap Sharpe CI
  data/synth.py        synthetic universe and market simulator
```
