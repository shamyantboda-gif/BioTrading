"""
Real-data adapters.

Everything in this subpackage exists to turn free, public data sources into the
*exact* contracts the rest of the repo already speaks (`CatalystEvent`,
`Program`/`Ontology`, the ``{ticker: DataFrame}`` price panel, `MarketContext`).
Nothing here changes the model, the cost engine, or the backtester — those are
reused verbatim. If an adapter emits a shape the synthetic path did not, that is
a bug in the adapter, not a licence to fork the engine.

Read `RESEARCH_REALDATA.md` before trusting any number this path prints. The
short version: the only free, historical, structured catalyst sources
(openFDA, ClinicalTrials.gov) are exactly the ones `schema.SOURCE_LATENCY_SEC`
tags as day-scale *lagging*, so the engine correctly refuses to treat them as
tradeable signal. Demonstrating that cleanly is the point of this path.
"""

from __future__ import annotations
