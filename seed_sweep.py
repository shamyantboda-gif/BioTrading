"""Multi-seed robustness check. 17 trades on one seed proves nothing."""
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from fda_alpha.backtest.costs import ExecutionConfig
from fda_alpha.backtest.engine import BacktestConfig, Backtester, PriceBook
from fda_alpha.backtest.report import summarize
from fda_alpha.data.synth import (TICKER_META, build_universe, generate_events,
                                  listed_spans, market_context_fn, simulate_prices)
from fda_alpha.ontology import Ontology
from fda_alpha.readthrough import KernelParams, ReadThroughKernel
from fda_alpha.signal import SignalConfig, SignalEngine

end = datetime(2025, 12, 31, tzinfo=timezone.utc); start = end - timedelta(days=365*6)
programs, indications, links = build_universe()
onto = Ontology(programs, indications, links)
spans = listed_spans(start, end)

rows = []
for seed in range(12):
    ev = generate_events(programs, start, end, n_events=850, seed=seed)
    px = simulate_prices(programs, indications, links, ev, start, end, seed=100+seed)
    book, mkt = PriceBook(px), market_context_fn(px)
    split = start + (end - start) * 0.5
    test = [e for e in ev if e.t_wire >= split]
    for label, kw in [
        ("naive (own+peer, no gate)", dict(trade_own_leg=True, min_edge_over_cost=0.0)),
        ("peer only, no gate",        dict(trade_own_leg=False, min_edge_over_cost=0.0)),
        ("peer + edge/cost 2.5x",     dict(trade_own_leg=False, min_edge_over_cost=2.5)),
    ]:
        eng = SignalEngine(onto, ReadThroughKernel(onto, KernelParams()),
                           SignalConfig(own_leg_enabled=kw["trade_own_leg"]))
        bt = Backtester(onto, eng, book, ExecutionConfig(),
                        BacktestConfig(capital=5e6, seed=seed, **kw))
        tr = bt.run(test, mkt, listed_on=spans)
        s = summarize(tr, 5e6) if not tr.empty else {}
        rows.append(dict(seed=seed, variant=label, trades=s.get("n_trades",0),
                         ret_pct=s.get("total_return_pct",0.0),
                         dir_acc=s.get("directional_accuracy",np.nan),
                         avg_bps=s.get("avg_pnl_bps_of_notional",np.nan)))

df = pd.DataFrame(rows)
g = df.groupby("variant").agg(
    n_seeds=("seed","size"), med_trades=("trades","median"),
    mean_ret_pct=("ret_pct","mean"), med_ret_pct=("ret_pct","median"),
    std_ret_pct=("ret_pct","std"), pct_seeds_positive=("ret_pct", lambda s:(s>0).mean()),
    mean_dir_acc=("dir_acc","mean"), mean_bps=("avg_bps","mean"))
print("\nROBUSTNESS ACROSS 12 INDEPENDENT SIMULATED HISTORIES")
print("="*100)
print(g.to_string(float_format=lambda x: f"{x:,.3f}"))
