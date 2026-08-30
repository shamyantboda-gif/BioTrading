"""
Worked examples of the cross-effect kernel: when does a rival's win help you?
"""
from datetime import datetime, timezone
import pandas as pd
from fda_alpha.data.synth import TICKER_META, build_universe
from fda_alpha.ontology import Ontology
from fda_alpha.readthrough import KernelParams, ReadThroughKernel
from fda_alpha.schema import CatalystEvent, EventType, SourceKind
from fda_alpha.surprise import event_day_vol, prior_pos, signed_surprise

programs, indications, links = build_universe()
onto = Ontology(programs, indications, links)
k = ReadThroughKernel(onto, KernelParams())
t = datetime(2025, 6, 3, 11, 0, tzinfo=timezone.utc)

def ev(pid, etype, pol, strength=0.85):
    p = onto.programs[pid]
    return CatalystEvent("X", t, t, SourceKind.PR_WIRE, etype, pid, p.ticker,
                         polarity=pol, strength=strength, confidence=1.0)

def show(title, e, peers):
    j = onto.programs[e.program_id]
    s = signed_surprise(e, prior_pos(j.phase, market_implied=j.pos_prior))
    print(f"\n{title}")
    print(f"  event: {j.ticker} {j.drug} ({j.target}, {j.indication}) "
          f"{e.event_type.value} pol={e.polarity:+.0f}   signed_surprise={s:+.2f}")
    rows=[]
    for pid in peers:
        i = onto.programs[pid]
        cap, dvol, *_ = TICKER_META[i.ticker]
        v = event_day_vol(dvol, cap, is_binary_catalyst=False)
        mv, br = k.peer_move(pid, e, s, v)
        rel = onto.relation(pid, e.program_id)
        rows.append({
            "peer": f"{i.ticker} ({i.target}/{i.indication})",
            "mech": round(rel.mechanism_sim,2), "indic": round(rel.indication_overlap,2),
            "headroom": round(onto.indication(i.indication).headroom,2),
            "lead_gap_mo": round(rel.lead_gap_months,0),
            "valid": round(br.get("validation",0)*100,2),
            "displ": round(br.get("displacement",0)*100,2),
            "safety": round(br.get("class_safety",0)*100,2),
            "econ": round(br.get("economic",0)*100,2),
            "NET %": round(mv*100,2),
        })
    print(pd.DataFrame(rows).to_string(index=False))

# 1. The exact question: rival's drug excels; peers have their own good programs.
show("[1] METB posts positive Ph3 in OBESITY (headroom 0.88, big untapped market)",
     ev("METB-gipglp-obes", EventType.TOPLINE_EFFICACY, +1),
     ["BIGP-glp1-obes","ORLX-oralglp-obes","AMYL-amylin-obes","METB-gipglp-t2d"])

# 2. Same event shape, saturated zero-sum market
show("[2] ADCO posts positive Ph3 in NSCLC 2L (headroom 0.10, zero-sum)",
     ev("ADCO-adc-nsclc2", EventType.TOPLINE_EFFICACY, +1),
     ["ONCV-pd1-nsclc2","KRSX-g12c-nsclc2","IMTX-tigit-nsclc1"])

# 3. Same mechanism, DIFFERENT indication -> pure validation
show("[3] SILN (siRNA/TTR, ATTR-PN) positive Ph3 — read into TTR peers",
     ev("SILN-sirna-attr", EventType.TOPLINE_EFFICACY, +1),
     ["GENE-crispr-attr","VECT-aav-hemb"])

# 4. Safety: class effect vs competitive relief
show("[4] CNXS (anti-amyloid) reports a SAFETY SIGNAL in early Alzheimer's",
     ev("CNXS-abeta-ad", EventType.SAFETY_SIGNAL, -1),
     ["NRDG-tau-ad","SYNU-asyn-pd","LRKX-lrrk2-pd"])
