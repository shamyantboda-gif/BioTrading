"""
A hand-built ontology of real, liquid US-listed biotech programs.

The README calls the program graph "the actual moat ... cannot be scraped".
This is a deliberately small, defensible slice of it — three complexes chosen
because their read-through structure is real and well documented:

* **Incretin / obesity** — high market headroom (most patients untreated), so a
  rival's win *validates* the class more than it displaces. LLY and NVO are
  marketed anchors; VKTX / GPCR / ALT / AMGN are the challengers.
* **NSCLC checkpoint + KRAS** — low headroom, crowded 2L. A rival's win is close
  to a direct transfer. PD-1 (checkpoint) and KRAS_G12C (ras) programs.
* **TTR amyloidosis** — the textbook validation case: one target (TTR) across
  four modalities (siRNA, ASO, stabilizer, CRISPR) and two *different* diseases
  (attr_cm vs attr_pn). Same mechanism, low indication overlap -> validation
  survives.

Every ``target`` here is a key in ``ontology.PATHWAY``, every ``indication`` is
a key in ``ontology.INDICATION_FAMILY``, and every ``modality`` is a key in
``ontology.MODALITY_FAMILY`` — so the existing similarity logic works unchanged.

Market-cap / borrow figures in ``REAL_TICKER_META`` are approximate, point-in-
time-agnostic order-of-magnitude values used only for sizing and the spread
proxy; realized vol and ADV are computed from price history at run time. They
are NOT a point-in-time fundamentals feed and should not be read as one.
"""

from __future__ import annotations

from ..schema import Indication, Phase, Program


def build_universe() -> tuple[list[Program], list[Indication], dict]:
    programs = [
        # ---- Incretin / obesity: high headroom, rival win validates ---------
        Program("LLY-tirze-obes", "LLY", "tirzepatide", "GIPR", "peptide",
                "obesity", Phase.MARKETED, ev_share=0.30, months_to_market=0, pos_prior=0.95),
        Program("NVO-sema-obes", "NVO", "semaglutide", "GLP-1R", "peptide",
                "obesity", Phase.MARKETED, ev_share=0.45, months_to_market=0, pos_prior=0.95),
        Program("VKTX-vk2735-obes", "VKTX", "vk2735", "GLP-1R", "peptide",
                "obesity", Phase.P3, ev_share=0.85, months_to_market=30, pos_prior=0.50),
        Program("GPCR-oral-obes", "GPCR", "aleniglipron", "GLP-1R", "small_molecule",
                "obesity", Phase.P2, ev_share=0.80, months_to_market=42, pos_prior=0.38),
        Program("ALT-pemvi-obes", "ALT", "pemvidutide", "GCGR", "peptide",
                "obesity", Phase.P2, ev_share=0.70, months_to_market=44, pos_prior=0.34),
        Program("AMGN-marit-obes", "AMGN", "maridebart", "GIPR", "peptide",
                "obesity", Phase.P2, ev_share=0.12, months_to_market=40, pos_prior=0.45),

        # ---- NSCLC: checkpoint + KRAS, low headroom, near zero-sum ----------
        Program("MRK-pembro-nsclc1", "MRK", "pembrolizumab", "PD-1", "mab",
                "nsclc_1l", Phase.MARKETED, ev_share=0.40, months_to_market=0, pos_prior=0.95),
        Program("BMY-nivo-nsclc1", "BMY", "nivolumab", "PD-1", "mab",
                "nsclc_1l", Phase.MARKETED, ev_share=0.20, months_to_market=0, pos_prior=0.95),
        Program("SMMT-ivo-nsclc1", "SMMT", "ivonescimab", "PD-1", "bispecific",
                "nsclc_1l", Phase.P3, ev_share=0.80, months_to_market=18, pos_prior=0.50),
        Program("AMGN-soto-nsclc2", "AMGN", "sotorasib", "KRAS_G12C", "small_molecule",
                "nsclc_2l", Phase.MARKETED, ev_share=0.08, months_to_market=0, pos_prior=0.90),
        Program("BMY-adagra-nsclc2", "BMY", "adagrasib", "KRAS_G12C", "small_molecule",
                "nsclc_2l", Phase.MARKETED, ev_share=0.06, months_to_market=0, pos_prior=0.85),
        Program("RVMD-daraxo-nsclc2", "RVMD", "daraxonrasib", "KRAS_G12D", "small_molecule",
                "nsclc_2l", Phase.P3, ev_share=0.85, months_to_market=28, pos_prior=0.45),

        # ---- TTR amyloidosis: one target, four modalities, two diseases ----
        Program("ALNY-vutri-attrcm", "ALNY", "vutrisiran", "TTR", "sirna",
                "attr_cm", Phase.MARKETED, ev_share=0.55, months_to_market=0, pos_prior=0.90),
        Program("IONS-eplon-attrpn", "IONS", "eplontersen", "TTR", "aso",
                "attr_pn", Phase.MARKETED, ev_share=0.30, months_to_market=0, pos_prior=0.85),
        Program("BBIO-acora-attrcm", "BBIO", "acoramidis", "TTR", "small_molecule",
                "attr_cm", Phase.MARKETED, ev_share=0.80, months_to_market=0, pos_prior=0.85),
        Program("NTLA-nexz-attrcm", "NTLA", "nexiguran", "TTR", "lnp_crispr",
                "attr_cm", Phase.P3, ev_share=0.70, months_to_market=24, pos_prior=0.45),
    ]

    indications = [
        Indication("obesity", 100.0, headroom=0.88, winner_take_most=0.30),
        Indication("nsclc_1l", 22.0, headroom=0.20, winner_take_most=0.80),
        Indication("nsclc_2l", 8.0, headroom=0.10, winner_take_most=0.75),
        Indication("attr_cm", 14.0, headroom=0.55, winner_take_most=0.65),
        Indication("attr_pn", 5.0, headroom=0.25, winner_take_most=0.70),
    ]

    # No fabricated contractual links. Add real royalty/milestone terms here if
    # you have them; inventing them would be exactly the kind of unsourced
    # number the README warns against.
    economic_links: dict[tuple[str, str], float] = {}

    return programs, indications, economic_links


# Approximate market-cap ($bn), borrowability and borrow fee (bps annual).
# Order-of-magnitude, static, and clearly not point-in-time. Large caps borrow
# cheaply; small-cap challengers are pricier and less certain to locate.
REAL_TICKER_META: dict[str, tuple[float, bool, float]] = {
    #        cap_bn, borrowable, borrow_bps
    "LLY":  (700.0, True, 30.0),
    "NVO":  (350.0, True, 30.0),
    "MRK":  (220.0, True, 30.0),
    "BMY":  (100.0, True, 35.0),
    "AMGN": (150.0, True, 30.0),
    "ALNY": (45.0,  True, 45.0),
    "IONS": (7.0,   True, 60.0),
    "BBIO": (8.0,   True, 90.0),
    "VKTX": (3.0,   True, 250.0),
    "RVMD": (8.0,   True, 120.0),
    "SMMT": (18.0,  True, 150.0),
    "NTLA": (2.5,   True, 400.0),
    "GPCR": (1.5,   True, 300.0),
    "ALT":  (0.6,   True, 800.0),
}


# Per-program catalyst search terms, used by the openFDA and ClinicalTrials.gov
# adapters to attribute a real filing/trial back to a program. Kept next to the
# ontology so the two never drift apart.
PROGRAM_QUERIES: dict[str, dict[str, str]] = {
    "LLY-tirze-obes":    {"generic": "tirzepatide",  "sponsor": "Eli Lilly"},
    "NVO-sema-obes":     {"generic": "semaglutide",  "sponsor": "Novo Nordisk"},
    "VKTX-vk2735-obes":  {"generic": "VK2735",       "sponsor": "Viking Therapeutics"},
    "GPCR-oral-obes":    {"generic": "aleniglipron", "sponsor": "Structure Therapeutics"},
    "ALT-pemvi-obes":    {"generic": "pemvidutide",  "sponsor": "Altimmune"},
    "AMGN-marit-obes":   {"generic": "maridebart",   "sponsor": "Amgen"},
    "MRK-pembro-nsclc1": {"generic": "pembrolizumab", "sponsor": "Merck"},
    "BMY-nivo-nsclc1":   {"generic": "nivolumab",    "sponsor": "Bristol-Myers Squibb"},
    "SMMT-ivo-nsclc1":   {"generic": "ivonescimab",  "sponsor": "Summit Therapeutics"},
    "AMGN-soto-nsclc2":  {"generic": "sotorasib",    "sponsor": "Amgen"},
    "BMY-adagra-nsclc2": {"generic": "adagrasib",    "sponsor": "Mirati"},
    "RVMD-daraxo-nsclc2": {"generic": "daraxonrasib", "sponsor": "Revolution Medicines"},
    "ALNY-vutri-attrcm": {"generic": "vutrisiran",   "sponsor": "Alnylam"},
    "IONS-eplon-attrpn": {"generic": "eplontersen",  "sponsor": "Ionis"},
    "BBIO-acora-attrcm": {"generic": "acoramidis",   "sponsor": "BridgeBio"},
    "NTLA-nexz-attrcm":  {"generic": "nexiguran",    "sponsor": "Intellia"},
}


def ticker_caps() -> dict[str, float]:
    """Ticker -> approximate market cap ($bn), for the price adapter's spread proxy."""
    return {t: meta[0] for t, meta in REAL_TICKER_META.items()}
