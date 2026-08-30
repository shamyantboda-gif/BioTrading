"""
The drug-development ontology and the pairwise peer graph.

The read-through model needs to answer, for any two programs i and j:
  - how similar is the biology?          -> mechanism_similarity
  - do they fight over the same patients? -> indication_overlap
  - who gets there first?                 -> lead_gap_months
  - is there a contractual link?          -> economic_link

Everything the cross-effect kernel does is a function of those four numbers
plus the market structure of the indication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schema import Indication, Phase, Program


# Modality families. Read-through within a family is strong for platform
# events ("first approval of an siRNA in the CNS") even across targets.
MODALITY_FAMILY: dict[str, str] = {
    "small_molecule": "sm",
    "peptide": "sm",
    "mab": "biologic",
    "bispecific": "biologic",
    "adc": "biologic",
    "fusion_protein": "biologic",
    "sirna": "oligo",
    "aso": "oligo",
    "mrna": "oligo",
    "aav": "genetic_medicine",
    "lnp_crispr": "genetic_medicine",
    "car_t": "cell_therapy",
    "tcr_t": "cell_therapy",
    "allogeneic_cell": "cell_therapy",
}

# Platform maturity in [0,1]. Immature platforms get much larger validation
# read-through: the first successful in-vivo CRISPR readout repriced the whole
# category, whereas the twentieth anti-PD-1 datapoint repriced nothing.
PLATFORM_MATURITY: dict[str, float] = {
    "sm": 0.95,
    "biologic": 0.85,
    "oligo": 0.55,
    "cell_therapy": 0.45,
    "genetic_medicine": 0.25,
}

# Pathway adjacency: targets that sit on the same biological axis. Used when
# targets differ but the mechanism read-through is still real.
PATHWAY: dict[str, str] = {
    "PD-1": "checkpoint", "PD-L1": "checkpoint", "CTLA-4": "checkpoint",
    "LAG-3": "checkpoint", "TIGIT": "checkpoint",
    "GLP-1R": "incretin", "GIPR": "incretin", "GCGR": "incretin",
    "AMYLIN": "incretin",
    "ABETA": "amyloid", "TAU": "neurodeg_proteinopathy",
    "ASYN": "neurodeg_proteinopathy", "LRRK2": "neurodeg_proteinopathy",
    "TTR": "amyloid",
    "FXI": "coagulation", "FXIa": "coagulation",
    "IL-23": "il_axis", "IL-17": "il_axis", "TL1A": "il_axis",
    "TYK2": "jak_tyk", "JAK1": "jak_tyk",
    "CD19": "b_cell", "BCMA": "plasma_cell", "CD20": "b_cell",
    "KRAS_G12C": "ras", "KRAS_G12D": "ras", "SOS1": "ras", "SHP2": "ras",
}

# Indication adjacency: distinct indications whose patients partly overlap or
# whose regulatory logic is shared.
INDICATION_FAMILY: dict[str, str] = {
    "nsclc_1l": "lung", "nsclc_2l": "lung", "sclc": "lung",
    "obesity": "cardiometabolic", "t2d": "cardiometabolic",
    "hfpef": "cardiometabolic", "nash": "cardiometabolic",
    "alzheimers_early": "neurodeg", "alzheimers_mild": "neurodeg",
    "parkinsons": "neurodeg", "als": "neurodeg",
    "uc": "ibd", "crohns": "ibd",
    "psoriasis": "derm_immuno", "atopic_derm": "derm_immuno",
    "attr_cm": "attr", "attr_pn": "attr",
    "dlbcl_3l": "b_cell_malignancy", "cll": "b_cell_malignancy",
    "mm_4l": "plasma_cell_malignancy",
}

PHASE_ORDER: dict[Phase, int] = {
    Phase.PRECLIN: 0, Phase.P1: 1, Phase.P1_2: 2, Phase.P2: 3,
    Phase.P2_3: 4, Phase.P3: 5, Phase.REGISTRATIONAL: 6, Phase.MARKETED: 7,
}


@dataclass(frozen=True)
class PairRelation:
    """Everything the read-through kernel needs about the ordered pair (i, j)."""

    mechanism_sim: float      # [0,1]
    safety_class_sim: float   # [0,1] — strictly target/pathway, not modality
    indication_overlap: float # [0,1]
    platform_sim: float       # [0,1]
    lead_gap_months: float    # >0 means i reaches market BEFORE j
    economic_link: float      # [-1,1]; +1 = i earns royalties on j
    same_company: bool


class Ontology:
    """Registry of programs, indications and contractual links."""

    def __init__(
        self,
        programs: Iterable[Program],
        indications: Iterable[Indication],
        economic_links: dict[tuple[str, str], float] | None = None,
    ) -> None:
        self.programs: dict[str, Program] = {p.program_id: p for p in programs}
        self.indications: dict[str, Indication] = {
            i.indication_id: i for i in indications
        }
        # keyed by (ticker_i, ticker_j) -> link strength in [-1,1]
        self.economic_links = economic_links or {}
        self._cache: dict[tuple[str, str], PairRelation] = {}

    # -- lookups ----------------------------------------------------------

    def by_ticker(self, ticker: str) -> list[Program]:
        return [p for p in self.programs.values() if p.ticker == ticker]

    def indication(self, indication_id: str) -> Indication:
        return self.indications.get(
            indication_id, Indication(indication_id, peak_sales_usd_bn=1.0)
        )

    def universe(self) -> list[str]:
        return sorted({p.ticker for p in self.programs.values()})

    # -- similarity components -------------------------------------------

    @staticmethod
    def _mechanism_similarity(a: Program, b: Program) -> float:
        """
        1.00  identical target and modality family (true me-too)
        0.80  identical target, different modality (e.g. siRNA vs ASO on TTR)
        0.55  same pathway, different target (PD-1 vs PD-L1)
        0.20  same modality family only
        0.00  unrelated
        """
        fam_a = MODALITY_FAMILY.get(a.modality, a.modality)
        fam_b = MODALITY_FAMILY.get(b.modality, b.modality)

        if a.target == b.target:
            return 1.0 if fam_a == fam_b else 0.80

        path_a, path_b = PATHWAY.get(a.target), PATHWAY.get(b.target)
        if path_a is not None and path_a == path_b:
            return 0.55

        return 0.20 if fam_a == fam_b else 0.0

    # Suffixes that denote a line of therapy or treatment setting rather than
    # a distinct disease. "nsclc_1l"/"nsclc_2l" are the same disease at
    # different lines; "attr_pn"/"attr_cm" are different diseases that happen
    # to share a protein. Conflating them badly overstates competition.
    LINE_SUFFIXES = {"1l", "2l", "3l", "4l", "5l", "adj", "neoadj", "maint",
                     "1st", "2nd", "frontline", "relapsed", "refractory"}

    @classmethod
    def _indication_overlap(cls, a: Program, b: Program) -> float:
        """
        1.00  same indication AND same line of therapy
        0.60  same disease, different line
        0.35  same therapeutic family
        0.00  unrelated
        """
        if a.indication == b.indication:
            return 1.0

        stem_a, _, suf_a = a.indication.rpartition("_")
        stem_b, _, suf_b = b.indication.rpartition("_")
        if (stem_a and stem_a == stem_b
                and suf_a in cls.LINE_SUFFIXES and suf_b in cls.LINE_SUFFIXES):
            return 0.60

        fam_a = INDICATION_FAMILY.get(a.indication)
        fam_b = INDICATION_FAMILY.get(b.indication)
        if fam_a is not None and fam_a == fam_b:
            return 0.35

        return 0.0

    @classmethod
    def _safety_class_similarity(cls, a: Program, b: Program) -> float:
        """
        Bug 2: a tox signal does NOT travel between two antibodies merely
        because both are antibodies. ARIA is an amyloid-clearance effect, not
        an IgG effect. Class safety transmits along the TARGET or PATHWAY, or
        along a genuinely platform-level toxicity (AAV hepatotoxicity, CAR-T
        cytokine release) — never along modality family alone.
        """
        mech = cls._mechanism_similarity(a, b)
        target_or_pathway = mech if mech >= 0.5 else 0.0
        platform_tox = cls._platform_similarity(a, b) * 0.9
        return max(target_or_pathway, platform_tox)

    @staticmethod
    def _platform_similarity(a: Program, b: Program) -> float:
        """
        Platform read-through is inversely weighted by platform maturity: a
        readout for an immature modality moves every peer using that modality.
        """
        fam_a = MODALITY_FAMILY.get(a.modality, a.modality)
        fam_b = MODALITY_FAMILY.get(b.modality, b.modality)
        if fam_a != fam_b:
            return 0.0
        return 1.0 - PLATFORM_MATURITY.get(fam_a, 0.9)

    # -- pair relation ----------------------------------------------------

    def relation(self, pid_i: str, pid_j: str) -> PairRelation:
        """Ordered relation: how program j's news should be read for program i."""
        key = (pid_i, pid_j)
        if key in self._cache:
            return self._cache[key]

        i, j = self.programs[pid_i], self.programs[pid_j]
        rel = PairRelation(
            mechanism_sim=self._mechanism_similarity(i, j),
            safety_class_sim=self._safety_class_similarity(i, j),
            indication_overlap=self._indication_overlap(i, j),
            platform_sim=self._platform_similarity(i, j),
            lead_gap_months=j.months_to_market - i.months_to_market,
            economic_link=self.economic_links.get((i.ticker, j.ticker), 0.0),
            same_company=(i.ticker == j.ticker),
        )
        self._cache[key] = rel
        return rel

    def peers_of(self, pid_j: str, min_relevance: float = 0.15) -> list[str]:
        """Programs whose price should react to news on program j."""
        out = []
        for pid_i in self.programs:
            if pid_i == pid_j:
                continue
            r = self.relation(pid_i, pid_j)
            relevance = max(
                r.mechanism_sim, r.indication_overlap,
                r.platform_sim, abs(r.economic_link),
            )
            if relevance >= min_relevance:
                out.append(pid_i)
        return out
