"""
Turning a press release into a typed CatalystEvent.

The rule-based layer below is fast (microseconds) and covers the standard
phrasings. It emits a `confidence` score; anything below the SignalConfig
floor does not trade. That gate is the point — a fast wrong classification is
worse than no classification.

Two failure modes worth designing against:

  * "MET THE PRIMARY ENDPOINT" in a headline about a *secondary* cohort, or in
    a sentence that continues "...but did not meet the key secondary endpoint
    of overall survival." Headline-only parsing gets these backwards. Require
    corroboration from the first body paragraph before assigning strength.

  * Negation and hedging: "did not meet", "failed to demonstrate", "was not
    statistically significant". Order the patterns so negatives are tested
    first; a substring matcher that finds "met the primary endpoint" inside
    "did not meet the primary endpoint" will lose you a lot of money.

For production, run this rule layer for the sub-second decision and an LLM
classifier in parallel for the 3-8 second confirmation, then reconcile: if the
LLM disagrees, flatten. The rule layer's job is speed; the model's job is
being right.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from ..schema import CatalystEvent, EventType, SourceKind

# Order matters: negatives before positives.
_PATTERNS: list[tuple[re.Pattern, EventType, float, float]] = [
    # (regex, type, polarity, strength)
    (re.compile(r"\b(did not|failed to|does not) (meet|achieve|demonstrate)\b", re.I),
     EventType.TOPLINE_EFFICACY, -1.0, 0.85),
    (re.compile(r"\bnot statistically significant\b", re.I),
     EventType.TOPLINE_EFFICACY, -1.0, 0.80),
    (re.compile(r"\bcomplete response letter\b|\bCRL\b", re.I),
     EventType.CRL, -1.0, 0.90),
    (re.compile(r"\bclinical hold\b", re.I),
     EventType.CLINICAL_HOLD, -1.0, 0.85),
    (re.compile(r"\bdiscontinu\w+ (the )?(development|program)\b", re.I),
     EventType.PROGRAM_DISCONTINUATION, -1.0, 0.90),
    (re.compile(r"\b(patient death|treatment-related death|grade 5)\b", re.I),
     EventType.SAFETY_SIGNAL, -1.0, 0.95),
    (re.compile(r"\b(serious adverse events?|hepatotox\w+|drug-induced liver injury)\b", re.I),
     EventType.SAFETY_SIGNAL, -1.0, 0.70),
    (re.compile(r"\bterminat\w+ (the )?(trial|study)\b", re.I),
     EventType.TRIAL_STATUS, -1.0, 0.75),

    (re.compile(r"\bapprov\w+ (of|for)?\s*\w*\s*(NDA|BLA|sNDA|sBLA)\b|"
                r"\bFDA approves\b|\breceives? FDA approval\b", re.I),
     EventType.APPROVAL, 1.0, 0.90),
    (re.compile(r"\b(met|achieved) (the |its |all )?(co-)?primary endpoint", re.I),
     EventType.TOPLINE_EFFICACY, 1.0, 0.85),
    (re.compile(r"\bstatistically significant\b.*\bimprovement\b", re.I),
     EventType.TOPLINE_EFFICACY, 1.0, 0.75),
    (re.compile(r"\bstopped early for (overwhelming )?efficacy\b", re.I),
     EventType.INTERIM_ANALYSIS, 1.0, 0.95),
    (re.compile(r"\bfutility\b", re.I),
     EventType.INTERIM_ANALYSIS, -1.0, 0.90),
    (re.compile(r"\badvisory committee\b.*\bvoted?\b", re.I),
     EventType.ADCOM, 0.0, 0.60),
    (re.compile(r"\b(breakthrough therapy|fast track|orphan drug|RMAT) designation\b", re.I),
     EventType.DESIGNATION, 1.0, 0.35),
    (re.compile(r"\bpriority review\b", re.I),
     EventType.PDUFA_SHIFT, 1.0, 0.40),
    (re.compile(r"\bextend\w+ the (PDUFA|review) (date|period)\b", re.I),
     EventType.PDUFA_SHIFT, -1.0, 0.55),
    (re.compile(r"\b(license|collaboration) agreement\b.*\$\s?\d", re.I),
     EventType.PARTNERSHIP, 1.0, 0.55),
]

_ADCOM_VOTE = re.compile(r"\bvoted?\s+(\d{1,2})\s*(?:-|to)\s*(\d{1,2})\b", re.I)

# Strength boosters found in the body.
_BOOSTERS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\boverall survival\b", re.I), 0.20),
    (re.compile(r"\bp\s*[<=]\s*0?\.000\d", re.I), 0.15),
    (re.compile(r"\bhazard ratio[^.]{0,30}0\.[0-5]\d", re.I), 0.15),
    (re.compile(r"\bhighly statistically significant\b", re.I), 0.10),
]
_DAMPENERS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bdid not (meet|achieve)[^.]{0,60}\bsecondary\b", re.I), -0.25),
    (re.compile(r"\bnumerical(ly)? (improvement|trend)\b", re.I), -0.20),
    (re.compile(r"\bexploratory\b|\bpost-hoc\b", re.I), -0.20),
    (re.compile(r"\bsubgroup\b", re.I), -0.15),
    (re.compile(r"\bsingle-arm\b|\bopen-label\b", re.I), -0.10),
]


def parse_release(
    headline: str,
    body: str,
    ticker: str,
    program_id: str,
    t_wire: datetime,
    source: SourceKind = SourceKind.PR_WIRE,
    t_ingest: datetime | None = None,
) -> CatalystEvent | None:
    text = f"{headline}\n{body[:2000]}"
    hit = None
    for pat, etype, pol, strength in _PATTERNS:
        if pat.search(headline) or pat.search(body[:600]):
            hit = (pat, etype, pol, strength)
            break
    if hit is None:
        return None

    _, etype, polarity, strength = hit
    confidence = 0.72

    # Headline match is worth more than a body match.
    if hit[0].search(headline):
        confidence += 0.18

    # AdCom: recover the actual vote margin.
    if etype == EventType.ADCOM:
        m = _ADCOM_VOTE.search(text)
        if m:
            yes, no = int(m.group(1)), int(m.group(2))
            total = max(yes + no, 1)
            polarity = 1.0 if yes > no else -1.0
            strength = min(1.0, abs(yes - no) / total + 0.25)
            confidence += 0.10

    for pat, delta in _BOOSTERS:
        if pat.search(text):
            strength += delta
    for pat, delta in _DAMPENERS:
        if pat.search(text):
            strength += delta
            confidence -= 0.06

    strength = max(0.05, min(1.0, strength))
    confidence = max(0.0, min(1.0, confidence))

    return CatalystEvent(
        event_id=CatalystEvent.make_id(source, t_wire, headline),
        t_wire=t_wire,
        t_ingest=t_ingest or datetime.now(timezone.utc),
        source=source,
        event_type=etype,
        program_id=program_id,
        ticker=ticker,
        polarity=polarity,
        strength=strength,
        confidence=confidence,
        headline=headline,
        raw={"body_head": body[:500]},
    )
