"""
Final scoring.

This file combines all candidate signals into one score.

Basic idea:
1. base_fit = weighted average of job-fit signals.
2. final_score = base_fit multiplied by penalties/bonuses.

Penalties catch things like services-only careers, wrong AI specialty, stopped
coding, or impossible/fake profile details.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from . import jd_spec as JD
from .honeypot import impossibility_report, honeypot_penalty


def base_fit(feat: Dict[str, Any], semantic: float) -> float:
    """Calculate the main job-fit score before penalties."""
    w = JD.WEIGHTS
    return (
        w["title_career"] * feat["title_career"]
        + w["career_evidence"] * feat["career_evidence"]
        + w["skills_trust"] * feat["skills_trust"]
        + w["semantic"] * semantic
        + w["experience"] * feat["experience"]
        + w["location"] * feat["location"]
        + w["education"] * feat["education"]
    )


def gate_multipliers(cand: Dict[str, Any], feat: Dict[str, Any]) -> Dict[str, float]:
    """Calculate bonuses and penalties that adjust the base score."""
    mults: Dict[str, float] = {}

    # Pure services-company careers are a weaker fit for this product role.
    # Continuous penalty based on fraction of career spent at services firms
    services_frac = feat.get("services_frac", 0.0)
    mults["services_penalty"] = 1.0 - 0.45 * services_frac

    # CV/speech-only AI is less relevant when there is no NLP/IR signal.
    mults["distractor_penalty"] = 0.7 if feat.get("distractor_dominated") else 1.0

    # This role wants a senior who still writes code.
    mults["coding_recency_penalty"] = 0.8 if feat.get("stopped_coding") else 1.0

    # Product-company experience is a small positive signal.
    mults["product_bonus"] = 1.04 if feat.get("has_product_company") else 1.0

    # Impossible profile details get a large penalty.
    n_markers, reasons = impossibility_report(cand)
    mults["honeypot_penalty"] = honeypot_penalty(n_markers)
    mults["_honeypot_markers"] = n_markers
    mults["_honeypot_reasons"] = reasons
    return mults


def score_candidate(
    cand: Dict[str, Any],
    feat: Dict[str, Any],
    semantic: float,
    behavioral: Dict[str, Any],
) -> Dict[str, Any]:
    """Return all score details for one candidate."""
    fit = base_fit(feat, semantic)
    gates = gate_multipliers(cand, feat)

    final = (
        fit
        * behavioral["behavioral_mult"]
        * gates["services_penalty"]
        * gates["distractor_penalty"]
        * gates["coding_recency_penalty"]
        * gates["product_bonus"]
        * gates["honeypot_penalty"]
    )

    return {
        "candidate_id": cand["candidate_id"],
        "final_score": float(final),
        "base_fit": float(fit),
        "semantic": float(semantic),
        # Keep useful details so reasoning.py can explain the rank.
        **{k: v for k, v in feat.items()
           if k in ("title_career", "career_evidence", "skills_trust",
                    "experience", "location", "education", "yoe",
                    "matched_core_skills", "n_core_skills", "evidence_hits",
                    "nontech_fraction", "has_product_company", "services_only",
                    "services_frac", "distractor_dominated", "nlp_ir_present",
                    "willing_to_relocate", "stopped_coding")},
        **behavioral,
        "honeypot_markers": gates["_honeypot_markers"],
        "honeypot_reasons": gates["_honeypot_reasons"],
        "services_penalty": gates["services_penalty"],
        "distractor_penalty": gates["distractor_penalty"],
        "coding_recency_penalty": gates["coding_recency_penalty"],
    }


def normalise_scores(scored: List[Dict[str, Any]]) -> None:
    """
    Convert raw scores into clean display scores from about 0.08 to 0.99.

    This is only for the CSV score column. Ranking is already decided before
    this function runs.
    """
    if not scored:
        return
    raw = np.array([s["final_score"] for s in scored], dtype=float)
    lo = np.percentile(raw, 5)
    hi = np.percentile(raw, 95)
    span = (hi - lo) or 1.0
    robust = np.clip((raw - lo) / span, 0.0, 1.0)        # raw-score view

    n = len(scored)
    rank_curve = np.linspace(1.0, 0.0, n) ** 1.15        # rank-position view

    blended = 0.55 * robust + 0.45 * rank_curve          # in [0,1]
    display = 0.08 + 0.91 * blended

    # The validator requires scores to never increase as rank goes down.
    for i in range(1, n):
        if display[i] > display[i - 1]:
            display[i] = display[i - 1]
    for s, d in zip(scored, display):
        s["display_score"] = round(float(d), 4)
