"""
Reason text generation.

This file creates the short explanation shown in the CSV.

No LLM is used. Every sentence is built from real candidate fields and score
details, so the reason should not hallucinate.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _title_phrase(cand: Dict[str, Any]) -> str:
    """Short phrase for current title and experience."""
    p = cand.get("profile") or {}
    return f"{p.get('current_title','professional')} with {float(p.get('years_of_experience',0) or 0):.1f} yrs"


def _core_skill_phrase(sc: Dict[str, Any]) -> str:
    """Mention up to three strong matched skills."""
    sk: List[str] = [s for s in (sc.get("matched_core_skills") or []) if s]
    if not sk:
        return ""
    head = ", ".join(sk[:3])
    return f"core stack incl. {head}"


def _evidence_phrase(sc: Dict[str, Any]) -> str:
    """Mention the best career evidence found in the profile text."""
    hits = sc.get("evidence_hits") or []
    # Prefer evidence that best matches this job.
    priority = [
        "recommendation system", "recommender", "ranking system", "learning to rank",
        "search relevance", "semantic search", "vector search", "information retrieval",
        "hybrid search", "retrieval-augmented", "rag", "ndcg", "a/b test",
        "embeddings", "personalization", "real users", "at scale", "production",
    ]
    picked = [h for h in priority if h in hits][:2]
    if not picked:
        return ""
    nice = {
        "rag": "RAG", "ndcg": "NDCG-based evaluation", "a/b test": "A/B testing",
        "learning to rank": "learning-to-rank", "retrieval-augmented": "retrieval-augmented generation",
    }
    picked = [nice.get(p, p) for p in picked]
    return "career shows " + " & ".join(picked)


def _location_phrase(cand: Dict[str, Any], sc: Dict[str, Any]) -> str:
    """Explain whether the location is good for this job."""
    p = cand.get("profile") or {}
    loc = p.get("location", "")
    country = (p.get("country") or "").lower()
    if sc.get("location", 0) >= 0.92:
        return f"based in {loc} (target region)"
    if country == "india":
        return f"India-based ({loc})" + ("; open to relocate" if sc.get("willing_to_relocate") else "")
    if sc.get("willing_to_relocate"):
        return f"{loc}; willing to relocate"
    return f"{loc} (outside India)"


def _behavior_phrase(sc: Dict[str, Any]) -> str:
    """Summarize availability signals."""
    rr = sc.get("response_rate", 0)
    di = sc.get("days_inactive", 9999)
    bits = [f"recruiter response {rr:.0%}"]
    if di <= 30:
        bits.append("active this month")
    elif di >= 180:
        bits.append("inactive 6+ months")
    if sc.get("notice_period_days", 90) <= 30:
        bits.append("30-day or shorter notice")
    return ", ".join(bits)


def _concerns(sc: Dict[str, Any]) -> List[str]:
    """List any concerns worth mentioning in the explanation."""
    c: List[str] = []
    if sc.get("honeypot_markers", 0) > 0:
        c.append("profile has internal inconsistencies")
    if sc.get("services_only"):
        c.append("career entirely at services firms")
    if sc.get("distractor_dominated"):
        c.append("AI background is CV/speech, not NLP/IR")
    if sc.get("yoe", 6) < 5:
        c.append(f"below the 5-yr band ({sc.get('yoe',0):.1f} yrs)")
    elif sc.get("yoe", 6) > 9:
        c.append(f"above the 9-yr band ({sc.get('yoe',0):.1f} yrs)")
    if sc.get("response_rate", 1) < 0.2:
        c.append("low recruiter responsiveness")
    if sc.get("days_inactive", 0) >= 180:
        c.append("not recently active")
    if not sc.get("nlp_ir_present", True) and sc.get("n_core_skills", 0) == 0:
        c.append("no clear NLP/IR signal")
    return c


def build_reasoning(cand: Dict[str, Any], sc: Dict[str, Any], rank: int) -> str:
    """Build the final explanation sentence for one ranked candidate."""
    parts: List[str] = [_title_phrase(cand)]

    strong = [p for p in (_evidence_phrase(sc), _core_skill_phrase(sc)) if p]
    if strong:
        parts.append("; ".join(strong))

    parts.append(_location_phrase(cand, sc))
    parts.append(_behavior_phrase(sc))

    concerns = _concerns(sc)
    # Higher ranks should sound stronger; lower ranks can mention more concerns.
    if rank <= 15:
        tail = f" Minor concern: {concerns[0]}." if concerns else ""
        sentence = "; ".join(parts) + "." + tail
    elif rank <= 50:
        tail = f" Concerns: {', '.join(concerns[:2])}." if concerns else ""
        sentence = "; ".join(parts) + "." + tail
    else:
        # Ranks 51-100 are still selected, but not as strong as the top tier.
        lead = "Adjacent fit" if concerns else "Solid fit, just below the top tier"
        facts = [p for p in (_evidence_phrase(sc), _core_skill_phrase(sc)) if p]
        fact_str = ("; " + "; ".join(facts)) if facts else ""
        tail = f" - {', '.join(concerns[:2])}" if concerns else ""
        sentence = (f"{lead}: {_title_phrase(cand)}{fact_str}; "
                    f"{_location_phrase(cand, sc)}{tail}.")

    # Keep the CSV reason short and readable.
    return sentence.strip().replace("  ", " ")[:300]
