"""
Feature extraction.

This file converts a raw candidate profile into simple numeric signals.
Example signals: title match, real career evidence, trusted skills, experience,
location, education, company type, and coding recency.

Important idea:
We do not trust skills blindly. A candidate who only writes "RAG, LLM, FAISS"
but has no real work evidence should not beat a real AI engineer.
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import jd_spec as JD


def _lc(x: str | None) -> str:
    return (x or "").lower()


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except (ValueError, TypeError):
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        if isinstance(x, (int, float)):
            return float(x)
        import re
        m = re.search(r"[-+]?\d*\.\d+|\d+", str(x))
        if m:
            return float(m.group(0))
        return default
    except (ValueError, TypeError):
        return default


def _career_text(cand: Dict[str, Any]) -> str:
    """Join the text fields that describe what the candidate actually did."""
    p = cand.get("profile") or {}
    parts: List[str] = [_lc(p.get("headline")), _lc(p.get("summary"))]
    for h in cand.get("career_history", []) or []:
        parts.append(_lc(h.get("title")))
        parts.append(_lc(h.get("description")))
    return " \n ".join(parts)


# Each function below returns one group of recruiter-style signals.

def title_career_score(cand: Dict[str, Any]) -> Dict[str, float]:
    """
    Score how relevant the candidate's job titles are.

    Current title matters most, but past titles help too. If the whole career
    is non-technical, we strongly reduce the score to catch keyword-stuffers.
    """
    history = cand.get("career_history", []) or []
    current_title = _lc((cand.get("profile") or {}).get("current_title"))

    def relevance_of(title: str) -> float:
        # If the title is non-technical, reject it even if it has matching substrings
        if any(k in title for k in {
            "recruiter", "talent acquisition", "hr manager", "human resources",
            "sales", "marketing", "content writer", "copywriter", "designer",
            "graphic designer", "ui designer", "ux designer", "illustrator",
            "accountant", "auditor", "finance manager", "scrum master",
            "project manager", "program manager", "operations manager"
        }):
            return 0.0
        best = 0.0
        for key, val in JD.TITLE_RELEVANCE.items():
            if key in title:
                best = max(best, val)
        return best

    cur_rel = relevance_of(current_title)

    # Best relevant title from previous roles.
    past_rels = [relevance_of(_lc(h.get("title"))) for h in history]
    best_past = max(past_rels) if past_rels else 0.0

    # If most titles are non-technical, AI keywords are probably not enough.
    def is_nontech(title: str) -> bool:
        return any(k in title for k in JD.NON_TECH_TITLE_FAMILIES)

    all_titles = [current_title] + [_lc(h.get("title")) for h in history]
    nontech_frac = (
        sum(is_nontech(t) for t in all_titles) / len(all_titles)
        if all_titles else 0.0
    )

    # Blend current (0.6) and best-past (0.4).
    raw = 0.6 * cur_rel + 0.4 * best_past

    # Strongly reduce title score for mostly non-technical careers.
    if nontech_frac >= 0.99 and raw < 0.6:
        raw *= 0.05
    elif nontech_frac >= 0.6 and raw < 0.6:
        raw *= 0.35

    return {
        "title_career": min(raw, 1.0),
        "current_title_relevance": cur_rel,
        "nontech_fraction": nontech_frac,
    }


def career_evidence_score(cand: Dict[str, Any]) -> Dict[str, float]:
    """
    Score real evidence from the candidate's summary and job descriptions.

    This catches people who actually built search, ranking, recommender, RAG,
    or production ML systems, even if their skills list is not perfectly written.
    """
    text = _career_text(cand)
    score = 0.0
    hits: List[str] = []
    for phrase, w in JD.EVIDENCE_PHRASES.items():
        if phrase in text:
            score += w
            hits.append(phrase)
    # Convert raw phrase points into a 0..1 score with diminishing returns.
    norm = 1.0 - 2.71828 ** (-score / 3.0)
    return {"career_evidence": norm, "evidence_hits": hits, "evidence_raw": score}


def skills_trust_score(cand: Dict[str, Any]) -> Dict[str, float]:
    """
    Score skills, but only trust them when they have evidence.

    A skill gets more credit if it has endorsements, months of use, high
    proficiency, or an assessment score. Skills pasted with no evidence get
    very little credit.
    """
    skills = cand.get("skills", []) or []
    assessments = (cand.get("redrob_signals", {}) or {}).get(
        "skill_assessment_scores", {}
    ) or {}
    assess_lc = {_lc(k): v for k, v in assessments.items()}

    prof_w = {"beginner": 0.4, "intermediate": 0.7, "advanced": 0.9, "expert": 1.0}

    core_credit = 0.0
    n_core = 0
    n_distractor = 0
    n_nlp_ir = 0
    matched_core: List[str] = []

    for s in skills:
        name = _lc(s.get("name"))
        prof = prof_w.get(s.get("proficiency", "intermediate"), 0.6)
        endorse = _safe_int(s.get("endorsements"))
        dur = _safe_int(s.get("duration_months"))

        # Trust goes up when the skill has endorsements or real usage duration.
        trust = 1.0
        if endorse == 0 and dur == 0:
            trust = 0.15            # pure paste, no evidence
        elif endorse == 0 or dur == 0:
            trust = 0.55
        else:
            trust = min(1.0, 0.6 + endorse / 60.0 + dur / 120.0)

        # A good assessment score makes the skill claim more believable.
        a = assess_lc.get(name)
        if a is not None:
            trust *= 0.7 + 0.6 * (a / 100.0)   # 0.7..1.3

        weight = 0.0
        if name in JD.CORE_RETRIEVAL_SKILLS:
            weight = 1.0
            n_nlp_ir += 1
        elif name in JD.CORE_ML_SKILLS:
            weight = 0.6
            if name in {"nlp", "llms"}:
                n_nlp_ir += 1
        elif name in JD.NICE_TO_HAVE_SKILLS:
            weight = 0.35
        elif name in JD.MLOPS_SKILLS:
            weight = 0.12
        elif name in JD.DISTRACTOR_SKILLS:
            weight = 0.0
            n_distractor += 1

        if weight > 0:
            core_credit += weight * prof * trust
            if weight >= 0.6:
                matched_core.append(s.get("name"))
            n_core += 1

    # A few strong, trusted core skills are enough for a high score.
    norm = 1.0 - 2.71828 ** (-core_credit / 2.5)

    distractor_dominated = (n_distractor >= 2 and n_nlp_ir == 0)

    return {
        "skills_trust": norm,
        "matched_core_skills": matched_core,
        "n_core_skills": n_core,
        "n_distractor_skills": n_distractor,
        "distractor_dominated": distractor_dominated,
        "nlp_ir_present": n_nlp_ir > 0,
    }


def experience_score(cand: Dict[str, Any]) -> Dict[str, float]:
    """Score years of experience with smooth interpolation around the 6-8 yr ideal band."""
    yoe = _safe_float((cand.get("profile") or {}).get("years_of_experience", 0))
    if JD.EXP_IDEAL_LOW <= yoe <= JD.EXP_IDEAL_HIGH:
        s = 1.0
    elif JD.EXP_SOFT_LOW <= yoe < JD.EXP_IDEAL_LOW:
        # Linear interpolation between 0.85 and 1.0
        frac = (yoe - JD.EXP_SOFT_LOW) / (JD.EXP_IDEAL_LOW - JD.EXP_SOFT_LOW)
        s = 0.85 + 0.15 * frac
    elif JD.EXP_IDEAL_HIGH < yoe <= JD.EXP_SOFT_HIGH:
        # Linear interpolation between 1.0 and 0.85
        frac = (JD.EXP_SOFT_HIGH - yoe) / (JD.EXP_SOFT_HIGH - JD.EXP_IDEAL_HIGH)
        s = 0.85 + 0.15 * frac
    else:
        # Linear decay outside the soft band
        if yoe < JD.EXP_SOFT_LOW:
            dist = JD.EXP_SOFT_LOW - yoe
        else:
            dist = yoe - JD.EXP_SOFT_HIGH
        s = max(0.15, 0.85 - 0.12 * dist)
    return {"experience": s, "yoe": yoe}


def location_score(cand: Dict[str, Any]) -> Dict[str, float]:
    """Score location fit for the job."""
    p = cand.get("profile") or {}
    loc = _lc(p.get("location"))
    country = _lc(p.get("country"))
    sig = cand.get("redrob_signals", {}) or {}
    relocate = bool(sig.get("willing_to_relocate", False))

    if any(c in loc for c in JD.PREFERRED_CITIES):
        s = 1.0
    elif any(c in loc for c in JD.WELCOME_CITIES):
        s = 0.92
    elif any(c in loc for c in JD.TIER1_CITIES):
        s = 0.88 if relocate else 0.82
    elif country == "india":
        s = 0.7 if relocate else 0.55
    else:
        # Outside India is harder for this job, even with relocation.
        s = 0.35 if relocate else 0.18
    return {"location": s, "willing_to_relocate": relocate}


def education_score(cand: Dict[str, Any]) -> Dict[str, float]:
    """Give a small boost for stronger/relevant education, including degree levels."""
    edu = cand.get("education", []) or []
    if not edu:
        return {"education": 0.5}
    tier_w = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.45, "unknown": 0.55}
    best_tier = max((tier_w.get(e.get("tier", "unknown"), 0.55) for e in edu), default=0.55)
    
    relevant_fields = {"computer", "data", "machine learning", "artificial intelligence",
                       "statistics", "mathematics", "electronics", "information"}
    field_bonus = 0.0
    for e in edu:
        f = _lc(e.get("field_of_study"))
        if any(rf in f for rf in relevant_fields):
            field_bonus = 0.15
            break

    # Degree level bonus (PhD, Master's, MS, MTech)
    degree_bonus = 0.0
    for e in edu:
        d = _lc(e.get("degree"))
        if any(deg in d for deg in {"phd", "doctor", "master", "mtech", "ms", "m.s.", "m.tech."}):
            degree_bonus = 0.08
            break

    return {"education": min(1.0, 0.80 * best_tier + field_bonus + degree_bonus)}


def company_signals(cand: Dict[str, Any]) -> Dict[str, Any]:
    """Detect services-only careers and product-company experience."""
    history = cand.get("career_history", []) or []
    companies = [_lc(h.get("company")) for h in history]
    companies.append(_lc((cand.get("profile") or {}).get("current_company")))
    companies = [c for c in companies if c]

    def is_services(c: str) -> bool:
        return any(s in c for s in JD.SERVICES_COMPANIES)

    def is_product(c: str) -> bool:
        return any(s in c for s in JD.PRODUCT_COMPANY_HINTS)

    if not companies:
        return {"services_only": False, "has_product_company": False}

    services_frac = sum(is_services(c) for c in companies) / len(companies)
    has_product = any(is_product(c) for c in companies)
    return {
        "services_only": services_frac >= 0.99,
        "services_frac": services_frac,
        "has_product_company": has_product,
    }


def coding_recency_signal(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if the candidate seems to have stopped hands-on coding.
    Senior engineers are fine, but this role still wants someone who builds.
    Checks the description of the top 2 roles.
    """
    history = sorted(
        cand.get("career_history", []) or [],
        key=lambda h: h.get("start_date") or "",
        reverse=True,
    )
    if not history:
        return {"stopped_coding": False}
    # Check top 2 most recent roles to avoid false penalties on transition roles
    recent = " ".join(_lc(history[i].get("description")) for i in range(min(2, len(history))))
    non_coding = sum(1 for p in JD.NON_CODING_PHRASES if p in recent)
    build_verbs = ("built", "implemented", "wrote", "developed", "coded",
                   "designed", "engineered", "shipped", "deployed", "trained")
    has_build = any(v in recent for v in build_verbs)
    stopped = (non_coding >= 2 and not has_build)
    return {"stopped_coding": stopped}


def extract_features(cand: Dict[str, Any]) -> Dict[str, Any]:
    """Run all feature functions and return one combined dictionary."""
    f: Dict[str, Any] = {"candidate_id": cand["candidate_id"]}
    f.update(title_career_score(cand))
    f.update(career_evidence_score(cand))
    f.update(skills_trust_score(cand))
    f.update(experience_score(cand))
    f.update(location_score(cand))
    f.update(education_score(cand))
    f.update(company_signals(cand))
    f.update(coding_recency_signal(cand))
    return f


def candidate_text_for_embedding(cand: Dict[str, Any]) -> str:
    """
    Build the text used for semantic matching.

    We use headline, summary, and recent role descriptions because those best
    describe what the candidate can do.
    """
    p = cand.get("profile") or {}
    chunks = [p.get("headline") or "", p.get("summary") or ""]
    for h in (cand.get("career_history", []) or [])[:2]:
        chunks.append(f"{h.get('title','')}: {h.get('description','')}")
    return " ".join(c for c in chunks if c)[:2000]
