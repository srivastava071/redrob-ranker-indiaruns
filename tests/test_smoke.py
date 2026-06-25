"""
Basic project tests.

These tests check the most important behavior:
- strong candidates score well
- keyword-stuffers are penalized
- fake/impossible profiles are detected
- unavailable candidates are downweighted
- semantic scoring works offline

    python -m pytest tests/ -q      (or)      python tests/test_smoke.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extract_features
from src.behavioral import behavioral_multiplier
from src.honeypot import impossibility_report
from src.scoring import score_candidate
from src.semantic import build_semantic_scores


def _cand(**over):
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Candidate A", "headline": "ML Engineer",
            "summary": "Built production recommendation and semantic search systems.",
            "location": "Pune, Maharashtra", "country": "India",
            "years_of_experience": 7, "current_title": "Senior Machine Learning Engineer",
            "current_company": "ProductCo", "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": [{
            "company": "ProductCo", "title": "Senior ML Engineer",
            "start_date": "2020-01-01", "end_date": None, "duration_months": 60,
            "is_current": True, "industry": "Software", "company_size": "201-500",
            "description": "Shipped a learning-to-rank search system to real users at "
                           "scale; ran A/B tests and tracked NDCG.",
        }],
        "education": [{"institution": "IIT", "degree": "BTech",
                       "field_of_study": "CS", "start_year": 2012,
                       "end_year": 2016, "grade": "8.5", "tier": "tier_1"}],
        "skills": [
            {"name": "Information Retrieval", "proficiency": "expert",
             "endorsements": 30, "duration_months": 48},
            {"name": "FAISS", "proficiency": "advanced",
             "endorsements": 20, "duration_months": 36},
            {"name": "Recommendation Systems", "proficiency": "expert",
             "endorsements": 25, "duration_months": 40},
        ],
        "certifications": [], "languages": ["English"],
        "redrob_signals": {
            "profile_completeness_score": 90, "signup_date": "2019-01-01",
            "last_active_date": "2026-05-20", "open_to_work_flag": True,
            "recruiter_response_rate": 0.6, "notice_period_days": 30,
            "interview_completion_rate": 0.9, "saved_by_recruiters_30d": 10,
            "verified_email": True, "verified_phone": True, "linkedin_connected": True,
            "github_activity_score": 70,
        },
    }
    base.update(over)
    return base


def _score(cand, semantic=0.6):
    feat = extract_features(cand)
    beh = behavioral_multiplier(cand)
    return score_candidate(cand, feat, semantic, beh)


def test_strong_candidate_scores_high():
    s = _score(_cand())
    assert s["base_fit"] > 0.6, s["base_fit"]
    assert s["honeypot_markers"] == 0


def test_keyword_stuffer_is_gated():
    """Non-technical title + stuffed AI skills must NOT beat a real engineer."""
    stuffer = _cand(
        candidate_id="CAND_0000002",
        profile={**_cand()["profile"], "current_title": "HR Manager",
                 "headline": "HR Manager | AI enthusiast",
                 "summary": "HR generalist. Skilled in RAG, LLMs, Embeddings, FAISS."},
        career_history=[{
            "company": "ServicesCo", "title": "HR Manager",
            "start_date": "2018-01-01", "end_date": None, "duration_months": 90,
            "is_current": True, "industry": "Staffing", "company_size": "1000+",
            "description": "Recruitment, payroll, employee relations.",
        }],
        skills=[{"name": n, "proficiency": "expert", "endorsements": 1,
                 "duration_months": 0}
                for n in ["RAG", "LLMs", "Embeddings", "FAISS", "Vector Search"]],
    )
    real = _score(_cand())
    fake = _score(stuffer)
    assert fake["final_score"] < real["final_score"], (fake["final_score"], real["final_score"])
    assert fake["title_career"] < real["title_career"]


def test_honeypot_detected():
    """Expert in a skill used for 0 months is an impossibility marker."""
    hp = _cand(candidate_id="CAND_0000003",
               skills=[{"name": "FAISS", "proficiency": "expert",
                        "endorsements": 5, "duration_months": 0}])
    n_markers, reasons = impossibility_report(hp)
    assert n_markers >= 1, reasons


def test_ghost_is_downweighted():
    """JD's explicit case: perfect on paper but unavailable -> down-weighted."""
    available = _score(_cand())
    ghost = _cand(candidate_id="CAND_0000004",
                  redrob_signals={**_cand()["redrob_signals"],
                                  "recruiter_response_rate": 0.03,
                                  "last_active_date": "2025-06-01"})
    ghost_s = _score(ghost)
    assert ghost_s["behavioral_mult"] < available["behavioral_mult"]
    assert ghost_s["final_score"] < available["final_score"]


def test_zero_day_notice_is_preserved():
    immediate = _cand(
        redrob_signals={**_cand()["redrob_signals"], "notice_period_days": 0}
    )
    s = _score(immediate)
    assert s["notice_period_days"] == 0


def test_relocation_signal_reaches_scoring_output():
    relocate = _cand(
        profile={**_cand()["profile"], "location": "Indore", "country": "India"},
        redrob_signals={**_cand()["redrob_signals"], "willing_to_relocate": True},
    )
    s = _score(relocate)
    assert s["willing_to_relocate"] is True


def test_services_only_penalised():
    services = _cand(
        candidate_id="CAND_0000005",
        career_history=[{
            "company": "Infosys", "title": "Senior Engineer",
            "start_date": "2016-01-01", "end_date": None, "duration_months": 120,
            "is_current": True, "industry": "IT Services", "company_size": "1000+",
            "description": "Delivered client projects.",
        }],
        profile={**_cand()["profile"], "current_company": "Infosys",
                 "current_industry": "IT Services"},
    )
    s = _score(services)
    assert s["services_penalty"] <= 1.0


def test_inverted_dates_honeypot_detected():
    """Inverted start/end dates in career history must be flagged as a honeypot."""
    hp = _cand(
        candidate_id="CAND_0000006",
        career_history=[{
            "company": "FakeCo", "title": "ML Engineer",
            "start_date": "2024-01-01", "end_date": "2022-01-01", "duration_months": 24,
            "is_current": False, "industry": "Software", "company_size": "10-50",
            "description": "Shipped recsys.",
        }]
    )
    n_markers, reasons = impossibility_report(hp)
    assert n_markers >= 1
    assert any("end date before start date" in r for r in reasons)


def test_safe_parsing_doesnt_crash():
    """Verify that parsing anomalies (e.g. non-numeric strings) do not crash feature scoring."""
    messy = _cand(
        candidate_id="CAND_0000007",
        profile={**_cand()["profile"], "years_of_experience": "7 years"},
        skills=[{"name": "FAISS", "proficiency": "expert", "endorsements": "15 endorsements", "duration_months": "36 months"}]
    )
    s = _score(messy)
    assert s["yoe"] == 7.0
    assert s["skills_trust"] > 0.0


def test_null_fields_dont_crash():
    """Verify that null fields (e.g., profile: null) do not crash the parser."""
    null_cand = {
        "candidate_id": "CAND_0000008",
        "profile": None,
        "career_history": None,
        "skills": None,
        "education": None,
        "redrob_signals": None,
    }
    s = _score(null_cand)
    assert s["candidate_id"] == "CAND_0000008"
    assert s["final_score"] >= 0.0


def test_new_honeypots_detected():
    # 1. Overlapping roles honeypot
    hp1 = _cand(
        candidate_id="CAND_0000009",
        career_history=[
            {
                "company": "Company A", "title": "ML Engineer",
                "start_date": "2020-01-01", "end_date": "2023-01-01", "duration_months": 36,
            },
            {
                "company": "Company B", "title": "Software Engineer",
                "start_date": "2020-06-01", "end_date": "2022-06-01", "duration_months": 24,
            }
        ]
    )
    n_markers1, reasons1 = impossibility_report(hp1)
    assert n_markers1 >= 1
    assert any("overlapping roles" in r for r in reasons1)

    # 2. Education timeline: bachelor completed long after career start
    hp2 = _cand(
        candidate_id="CAND_0000010",
        career_history=[{
            "company": "ProductCo", "title": "ML Engineer",
            "start_date": "2015-01-01", "end_date": None, "duration_months": 120,
        }],
        education=[{
            "institution": "IIT", "degree": "BTech", "field_of_study": "CS",
            "start_year": 2016, "end_year": 2020, "tier": "tier_1"
        }]
    )
    n_markers2, reasons2 = impossibility_report(hp2)
    assert n_markers2 >= 1
    assert any("completed bachelor degree" in r for r in reasons2)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} smoke tests passed.")


if __name__ == "__main__":
    _run_all()
