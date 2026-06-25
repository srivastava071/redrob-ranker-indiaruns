"""
Fake/inconsistent profile detection.

This file finds profiles with details that do not make sense.

Examples:
- expert skill with 0 months of use
- job durations that do not match start/end dates
- skill used longer than the person's whole career
- stated experience much lower than career history implies
- inverted start/end dates
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _safe_int(x: Any, default: int = 0) -> int:
    """Safely convert value to int, handling formatting inconsistencies or None."""
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except (ValueError, TypeError):
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Safely convert value to float, extracting numbers from strings if needed."""
    try:
        if x is None or x == "":
            return default
        if isinstance(x, (int, float)):
            return float(x)
        # Try regex extract
        import re
        m = re.search(r"[-+]?\d*\.\d+|\d+", str(x))
        if m:
            return float(m.group(0))
        return default
    except (ValueError, TypeError):
        return default


def _parse_year(date_str: str | None) -> float | None:
    """Convert YYYY-MM-DD or YYYY/MM/DD into a decimal year like 2024.5."""
    if not date_str:
        return None
    try:
        clean_date = str(date_str).replace("/", "-").strip()
        parts = clean_date.split("-")
        y = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 6
        return y + m / 12.0
    except (ValueError, IndexError, TypeError):
        return None


# Fixed "now" for this dataset, so results stay reproducible.
DATASET_NOW = 2026.5


def impossibility_report(cand: Dict[str, Any]) -> Tuple[int, List[str]]:
    """
    Return how many impossible details were found and why.

    More markers means the profile is more likely fake/inconsistent.
    """
    reasons: List[str] = []
    profile = cand.get("profile") or {}
    yoe = _safe_float(profile.get("years_of_experience", 0))
    history = cand.get("career_history", []) or []
    skills = cand.get("skills", []) or []

    # 1. Expert proficiency with zero months of usage.
    expert_zero = [
        s.get("name", "Unknown") for s in skills
        if s.get("proficiency") == "expert" and _safe_int(s.get("duration_months")) == 0
    ]
    if len(expert_zero) >= 1:
        reasons.append(
            f"'expert' in {len(expert_zero)} skill(s) with 0 months used "
            f"({', '.join(expert_zero[:3])})"
        )

    # 2. Total role months should roughly fit stated experience.
    total_months = sum(_safe_int(h.get("duration_months")) for h in history)
    if total_months > yoe * 12 * 1.8 + 30:
        reasons.append(
            f"career tenures sum to {total_months} months but stated "
            f"experience is only {yoe:g} years"
        )

    # 3. Role duration should roughly match its start/end dates.
    for h in history:
        start = _parse_year(h.get("start_date"))
        end = _parse_year(h.get("end_date")) if h.get("end_date") else DATASET_NOW
        if start is not None and end is not None:
            # Check for inverted dates
            if end < start:
                reasons.append(
                    f"role at {h.get('company','?')} has an end date before start date"
                )
                break
            # Check for future dates
            if start > DATASET_NOW:
                reasons.append(
                    f"role at {h.get('company','?')} lists a start date in the future"
                )
                break
            span_months = (end - start) * 12.0
            dur = _safe_int(h.get("duration_months"))
            if span_months >= 0 and abs(span_months - dur) > 30:
                reasons.append(
                    f"role at {h.get('company','?')} lists {dur} months but its "
                    f"dates span ~{span_months:.0f} months"
                )
                break

    # 4. Skill duration should not be longer than the whole career.
    long_skill = [
        s.get("name", "Unknown") for s in skills
        if _safe_int(s.get("duration_months")) > yoe * 12 + 36
    ]
    if long_skill:
        reasons.append(
            f"skill used longer than entire career ({long_skill[0]})"
        )

    # 5. Earliest role should not imply much more experience than stated.
    starts = [_parse_year(h.get("start_date")) for h in history]
    starts = [s for s in starts if s is not None]
    if starts:
        implied = DATASET_NOW - min(starts)
        if implied > yoe + 5:
            reasons.append(
                f"earliest role began ~{implied:.0f} years ago but stated "
                f"experience is {yoe:g} years"
            )

    # 6. Check for overlapping roles (concurrent full-time careers)
    has_overlap = False
    for i in range(len(history)):
        if has_overlap:
            break
        for j in range(i + 1, len(history)):
            h1, h2 = history[i], history[j]
            comp1 = (h1.get("company") or "").strip().lower()
            comp2 = (h2.get("company") or "").strip().lower()
            if not comp1 or not comp2 or comp1 == comp2:
                continue
            
            s1 = _parse_year(h1.get("start_date"))
            e1 = _parse_year(h1.get("end_date")) if h1.get("end_date") else DATASET_NOW
            s2 = _parse_year(h2.get("start_date"))
            e2 = _parse_year(h2.get("end_date")) if h2.get("end_date") else DATASET_NOW
            
            if s1 is not None and e1 is not None and s2 is not None and e2 is not None:
                overlap_start = max(s1, s2)
                overlap_end = min(e1, e2)
                if overlap_end > overlap_start:
                    overlap_months = (overlap_end - overlap_start) * 12.0
                    if overlap_months > 12.0:
                        reasons.append(
                            f"overlapping roles at {h1.get('company','?')} and {h2.get('company','?')} "
                            f"for {overlap_months:.1f} months"
                        )
                        has_overlap = True
                        break

    # 7. Check for education timeline inconsistencies
    edu = cand.get("education", []) or []
    for e in edu:
        start_y = _safe_int(e.get("start_year"))
        end_y = _safe_int(e.get("end_year"))
        deg = (e.get("degree") or "").strip().lower()
        if start_y > 0 and end_y > start_y:
            duration = end_y - start_y
            if duration > 7 and any(x in deg for x in {"b.tech", "btech", "b.e", "b.s", "bachelor"}):
                reasons.append(
                    f"bachelor degree at {e.get('institution','?')} took {duration} years"
                )
                break
                
    if starts:
        earliest_career = min(starts)
        for e in edu:
            end_y = _safe_int(e.get("end_year"))
            deg = (e.get("degree") or "").strip().lower()
            if end_y > 0 and any(x in deg for x in {"b.tech", "btech", "b.e", "b.s", "bachelor"}):
                if end_y > earliest_career + 3.0:
                    reasons.append(
                        f"completed bachelor degree in {end_y} but started career in {int(earliest_career)}"
                    )
                    break

    return len(reasons), reasons


def honeypot_penalty(n_markers: int) -> float:
    """Return the score multiplier for fake/inconsistent profiles."""
    if n_markers <= 0:
        return 1.0
    if n_markers == 1:
        return 0.12     # one impossible detail is already serious
    return 0.02         # two or more should fall near the bottom
