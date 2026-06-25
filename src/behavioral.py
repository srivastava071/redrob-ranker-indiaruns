"""
Availability and engagement scoring.

This file answers a simple recruiter question:
"Even if this person is good, are they reachable and likely to respond?"

Good availability gives a small boost. Very poor activity/response signals
trigger a stronger "ghost" penalty.
"""

from __future__ import annotations

from typing import Any, Dict


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
        import re
        m = re.search(r"[-+]?\d*\.\d+|\d+", str(x))
        if m:
            return float(m.group(0))
        return default
    except (ValueError, TypeError):
        return default


def _days_since(date_str: str | None, now=(2026, 6, 1)) -> float:
    """Return how many days have passed since a YYYY-MM-DD or YYYY/MM/DD date."""
    if not date_str:
        return 9999.0
    try:
        clean_date = str(date_str).replace("/", "-").strip()
        y, m, d = (int(x) for x in clean_date.split("-")[:3])
        from datetime import date
        return (date(*now) - date(y, m, d)).days
    except Exception:  # noqa: BLE001
        return 9999.0


def behavioral_multiplier(cand: Dict[str, Any]) -> Dict[str, Any]:
    s = cand.get("redrob_signals", {}) or {}

    response_rate = _safe_float(s.get("recruiter_response_rate", 0.0))
    interview_rate = _safe_float(s.get("interview_completion_rate", 0.0))
    completeness = _safe_float(s.get("profile_completeness_score", 0.0)) / 100.0
    open_flag = bool(s.get("open_to_work_flag", False))
    saved = _safe_int(s.get("saved_by_recruiters_30d", 0))
    notice = _safe_int(s.get("notice_period_days", 90), default=90)
    days_inactive = _days_since(s.get("last_active_date"))
    verified = sum(bool(s.get(k, False))
                   for k in ("verified_email", "verified_phone", "linkedin_connected"))

    # Start at 1.0 and add small boosts/penalties around it.
    adj = 0.0

    # Recruiter response rate is the strongest availability signal.
    if response_rate >= 0.5:
        adj += 0.035
    elif response_rate >= 0.30:
        adj += 0.0
    elif response_rate >= 0.15:
        adj -= 0.035
    else:
        adj -= 0.06          # the ghost gate below handles the worst cases

    # Recently active candidates are more reachable.
    if days_inactive <= 30:
        adj += 0.025
    elif days_inactive <= 90:
        adj += 0.0
    elif days_inactive <= 180:
        adj -= 0.035
    else:
        adj -= 0.06

    # "Open to work" helps, but not having it is not a hard penalty.
    if open_flag:
        adj += 0.02

    # Shorter notice period is better for hiring speed.
    if notice <= 30:
        adj += 0.02
    elif notice <= 60:
        adj += 0.01
    elif notice <= 90:
        adj += 0.0
    else:
        adj -= 0.03

    # Small positive signals for profile quality and reliability.
    if interview_rate >= 0.8:
        adj += 0.015
    elif interview_rate >= 0.5:
        adj += 0.005
    if completeness >= 0.85:
        adj += 0.01
    if verified >= 3:
        adj += 0.01
    elif verified >= 2:
        adj += 0.005

    # If other recruiters saved them, that is a mild demand signal.
    if saved >= 8:
        adj += 0.02
    elif saved >= 3:
        adj += 0.01

    mult = 1.0 + adj
    # Keep normal availability changes gentle.
    mult = max(0.93, min(1.05, mult))

    # Stronger penalty for candidates who are probably not actually available.
    ghost = False
    if response_rate < 0.12 and days_inactive > 150:
        mult *= 0.50
        ghost = True
    elif response_rate < 0.08:
        mult *= 0.62
        ghost = True
    elif response_rate < 0.15 and days_inactive > 240:
        mult *= 0.70
        ghost = True

    mult = max(0.38, min(1.05, mult))

    return {
        "behavioral_mult": mult,
        "response_rate": response_rate,
        "days_inactive": days_inactive,
        "open_to_work": open_flag,
        "notice_period_days": notice,
        "interview_completion_rate": interview_rate,
        "saved_by_recruiters_30d": saved,
        "availability_flag": "ghost" if ghost else "available",
    }
