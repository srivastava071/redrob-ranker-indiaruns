#!/usr/bin/env python3
"""
Streamlit candidate ranker demo portal — Premium Edition.

This dashboard provides an interactive UI for ranking and auditing candidates for
the Senior AI Engineer job. It uses custom CSS, HTML rendering, and dynamic filtering.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import json
import time
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st

from src.features import extract_features, candidate_text_for_embedding
from src.behavioral import behavioral_multiplier
from src.semantic import build_semantic_scores
from src.scoring import score_candidate, normalise_scores
from src.reasoning import build_reasoning
from src import jd_spec as JD

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob · AI Candidate Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Init ───────────────────────────────────────────────────────
for key, default in [
    ("ranked", False),
    ("ranked_results", []),
    ("honeypots", []),
    ("by_id", {}),
    ("stats", {}),
    ("all_cands", []),
    ("light_mode", False),
    ("duration", 0.0),
    ("page", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.session_state["light_mode"] = st.toggle("☀️ Light Mode", value=st.session_state["light_mode"])
    st.markdown("---")
    st.markdown("### 📂 Data Source")
    uploaded = st.file_uploader("Upload candidates.jsonl", type=["jsonl", "json", "txt"])
    use_sample = st.checkbox("Use bundled sample", value=not bool(uploaded))
    st.markdown("")
    rank_clicked = st.button("🚀 Process & Rank Candidates", use_container_width=True)

light = st.session_state["light_mode"]

# ── Mega CSS ─────────────────────────────────────────────────────────────────
def _css() -> str:
    # Theme tokens
    if light:
        bg          = "#f0f2f5"
        bg2         = "#ffffff"
        sidebar_bg  = "#ffffff"
        text        = "#1e293b"
        text2       = "#475569"
        card_bg     = "rgba(255,255,255,0.82)"
        card_border = "rgba(0,0,0,0.06)"
        glass       = "rgba(255,255,255,0.6)"
        glow        = "rgba(56,189,248,0.08)"
        divider     = "rgba(0,0,0,0.06)"
        particle_c  = "rgba(56,189,248,0.06)"
        input_bg    = "#f8fafc"
        tab_bg      = "rgba(0,0,0,0.02)"
        tab_active  = "rgba(56,189,248,0.08)"
        reason_bg   = "rgba(139,92,246,0.04)"
        shimmer_c   = "rgba(0,0,0,0.04)"
    else:
        bg          = "#060a13"
        bg2         = "#0c1220"
        sidebar_bg  = "#0a0f1c"
        text        = "#e2e8f0"
        text2       = "#94a3b8"
        card_bg     = "rgba(15,23,42,0.55)"
        card_border = "rgba(255,255,255,0.06)"
        glass       = "rgba(15,23,42,0.4)"
        glow        = "rgba(56,189,248,0.06)"
        divider     = "rgba(255,255,255,0.06)"
        particle_c  = "rgba(56,189,248,0.04)"
        input_bg    = "rgba(15,23,42,0.6)"
        tab_bg      = "rgba(15,23,42,0.4)"
        tab_active  = "rgba(56,189,248,0.1)"
        reason_bg   = "rgba(139,92,246,0.08)"
        shimmer_c   = "rgba(255,255,255,0.03)"

    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ─── RESET & GLOBALS ─── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [data-testid="stAppViewContainer"], .stApp {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: {bg} !important;
    color: {text} !important;
}}
[data-testid="stSidebar"] {{
    background: {sidebar_bg} !important;
    border-right: 1px solid {card_border};
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: {text2} !important;
}}

/* ─── ANIMATED MESH BG ─── */
.stApp::before {{
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 600px 400px at 15% 20%, rgba(56,189,248,0.07) 0%, transparent 70%),
        radial-gradient(ellipse 500px 500px at 80% 70%, rgba(139,92,246,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 400px 300px at 50% 50%, rgba(168,85,247,0.04) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: meshDrift 20s ease-in-out infinite alternate;
}}
@keyframes meshDrift {{
    0%   {{ opacity: 1; }}
    50%  {{ opacity: 0.7; }}
    100% {{ opacity: 1; }}
}}

/* ─── ANIMATIONS ─── */
@keyframes fadeSlideUp {{
    from {{ opacity: 0; transform: translateY(24px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50%      {{ opacity: 0.6; }}
}}
@keyframes shimmer {{
    0%   {{ background-position: -200% 0; }}
    100% {{ background-position: 200% 0; }}
}}
@keyframes countUp {{
    from {{ opacity: 0; transform: scale(0.6); }}
    to   {{ opacity: 1; transform: scale(1); }}
}}
@keyframes borderGlow {{
    0%, 100% {{ border-color: rgba(56,189,248,0.15); }}
    50%      {{ border-color: rgba(139,92,246,0.25); }}
}}
@keyframes floatBadge {{
    0%, 100% {{ transform: translateY(0px); }}
    50%      {{ transform: translateY(-3px); }}
}}

/* ─── HERO BANNER ─── */
.hero {{
    position: relative;
    background: {glass};
    backdrop-filter: blur(24px) saturate(1.4);
    -webkit-backdrop-filter: blur(24px) saturate(1.4);
    border: 1px solid {card_border};
    border-radius: 20px;
    padding: 40px 44px;
    margin-bottom: 32px;
    overflow: hidden;
    animation: fadeSlideUp 0.6s ease-out;
}}
.hero::before {{
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 340px; height: 340px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
    transform: translate(30%, -40%);
    pointer-events: none;
}}
.hero::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(139,92,246,0.10) 0%, transparent 70%);
    border-radius: 50%;
    transform: translate(-30%, 50%);
    pointer-events: none;
}}
.hero-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(139,92,246,0.15));
    color: #38bdf8;
    border: 1px solid rgba(56,189,248,0.2);
    margin-bottom: 16px;
    animation: floatBadge 3s ease-in-out infinite;
}}
.hero-title {{
    font-weight: 900;
    font-size: 2.8rem;
    line-height: 1.1;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 40%, #c084fc 70%, #f472b6 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 6s linear infinite;
    margin-bottom: 10px;
    position: relative;
    z-index: 1;
}}
.hero-sub {{
    font-size: 1rem;
    color: {text2};
    font-weight: 400;
    max-width: 620px;
    line-height: 1.6;
    position: relative;
    z-index: 1;
}}
.hero-sub strong {{ color: {text}; }}

/* ─── KPI CARDS ─── */
.kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
}}
.kpi {{
    background: {card_bg};
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid {card_border};
    border-radius: 16px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    transition: all 0.35s cubic-bezier(.4,0,.2,1);
    animation: fadeSlideUp 0.5s ease-out backwards;
}}
.kpi:hover {{
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    border-color: rgba(56,189,248,0.3);
}}
.kpi::after {{
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, {glow}, transparent 60%);
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.35s;
}}
.kpi:hover::after {{ opacity: 1; }}
.kpi-icon {{
    font-size: 1.6rem;
    margin-bottom: 10px;
    display: block;
}}
.kpi-label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {text2};
    font-weight: 600;
    margin-bottom: 6px;
}}
.kpi-val {{
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
    animation: countUp 0.6s cubic-bezier(.2,.8,.4,1) backwards;
}}
.kpi-sub {{
    font-size: 0.72rem;
    color: {text2};
    margin-top: 8px;
    font-weight: 400;
}}
.kpi-val.v-blue   {{ color: #38bdf8; }}
.kpi-val.v-purple {{ color: #a78bfa; }}
.kpi-val.v-red    {{ color: #f87171; }}
.kpi-val.v-green  {{ color: #34d399; }}
.kpi-val.v-amber  {{ color: #fbbf24; }}
.kpi:nth-child(1) {{ animation-delay: 0.05s; }}
.kpi:nth-child(2) {{ animation-delay: 0.12s; }}
.kpi:nth-child(3) {{ animation-delay: 0.19s; }}
.kpi:nth-child(4) {{ animation-delay: 0.26s; }}
.kpi:nth-child(5) {{ animation-delay: 0.33s; }}

/* ─── CANDIDATE CARD ─── */
.ccard {{
    background: {card_bg};
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid {card_border};
    border-radius: 18px;
    padding: 28px 30px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
    transition: all 0.35s cubic-bezier(.4,0,.2,1);
    animation: fadeSlideUp 0.5s ease-out backwards;
}}
.ccard:hover {{
    transform: translateY(-3px);
    box-shadow: 0 24px 48px rgba(0,0,0,0.18);
    border-color: rgba(139,92,246,0.3);
}}
.ccard::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    opacity: 0;
    transition: opacity 0.3s;
}}
.ccard:hover::before {{ opacity: 1; }}

.ccard-head {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 18px;
    flex-wrap: wrap;
    gap: 12px;
}}
.ccard-left {{ display: flex; align-items: center; gap: 14px; }}
.rank-chip {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 42px; height: 42px;
    border-radius: 12px;
    font-weight: 800;
    font-size: 0.95rem;
    color: #fff;
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    box-shadow: 0 4px 14px rgba(99,102,241,0.3);
    flex-shrink: 0;
}}
.rank-chip.top3 {{
    background: linear-gradient(135deg, #f59e0b, #ef4444);
    box-shadow: 0 4px 14px rgba(245,158,11,0.35);
    animation: floatBadge 2.5s ease-in-out infinite;
}}
.ccard-name {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {text};
    letter-spacing: -0.01em;
}}
.ccard-role {{
    font-size: 0.82rem;
    color: {text2};
    margin-top: 2px;
}}

/* Score gauge */
.score-ring {{
    position: relative;
    width: 64px; height: 64px;
    flex-shrink: 0;
}}
.score-ring svg {{ transform: rotate(-90deg); }}
.score-ring-bg {{
    fill: none;
    stroke: {divider};
    stroke-width: 5;
}}
.score-ring-fill {{
    fill: none;
    stroke-width: 5;
    stroke-linecap: round;
    transition: stroke-dashoffset 1s cubic-bezier(.4,0,.2,1);
}}
.score-ring-text {{
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.78rem;
    font-weight: 800;
    color: {text};
}}

/* Reason block */
.ccard-reason {{
    background: {reason_bg};
    border-left: 3px solid #8b5cf6;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: {text};
    line-height: 1.65;
    margin-bottom: 18px;
}}
.ccard-reason strong {{ color: #a78bfa; }}

/* Pills */
.pills {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
.p {{
    font-size: 0.68rem;
    padding: 4px 11px;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 0.01em;
    transition: transform 0.2s, box-shadow 0.2s;
}}
.p:hover {{ transform: scale(1.05); }}
.p-core  {{ background: rgba(56,189,248,0.10); color: #38bdf8; border: 1px solid rgba(56,189,248,0.25); }}
.p-ev    {{ background: rgba(139,92,246,0.10); color: #a78bfa; border: 1px solid rgba(139,92,246,0.25); }}
.p-good  {{ background: rgba(16,185,129,0.10); color: #34d399; border: 1px solid rgba(16,185,129,0.25); }}
.p-bad   {{ background: rgba(239,68,68,0.10);  color: #f87171; border: 1px solid rgba(239,68,68,0.25);  }}
.p-mute  {{ background: {shimmer_c}; color: {text2}; border: 1px solid {card_border}; }}

/* Detail grid */
.dgrid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    margin-top: 20px;
    padding-top: 18px;
    border-top: 1px solid {divider};
}}
@media (max-width: 768px) {{ .dgrid {{ grid-template-columns: 1fr; }} }}

.dbox {{
    background: {glass};
    border-radius: 14px;
    padding: 18px 20px;
    border: 1px solid {card_border};
}}
.dbox-title {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    color: #38bdf8;
    margin-bottom: 14px;
}}

/* Progress bars */
.pbar {{ margin-bottom: 12px; }}
.pbar:last-child {{ margin-bottom: 0; }}
.pbar-head {{
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: {text2};
    margin-bottom: 5px;
    font-weight: 500;
}}
.pbar-track {{
    background: {divider};
    border-radius: 8px;
    height: 7px;
    overflow: hidden;
    position: relative;
}}
.pbar-fill {{
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    background-size: 200% auto;
    animation: shimmer 3s linear infinite;
    transition: width 1s cubic-bezier(.4,0,.2,1);
}}

/* Info rows */
.irow {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    font-size: 0.82rem;
    border-bottom: 1px solid {divider};
}}
.irow:last-child {{ border-bottom: none; }}
.irow-k {{ color: {text2}; font-weight: 500; }}
.irow-v {{ color: {text}; font-weight: 600; }}

/* Section labels */
.sec-label {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {text2};
    margin: 14px 0 8px 0;
}}

/* ─── TABS ─── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: {tab_bg};
    border-radius: 12px;
    padding: 4px;
    border: 1px solid {card_border};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: {text2} !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.2s !important;
}}
.stTabs [aria-selected="true"] {{
    background: {tab_active} !important;
    color: #38bdf8 !important;
    box-shadow: 0 2px 8px rgba(56,189,248,0.1) !important;
}}

/* ─── BUTTONS ─── */
.stButton>button {{
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #8b5cf6 100%) !important;
    background-size: 200% auto !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.3s cubic-bezier(.4,0,.2,1) !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.3) !important;
}}
.stButton>button:hover {{
    background-position: right center !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.4) !important;
}}
.stButton>button:active {{
    transform: translateY(0px) !important;
}}

/* ─── HONEYPOT CARD ─── */
.hp-card {{
    background: {card_bg};
    backdrop-filter: blur(16px);
    border: 1px solid rgba(239,68,68,0.2);
    border-left: 4px solid #ef4444;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 18px;
    animation: fadeSlideUp 0.5s ease-out backwards;
    transition: all 0.3s;
}}
.hp-card:hover {{
    box-shadow: 0 12px 32px rgba(239,68,68,0.1);
    border-color: rgba(239,68,68,0.35);
}}

/* ─── ANALYTICS CARD ─── */
.acard {{
    background: {card_bg};
    backdrop-filter: blur(16px);
    border: 1px solid {card_border};
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    animation: fadeSlideUp 0.5s ease-out backwards;
}}
.acard-title {{
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid {divider};
    display: flex;
    align-items: center;
    gap: 8px;
}}
.alist {{
    list-style: none;
    padding: 0;
    margin: 0;
}}
.alist li {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    margin-bottom: 6px;
    border-radius: 10px;
    background: {glass};
    border: 1px solid {card_border};
    font-size: 0.84rem;
    transition: all 0.2s;
}}
.alist li:hover {{
    transform: translateX(4px);
    border-color: rgba(56,189,248,0.2);
}}
.alist-k {{ color: {text}; font-weight: 500; }}
.alist-v {{ color: #38bdf8; font-weight: 700; }}

/* ─── LANDING ─── */
.landing {{
    text-align: center;
    padding: 80px 30px;
    background: {glass};
    backdrop-filter: blur(20px);
    border: 1px solid {card_border};
    border-radius: 24px;
    animation: fadeSlideUp 0.7s ease-out;
    position: relative;
    overflow: hidden;
}}
.landing::before {{
    content: '';
    position: absolute;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.08), transparent 70%);
    top: -100px; left: 50%;
    transform: translateX(-50%);
    pointer-events: none;
}}
.landing-icon {{
    font-size: 4rem;
    margin-bottom: 20px;
    animation: floatBadge 3s ease-in-out infinite;
    display: inline-block;
}}
.landing-title {{
    font-size: 2rem;
    font-weight: 800;
    color: {text};
    margin-bottom: 12px;
    letter-spacing: -0.02em;
}}
.landing-desc {{
    font-size: 1rem;
    color: {text2};
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.7;
}}

/* ─── SKELETON ─── */
.skel {{
    background: linear-gradient(90deg, {shimmer_c} 25%, {glow} 50%, {shimmer_c} 75%);
    background-size: 200% 100%;
    animation: shimmer 1.8s ease-in-out infinite;
    border-radius: 16px;
    height: 180px;
    margin-bottom: 18px;
    border: 1px solid {card_border};
}}

/* ─── MISC ─── */
.stTabs [data-baseweb="tab-highlight"] {{ display: none !important; }}
.stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
div[data-testid="stExpander"] details {{
    border: 1px solid {card_border} !important;
    border-radius: 14px !important;
    background: {card_bg} !important;
}}
</style>"""

st.markdown(_css(), unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _c(html: str) -> str:
    """Strip leading whitespace from each line to prevent Streamlit code-block rendering."""
    return "\n".join(line.strip() for line in html.splitlines())


def _score_ring(score: float, size: int = 64) -> str:
    """SVG radial gauge for a candidate's display score."""
    r = (size - 10) / 2
    circ = 2 * math.pi * r
    pct = min(score, 1.0)
    offset = circ * (1 - pct)
    # colour gradient: low=red mid=amber high=green
    if pct >= 0.85:
        color = "#34d399"
    elif pct >= 0.65:
        color = "#38bdf8"
    else:
        color = "#fbbf24"
    return f"""
    <div class="score-ring" style="width:{size}px;height:{size}px;">
        <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
            <circle class="score-ring-bg" cx="{size//2}" cy="{size//2}" r="{r}"/>
            <circle class="score-ring-fill" cx="{size//2}" cy="{size//2}" r="{r}"
                    stroke="{color}" stroke-dasharray="{circ}" stroke-dashoffset="{offset}"/>
        </svg>
        <div class="score-ring-text">{score:.3f}</div>
    </div>"""


# ── Data loaders ─────────────────────────────────────────────────────────────
def load_jsonl(text: str) -> List[Dict[str, Any]]:
    """Convert uploaded JSONL or JSON text into a list of candidate dictionaries."""
    text_stripped = text.strip()
    if text_stripped.startswith("[") and text_stripped.endswith("]"):
        try:
            out = json.loads(text_stripped)
            if isinstance(out, list):
                return out
            return [out]
        except json.JSONDecodeError:
            pass

    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, list):
                out.extend(parsed)
            else:
                out.append(parsed)
        except json.JSONDecodeError:
            pass
    return out


def rank_candidates_pipeline(cands: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run full scoring on all candidates, return (top_100_normalised, honeypots)."""
    feats = [extract_features(c) for c in cands]
    beh = [behavioral_multiplier(c) for c in cands]
    texts = [candidate_text_for_embedding(c) for c in cands]
    sem = build_semantic_scores(texts, precomputed_path=None, prefer_embeddings=False)
    scored = [
        score_candidate(c, f, float(s), b)
        for c, f, b, s in zip(cands, feats, beh, sem)
    ]
    scored.sort(key=lambda s: (-s["final_score"], s["candidate_id"]))
    clean = [s for s in scored if s["honeypot_markers"] == 0]
    honeypots = [s for s in scored if s["honeypot_markers"] > 0]
    top_100 = clean[:100]
    normalise_scores(top_100)
    top_100.sort(key=lambda s: (-s["display_score"], s["candidate_id"]))
    by_id = {c["candidate_id"]: c for c in cands}
    for rnk, s in enumerate(top_100, 1):
        s["_rank"] = rnk
        s["_reasoning"] = build_reasoning(by_id[s["candidate_id"]], s, rnk)
    return top_100, honeypots


def compute_dataset_stats(cands, clean_top, honeypots):
    """Calculate interesting analytics from the dataset for the dashboard."""
    stats = {}
    n = len(cands)
    if n == 0:
        return stats
    yoes = [float((c.get("profile") or {}).get("years_of_experience", 0) or 0) for c in cands]
    stats["avg_yoe"] = sum(yoes) / n
    stats["max_yoe"] = max(yoes)
    locations = {}
    for c in cands:
        loc = (c.get("profile") or {}).get("location") or "Unknown"
        locations[loc] = locations.get(loc, 0) + 1
    stats["locations"] = sorted(locations.items(), key=lambda x: x[1], reverse=True)[:5]
    skills_count = {}
    for c in cands:
        for s in c.get("skills", []) or []:
            name = s.get("name")
            if name:
                skills_count[name.strip()] = skills_count.get(name.strip(), 0) + 1
    stats["top_skills"] = sorted(skills_count.items(), key=lambda x: x[1], reverse=True)[:6]
    n_top = len(clean_top)
    if n_top > 0:
        stats["product_pct"] = (sum(1 for s in clean_top if s.get("has_product_company")) / n_top) * 100
        stats["services_pct"] = (sum(1 for s in clean_top if s.get("services_only")) / n_top) * 100
        stats["stopped_coding_pct"] = (sum(1 for s in clean_top if s.get("stopped_coding")) / n_top) * 100
    else:
        stats["product_pct"] = stats["services_pct"] = stats["stopped_coding_pct"] = 0
    return stats


# ── Card renderers ───────────────────────────────────────────────────────────
def make_card_html(s: Dict[str, Any], by_id: Dict[str, Any]) -> str:
    """Compile custom candidate card HTML with detail sections and progress bars."""
    cand = by_id[s["candidate_id"]]
    p = cand.get("profile") or {}
    rank = s["_rank"]

    # Skills badges
    matched = s.get("matched_core_skills", [])
    sk_html = "".join(f'<span class="p p-core">{sk}</span>' for sk in matched) or '<span class="p p-mute">No core stack matched</span>'

    # Evidence
    ev = s.get("evidence_hits", [])[:5]
    ev_html = "".join(f'<span class="p p-ev">{e}</span>' for e in ev) or '<span class="p p-mute">No evidence phrases</span>'

    # Signal pills
    notice = s.get("notice_period_days", 90)
    sigs = []
    if s.get("willing_to_relocate"):
        sigs.append(("Open to Relocate", "p-good"))
    if notice <= 30:
        sigs.append((f"{notice}d Notice", "p-good"))
    else:
        sigs.append((f"{notice}d Notice", "p-mute"))
    if s.get("has_product_company"):
        sigs.append(("Product Co.", "p-good"))
    if s.get("services_only"):
        sigs.append(("Services Only", "p-bad"))
    if s.get("stopped_coding"):
        sigs.append(("Stopped Coding", "p-bad"))
    if s.get("distractor_dominated"):
        sigs.append(("Distractor", "p-bad"))
    sig_html = "".join(f'<span class="p {cls}">{label}</span>' for label, cls in sigs)

    # Progress bars
    bars = [
        ("Title & Career Fit", s.get("title_career", 0), "26%"),
        ("Career Evidence", s.get("career_evidence", 0), "22%"),
        ("Skills Trust", s.get("skills_trust", 0), "18%"),
        ("Semantic Match", s.get("semantic", 0), "14%"),
        ("Experience Fit", s.get("experience", 0), "10%"),
        ("Location Fit", s.get("location", 0), "6%"),
        ("Education Fit", s.get("education", 0), "4%"),
    ]
    bars_html = ""
    for label, val, weight in bars:
        pct = int(min(val, 1.0) * 100)
        bars_html += f"""<div class="pbar">
            <div class="pbar-head"><span>{label} <span style="opacity:0.5">({weight})</span></span><span>{pct}%</span></div>
            <div class="pbar-track"><div class="pbar-fill" style="width:{pct}%"></div></div>
        </div>"""

    # Score ring
    ring = _score_ring(s["display_score"])

    chip_cls = "rank-chip top3" if rank <= 3 else "rank-chip"

    return f"""
    <div class="ccard" style="animation-delay:{min(rank * 0.04, 0.8):.2f}s">
        <div class="ccard-head">
            <div class="ccard-left">
                <div class="{chip_cls}">{rank}</div>
                <div>
                    <div class="ccard-name">{s['candidate_id']}</div>
                    <div class="ccard-role">{p.get('current_title','Professional')} @ {p.get('current_company','N/A')} · {p.get('years_of_experience',0)} yrs</div>
                </div>
            </div>
            {ring}
        </div>

        <div class="ccard-reason"><strong>AI Assessment:</strong> {s['_reasoning']}</div>

        <div class="sec-label">🛠️ Core Stack Matches</div>
        <div class="pills">{sk_html}</div>

        <div class="sec-label">📝 Evidence Phrases</div>
        <div class="pills">{ev_html}</div>

        <div class="sec-label">🚦 Signals & Flags</div>
        <div class="pills">{sig_html}</div>

        <div class="dgrid">
            <div class="dbox">
                <div class="dbox-title">Fit Score Breakdown</div>
                {bars_html}
            </div>
            <div class="dbox">
                <div class="dbox-title">Recruitment Signals</div>
                <div class="irow"><span class="irow-k">Location</span><span class="irow-v">{p.get('location','N/A')}, {p.get('country','')}</span></div>
                <div class="irow"><span class="irow-k">Notice Period</span><span class="irow-v">{notice} days</span></div>
                <div class="irow"><span class="irow-k">Response Rate</span><span class="irow-v">{s.get('response_rate',0):.0%}</span></div>
                <div class="irow"><span class="irow-k">Last Active</span><span class="irow-v">{int(s.get('days_inactive',9999))}d ago</span></div>
                <div class="irow"><span class="irow-k">Interview Completion</span><span class="irow-v">{s.get('interview_completion_rate',0):.0%}</span></div>
                <div class="irow"><span class="irow-k">Saved by Recruiters (30d)</span><span class="irow-v">{s.get('saved_by_recruiters_30d',0)}</span></div>
                <div class="irow"><span class="irow-k">Behavioral Multiplier</span><span class="irow-v" style="color:#38bdf8">{s.get('behavioral_mult',1.0):.2f}×</span></div>
            </div>
        </div>
    </div>
    """


def make_honeypot_card_html(s: Dict[str, Any], by_id: Dict[str, Any]) -> str:
    """Compile quarantined honeypot details into red-accent cards."""
    cand = by_id[s["candidate_id"]]
    p = cand.get("profile") or {}
    reasons = s.get("honeypot_reasons", [])
    r_html = "".join(f'<li style="margin-bottom:6px;color:#fca5a5;">{r}</li>' for r in reasons)
    if not r_html:
        r_html = '<li style="color:#fca5a5">Inconsistent data detected</li>'
    return f"""
    <div class="hp-card">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
            <span class="rank-chip" style="background:linear-gradient(135deg,#ef4444,#b91c1c);box-shadow:0 4px 14px rgba(239,68,68,0.3);">⚠</span>
            <div>
                <div class="ccard-name">{s['candidate_id']}</div>
                <div class="ccard-role">{p.get('current_title','Professional')} · {p.get('years_of_experience',0)} yrs · {p.get('location','N/A')}</div>
            </div>
        </div>
        <div class="ccard-reason" style="border-left-color:#ef4444;background:rgba(239,68,68,0.06);">
            <strong style="color:#f87171;">Impossibility Indicators:</strong>
            <ul style="margin-top:8px;padding-left:20px;font-size:0.85rem;line-height:1.5;">{r_html}</ul>
        </div>
    </div>
    """


# ── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown(_c("""
<div class="hero">
    <div class="hero-badge">🔬 AI-Powered Talent Intelligence</div>
    <div class="hero-title">Redrob Candidate Discovery<br/>& Ranking Engine</div>
    <div class="hero-sub">
        Recruiter-modelled evaluation for <strong>Senior AI Engineer — Founding Team</strong>.
        Combining career trajectory analysis, skills-trust scoring, semantic matching, and behavioral signals
        to surface the best candidates from massive talent pools.
    </div>
</div>
"""), unsafe_allow_html=True)

# ── File Loading ─────────────────────────────────────────────────────────────
cands: List[Dict[str, Any]] = []
if uploaded is not None and not use_sample:
    cands = load_jsonl(uploaded.read().decode("utf-8", errors="ignore"))
elif use_sample:
    try:
        with open("sample_candidates.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            cands = data if isinstance(data, list) else [data]
    except FileNotFoundError:
        st.sidebar.warning("Bundled sample 'sample_candidates.json' not found.")

# ── Process Ranking ──────────────────────────────────────────────────────────
if rank_clicked and cands:
    # Show skeleton placeholders
    placeholder = st.empty()
    with placeholder.container():
        for _ in range(4):
            st.markdown('<div class="skel"></div>', unsafe_allow_html=True)

    with st.spinner("🧠 Analyzing careers, skills trust, and semantic relevance..."):
        t0 = time.time()
        top_100, honeypots = rank_candidates_pipeline(cands)
        duration = time.time() - t0

        st.session_state["ranked_results"] = top_100
        st.session_state["honeypots"] = honeypots
        st.session_state["by_id"] = {c["candidate_id"]: c for c in cands}
        st.session_state["all_cands"] = cands
        st.session_state["stats"] = compute_dataset_stats(cands, top_100, honeypots)
        st.session_state["ranked"] = True
        st.session_state["duration"] = duration
        st.session_state["page"] = 0

    placeholder.empty()
    st.sidebar.success(f"✅ Ranked {len(cands):,} profiles in {duration:.1f}s")

elif rank_clicked and not cands:
    st.sidebar.error("No candidate data loaded. Upload a file or check the sample box.")


# ── Main Dashboard ───────────────────────────────────────────────────────────
if st.session_state["ranked"]:
    top_100 = st.session_state["ranked_results"]
    honeypots = st.session_state["honeypots"]
    by_id = st.session_state["by_id"]
    stats = st.session_state["stats"]
    duration = st.session_state.get("duration", 0.0)

    # ── KPI Row ──
    n_scanned = len(st.session_state["all_cands"])
    n_shortlist = len(top_100)
    n_honeypots = len(honeypots)
    avg_fit = sum(s["display_score"] for s in top_100) / n_shortlist if n_shortlist else 0
    top_score = top_100[0]["display_score"] if top_100 else 0

    st.markdown(_c(f"""
    <div class="kpi-row">
        <div class="kpi">
            <span class="kpi-icon">📊</span>
            <div class="kpi-label">Candidates Scanned</div>
            <div class="kpi-val v-blue">{n_scanned:,}</div>
            <div class="kpi-sub">Total pool processed in {duration:.1f}s</div>
        </div>
        <div class="kpi">
            <span class="kpi-icon">🏆</span>
            <div class="kpi-label">Shortlisted</div>
            <div class="kpi-val v-purple">{n_shortlist}</div>
            <div class="kpi-sub">Genuine top matches</div>
        </div>
        <div class="kpi">
            <span class="kpi-icon">🚨</span>
            <div class="kpi-label">Quarantined</div>
            <div class="kpi-val v-red">{n_honeypots}</div>
            <div class="kpi-sub">Honeypot profiles flagged</div>
        </div>
        <div class="kpi">
            <span class="kpi-icon">📈</span>
            <div class="kpi-label">Avg Fit Score</div>
            <div class="kpi-val v-green">{avg_fit:.3f}</div>
            <div class="kpi-sub">Normalised display score</div>
        </div>
        <div class="kpi">
            <span class="kpi-icon">⭐</span>
            <div class="kpi-label">Top Score</div>
            <div class="kpi-val v-amber">{top_score:.4f}</div>
            <div class="kpi-sub">Best candidate match</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    # ── Sidebar Filters ──
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔍 Filters")
        search_query = st.text_input("🔎 Search Candidate ID", "").strip()
        min_yoe = st.slider("Min Experience (Years)", 0, 15, 0)
        location_opt = st.selectbox("Location", ["All Regions", "Pune & Noida (Preferred)", "India Only", "International Only"])
        company_opt = st.selectbox("Company Type", ["No Preference", "Product Co. Only", "Exclude Services-Only"])
        core_skills_list = sorted(list(JD.CORE_RETRIEVAL_SKILLS) + list(JD.CORE_ML_SKILLS))
        skills_filter = st.multiselect("Core Skills", core_skills_list)
        if st.button("↻ Reset Filters", use_container_width=True):
            st.rerun()

    # ── Apply Filters ──
    filtered = []
    for s in top_100:
        cid = s["candidate_id"]
        raw = by_id[cid]
        if search_query and search_query.lower() not in cid.lower():
            continue
        if s.get("yoe", 0) < min_yoe:
            continue
        loc = (raw.get("profile") or {}).get("location", "").lower()
        country = (raw.get("profile") or {}).get("country", "").lower()
        if location_opt == "Pune & Noida (Preferred)" and not any(p in loc for p in JD.PREFERRED_CITIES):
            continue
        if location_opt == "India Only" and country != "india":
            continue
        if location_opt == "International Only" and country == "india":
            continue
        if company_opt == "Product Co. Only" and not s.get("has_product_company"):
            continue
        if company_opt == "Exclude Services-Only" and s.get("services_only"):
            continue
        if skills_filter:
            matched_lc = [sk.lower() for sk in s.get("matched_core_skills", [])]
            if not all(sf.lower() in matched_lc for sf in skills_filter):
                continue
        filtered.append(s)

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs([
        "🏆 Top Candidates",
        "🚨 Quarantine Zone",
        "📊 Analytics & Insights",
    ])

    # ── Tab 1: Candidate Cards with pagination ──
    with tab1:
        st.markdown(f"**{len(filtered)}** candidate(s) match your filters")

        if not filtered:
            st.info("No candidates match your current filters. Try relaxing them in the sidebar.")
        else:
            PAGE_SIZE = 10
            total_pages = max(1, math.ceil(len(filtered) / PAGE_SIZE))
            page = st.session_state.get("page", 0)
            page = min(page, total_pages - 1)

            start = page * PAGE_SIZE
            end = start + PAGE_SIZE
            page_items = filtered[start:end]

            for s in page_items:
                st.markdown(_c(make_card_html(s, by_id)), unsafe_allow_html=True)

            # Pagination controls
            if total_pages > 1:
                cols = st.columns([1, 2, 1])
                with cols[0]:
                    if st.button("← Previous", disabled=(page == 0), use_container_width=True):
                        st.session_state["page"] = max(0, page - 1)
                        st.rerun()
                with cols[1]:
                    st.markdown(f"<div style='text-align:center;padding:10px;font-weight:600;'>Page {page + 1} of {total_pages}</div>", unsafe_allow_html=True)
                with cols[2]:
                    if st.button("Next →", disabled=(page >= total_pages - 1), use_container_width=True):
                        st.session_state["page"] = min(total_pages - 1, page + 1)
                        st.rerun()

    # ── Tab 2: Honeypots ──
    with tab2:
        st.markdown("### Quarantined Profiles")
        st.markdown("These profiles contained logical contradictions and were automatically removed from the ranking.")
        if not honeypots:
            st.success("✅ No honeypot or inconsistent profiles found in this dataset.")
        else:
            hp_search = st.text_input("🔎 Search quarantined IDs", "").strip()
            fhp = [h for h in honeypots if (not hp_search or hp_search.lower() in h["candidate_id"].lower())]
            st.caption(f"Showing {len(fhp)} quarantined candidate(s)")
            for h in fhp:
                st.markdown(_c(make_honeypot_card_html(h, by_id)), unsafe_allow_html=True)

    # ── Tab 3: Analytics ──
    with tab3:
        st.markdown("### Dataset & JD Analytics")
        st.markdown("Real-time stats computed from the loaded candidate pool.")

        if stats:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(_c(f"""
                <div class="acard">
                    <div class="acard-title" style="color:#38bdf8">📋 Pipeline Summary</div>
                    <ul class="alist">
                        <li><span class="alist-k">Avg Experience</span><span class="alist-v">{stats['avg_yoe']:.1f} yrs</span></li>
                        <li><span class="alist-k">Max Experience</span><span class="alist-v">{stats['max_yoe']:.0f} yrs</span></li>
                        <li><span class="alist-k">Product Co. Rate</span><span class="alist-v">{stats['product_pct']:.0f}%</span></li>
                        <li><span class="alist-k">Services-Only Rate</span><span class="alist-v" style="color:#f87171">{stats['services_pct']:.0f}%</span></li>
                        <li><span class="alist-k">Stopped Coding Rate</span><span class="alist-v" style="color:#fbbf24">{stats['stopped_coding_pct']:.0f}%</span></li>
                    </ul>
                </div>
                """), unsafe_allow_html=True)

                loc_html = "".join(
                    f'<li><span class="alist-k">{loc.title()}</span><span class="alist-v">{cnt}</span></li>'
                    for loc, cnt in stats.get("locations", [])
                )
                st.markdown(_c(f"""
                <div class="acard">
                    <div class="acard-title" style="color:#a78bfa">📍 Top Locations</div>
                    <ul class="alist">{loc_html}</ul>
                </div>
                """), unsafe_allow_html=True)

            with c2:
                skills_html = "".join(
                    f'<li><span class="alist-k">{skill}</span><span class="alist-v">{cnt} claims</span></li>'
                    for skill, cnt in stats.get("top_skills", [])
                )
                st.markdown(_c(f"""
                <div class="acard">
                    <div class="acard-title" style="color:#34d399">🛠️ Most Claimed Skills</div>
                    <ul class="alist">{skills_html}</ul>
                </div>
                """), unsafe_allow_html=True)

                st.markdown(_c(f"""
                <div class="acard">
                    <div class="acard-title" style="color:#fbbf24">🎯 Job Specification Target</div>
                    <div style="font-size:0.88rem;line-height:1.7;">
                        <strong style="color:#38bdf8">Role:</strong> Founding Team Senior AI Engineer<br/>
                        <strong style="color:#38bdf8">Ideal Experience:</strong> 6–8 years (soft band: 5–9)<br/>
                        <strong style="color:#a78bfa">Core Stack:</strong> Vector Search (FAISS, Pinecone, Qdrant, Milvus),
                        Sentence Transformers, RAG, Information Retrieval, Ranking Systems,
                        Evaluation (NDCG, MRR, MAP), Hybrid Search, Python.
                    </div>
                </div>
                """), unsafe_allow_html=True)

else:
    # ── Landing page ──
    st.markdown(_c("""
    <div class="landing">
        <div class="landing-icon">🔬</div>
        <div class="landing-title">Ready to Discover Top Talent</div>
        <div class="landing-desc">
            Upload a candidate pool (<code>.jsonl</code>) or use the bundled sample in the sidebar,
            then click <strong>Process & Rank</strong> to stream, score, and inspect the ranked shortlist.
        </div>
    </div>
    """), unsafe_allow_html=True)
