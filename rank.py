#!/usr/bin/env python3
"""
Main ranking script.

This file reads candidates, scores them for the Senior AI Engineer job,
keeps the best candidates, and writes the final CSV.

Simple flow:

1. Read candidate profiles.
2. Convert each profile into useful signals/features.
3. Add a semantic match score for the job description.
4. Combine everything into one final score.
5. Remove fake/inconsistent profiles and write the top results.

Example:
    python rank.py --candidates candidates.jsonl --out submission.csv --no-embeddings
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from typing import Any, Dict, List

from src.io_utils import stream_candidates
from src.features import extract_features, candidate_text_for_embedding
from src.behavioral import behavioral_multiplier
from src.semantic import build_semantic_scores
from src.scoring import score_candidate, normalise_scores
from src.reasoning import build_reasoning

TOP_K = 100


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    ap = argparse.ArgumentParser(description="Redrob candidate ranker")
    ap.add_argument("--candidates", required=True, help="path to candidates.jsonl(.gz)")
    ap.add_argument("--out", required=True, help="path to write submission CSV")
    ap.add_argument("--embeddings", default="data/candidate_embeddings.npy",
                    help="optional pre-computed candidate embeddings (.npy)")
    ap.add_argument("--no-embeddings", action="store_true",
                    help="force the TF-IDF semantic backend")
    ap.add_argument("--top", type=int, default=TOP_K)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    # Step 1: read each candidate and prepare the data needed for scoring.
    raw_cands: List[Dict[str, Any]] = []
    feats: List[Dict[str, Any]] = []
    behavior: List[Dict[str, Any]] = []
    texts: List[str] = []

    for cand in stream_candidates(args.candidates):
        raw_cands.append(cand)
        feats.append(extract_features(cand))
        behavior.append(behavioral_multiplier(cand))
        texts.append(candidate_text_for_embedding(cand))

    n = len(raw_cands)
    print(f"[1/4] loaded & featurised {n:,} candidates "
          f"({time.time()-t0:.1f}s)")

    # Step 2: compare candidate text with the job description.
    # If --no-embeddings is used, this runs the offline TF-IDF method.
    precomp = None if args.no_embeddings else args.embeddings
    semantic = build_semantic_scores(
        texts, precomputed_path=precomp, prefer_embeddings=not args.no_embeddings
    )
    print(f"[2/4] semantic scores computed ({time.time()-t0:.1f}s)")

    # Step 3: combine all feature scores into one final score per candidate.
    scored: List[Dict[str, Any]] = []
    for cand, feat, beh, sem in zip(raw_cands, feats, behavior, semantic):
        scored.append(score_candidate(cand, feat, float(sem), beh))
    print(f"[3/4] composite scoring done ({time.time()-t0:.1f}s)")

    # Step 4: sort best-to-worst. If scores tie, smaller candidate_id wins.
    scored.sort(key=lambda s: (-s["final_score"], s["candidate_id"]))

    # Fake/inconsistent profiles should never appear in the final shortlist.
    clean = [s for s in scored if s["honeypot_markers"] == 0]
    top = clean[: args.top]

    normalise_scores(top)
    # After rounding, two display scores can become equal. Keep ties ordered
    # by candidate_id so the validator accepts the file.
    top.sort(key=lambda s: (-s["display_score"], s["candidate_id"]))

    by_id = {c["candidate_id"]: c for c in raw_cands}

    rows: List[Dict[str, Any]] = []
    for rank, s in enumerate(top, start=1):
        reasoning = build_reasoning(by_id[s["candidate_id"]], s, rank)
        rows.append({
            "candidate_id": s["candidate_id"],
            "rank": rank,
            "score": f"{s['display_score']:.4f}",
            "reasoning": reasoning,
        })

    # Final safety check: scores must never increase as rank number goes down.
    _enforce_monotone(rows)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        w.writeheader()
        w.writerows(rows)

    print(f"[4/4] wrote {len(rows)} rows -> {args.out} ({time.time()-t0:.1f}s)")
    _quick_audit(top)
    return 0


def _enforce_monotone(rows: List[Dict[str, Any]]) -> None:
    """Make sure rank 1 has score >= rank 2 >= rank 3, and so on."""
    prev = None
    for r in rows:
        sc = float(r["score"])
        if prev is not None and sc > prev:
            sc = prev
            r["score"] = f"{sc:.4f}"
        prev = sc


def _quick_audit(top: List[Dict[str, Any]]) -> None:
    """Print a small summary so we can quickly inspect the output."""
    hp = sum(1 for s in top if s["honeypot_markers"] > 0)
    print(f"      honeypots in top-{len(top)}: {hp} "
          f"({100*hp/max(len(top),1):.1f}%)  [must be < 10%]")
    print("      top-5 preview:")
    for i, s in enumerate(top[:5], 1):
        skills = ", ".join((s.get("matched_core_skills") or [])[:3])
        print(f"        {i}. {s['candidate_id']}  fit={s['base_fit']:.3f} "
              f"final={s['final_score']:.3f}  core[{skills}]")


if __name__ == "__main__":
    sys.exit(main())
