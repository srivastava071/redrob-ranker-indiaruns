#!/usr/bin/env python3
"""
Optional helper for faster semantic scoring.

This script creates candidate embeddings once and saves them as a .npy file.
Then rank.py can load those saved vectors instead of calculating them again.

You do not need this file for the normal offline run. If embeddings are missing,
rank.py automatically uses the TF-IDF fallback.

Example:
    python build_embeddings.py \
        --candidates ./candidates.jsonl \
        --out data/candidate_embeddings.npy \
        --model sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from src.io_utils import stream_candidates
from src.features import candidate_text_for_embedding


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    ap = argparse.ArgumentParser(description="Pre-compute candidate embeddings")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="data/candidate_embeddings.npy")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--batch-size", type=int, default=256)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # noqa: BLE001
        print(f"[build_embeddings] sentence-transformers unavailable: {exc}")
        print("                   rank.py will use its TF-IDF fallback instead.")
        return 1

    print(f"[build_embeddings] loading model {args.model} ...")
    model = SentenceTransformer(args.model)   # CPU is enough for this project.

    texts = [candidate_text_for_embedding(c) for c in stream_candidates(args.candidates)]
    print(f"[build_embeddings] encoding {len(texts):,} candidate texts ...")

    cand_vecs = model.encode(
        texts, batch_size=args.batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype(np.float32)

    # We only save candidate vectors. rank.py creates the small JD vectors itself.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, cand_vecs)

    print(f"[build_embeddings] wrote {args.out} ({cand_vecs.shape})")
    print(f"[build_embeddings] done in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
