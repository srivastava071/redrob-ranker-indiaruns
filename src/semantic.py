"""
Semantic matching.

This file checks whether a candidate's profile text has the same meaning as
the job description, not just the exact same keywords.

It has two methods:
- Embeddings: better meaning match, if saved vectors/model are available.
- TF-IDF: always offline, no downloads, good fallback.

Both methods return scores from 0 to 1.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from . import jd_spec as JD

# Split the job description into themes so candidates can match one strong area.
JD_FACETS: List[str] = [
    "production embeddings based retrieval with sentence transformers and vector databases like FAISS Pinecone Qdrant Weaviate Milvus",
    "built and shipped an end to end ranking search retrieval recommendation or matching system to real users at scale",
    "designs evaluation frameworks for ranking systems using NDCG MRR MAP and offline to online A B testing",
    "applied machine learning at a product company not a services consulting firm, strong python engineer who writes production code",
    "modern NLP and large language models, transformers, fine tuning, hybrid dense and lexical search",
    JD.IDEAL_PROFILE_TEXT,
]


class TfidfBackend:
    """Offline text similarity using scikit-learn TF-IDF."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._Vec = TfidfVectorizer
        self.vectorizer = None
        self.jd_vec = None

    def fit(self, corpus: List[str]) -> None:
        # Fit on candidates plus JD text so they share one vocabulary.
        self.vectorizer = self._Vec(
            lowercase=True, stop_words="english",
            ngram_range=(1, 2), max_features=50000, sublinear_tf=True,
        )
        self.vectorizer.fit(corpus + JD_FACETS)
        self.jd_vec = self.vectorizer.transform(JD_FACETS)

    def score(self, cand_texts: List[str]) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity
        cand_mat = self.vectorizer.transform(cand_texts)
        sims = cosine_similarity(cand_mat, self.jd_vec)   # (N, n_facets)
        # Reward the strongest JD theme, with a small bonus for overall match.
        best = sims.max(axis=1)
        mean = sims.mean(axis=1)
        raw = 0.7 * best + 0.3 * mean
        # TF-IDF numbers are small, so rescale them to a useful 0..1 range.
        if raw.max() > 0:
            raw = raw / (np.percentile(raw, 99) + 1e-9)
        return np.clip(raw, 0.0, 1.0)


class EmbeddingBackend:
    """Semantic similarity using sentence-transformers embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cpu")
        self.jd_emb = self._normalise(
            self.model.encode(JD_FACETS, batch_size=32, show_progress_bar=False)
        )
        self.cand_emb: Optional[np.ndarray] = None

    @staticmethod
    def _normalise(m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=np.float32)
        n = np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        return m / n

    def load_precomputed(self, path: str) -> bool:
        if os.path.exists(path):
            self.cand_emb = self._normalise(np.load(path))
            return True
        return False

    def encode_candidates(self, cand_texts: List[str]) -> None:
        self.cand_emb = self._normalise(
            self.model.encode(cand_texts, batch_size=64, show_progress_bar=False)
        )

    def score(self, cand_texts: Optional[List[str]] = None) -> np.ndarray:
        if self.cand_emb is None:
            assert cand_texts is not None
            self.encode_candidates(cand_texts)
        sims = self.cand_emb @ self.jd_emb.T          # cosine similarity
        best = sims.max(axis=1)
        mean = sims.mean(axis=1)
        raw = 0.7 * best + 0.3 * mean
        # Convert cosine values into an easy 0..1 score.
        return np.clip((raw - 0.05) / 0.45, 0.0, 1.0)


def build_semantic_scores(
    cand_texts: List[str],
    precomputed_path: Optional[str] = None,
    prefer_embeddings: bool = True,
) -> np.ndarray:
    """
    Return one semantic score for each candidate.

    We try embeddings only when they are practical. If anything fails, we use
    TF-IDF so the project still runs offline.
    """
    if prefer_embeddings:
        # Avoid loading the embedding model for the full dataset unless cached
        # candidate embeddings already exist.
        have_cache = bool(precomputed_path and os.path.exists(precomputed_path))
        if have_cache or len(cand_texts) <= 2000:
            try:
                backend = EmbeddingBackend()
                if have_cache and backend.load_precomputed(precomputed_path):
                    return backend.score()
                if len(cand_texts) <= 2000:
                    return backend.score(cand_texts)
            except Exception as exc:  # noqa: BLE001  (robust fallback by design)
                print(f"[semantic] embedding backend unavailable ({exc}); using TF-IDF")

    tf = TfidfBackend()
    tf.fit(cand_texts)
    return tf.score(cand_texts)
