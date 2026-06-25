# Redrob Intelligent Candidate Ranker

Ranks 100,000 candidates for the **Senior AI Engineer — Founding Team** job
description the way a great recruiter would: rewarding genuine fit over keyword
density, weighing real-world availability, and quietly filtering out the
honeypots and keyword-stuffer traps planted in the dataset.

Built for the **India Runs — Data & AI Challenge** (Redrob: Intelligent
Candidate Discovery & Ranking).

---

## Reproduce the submission (one command)

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --no-embeddings
```

* **~65 seconds**, single CPU core, < 2 GB RAM, **no network, no GPU, no LLM**.
* Produces a validator-clean `submission.csv` (100 rows: `candidate_id,rank,score,reasoning`).
* `--no-embeddings` selects the deterministic, fully-offline TF-IDF semantic
  backend — exactly what produced the included `submission.csv`.

Validate it with the official validator:

```bash
python validate_submission.py submission.csv      # -> "Submission is valid."
```

---

## The problem, and why naïve approaches fail

The JD is looking for a specific person: someone who has **shipped production
embeddings-retrieval / ranking / recommendation systems at a product company**,
still writes code, and is reachable. The dataset is adversarial by design:

| Trap in the data | Naïve ranker does… | What we do |
|---|---|---|
| **Keyword stuffers** — an *HR Manager* listing RAG, FAISS, LLMs | ranks them top (the provided `sample_submission.csv` does exactly this) | a **keyword-stuffer gate**: a non-technical career collapses the title/career signal, so stuffed skills can't rescue them |
| **Plain-language Tier-5s** — built a recommender at a product company but never write "RAG" | misses them (no keywords to match) | a **semantic component** matches the *concepts* in their prose to the JD, surfacing them |
| **Behavioral twins** — near-identical profiles differing only in signals | can't separate them | an **availability modifier** + ghost gate ranks the reachable twin higher |
| **Honeypots** (~80) — impossible profiles (tenure > company age; "expert" in a skill used 0 months) | may rank them | **impossibility checks** detect and remove them from the shortlist |
| **Distractors** — CV/speech "AI" with no NLP/IR | counts them as AI fit | a **distractor penalty** down-weights CV/speech-only profiles |
| **Services-only careers** (TCS/Infosys/…) | treats as equal | a **services penalty** reflects the JD's product-company preference |

Our top-100 has **zero overlap** with the trap submission's keyword-stuffer
picks, and **zero honeypots**.

---

## How the score is built

For every candidate:

```
base_fit = 0.26·title_career      # role + career trajectory (keyword-stuffer gate lives here)
         + 0.22·career_evidence   # plain-language proof of shipping ranking/search/recsys
         + 0.18·skills_trust      # skills weighted by endorsements × duration × proficiency
         + 0.14·semantic          # similarity of their prose to the JD's concepts
         + 0.10·experience        # fit to the 6–8yr ideal (5–9 soft band)
         + 0.06·location          # Pune/Noida preferred; NCR/Hyd/Mumbai welcome
         + 0.04·education

final    = base_fit
         × behavioral_mult        # availability / responsiveness / recency  (gentle, tier-internal)
         × services_penalty       # services-only career
         × distractor_penalty     # CV/speech-only "AI"
         × coding_recency_penalty # senior who stopped writing code
         × product_bonus          # small nudge for product-company experience
         × honeypot_penalty       # impossible profiles
```

Every weight and multiplier maps to an **explicit clause in the JD**, and every
per-candidate value is logged, so the ranking is fully auditable and the
`reasoning` column is grounded in real numbers (never an LLM hallucination).

### Three design choices worth calling out

1. **Behavioral signals re-rank *within* a tier, they don't eject talent.**
   An earlier multiplicative model multiplied many sub-1.0 factors together, so
   a perfectly *available* strong candidate compounded down and fell out of the
   shortlist. The current model is **additive and bounded** (`~[0.93, 1.05]` for
   anyone genuinely reachable) with a separate **sharp "ghost" gate** for the
   JD's explicit case — *"perfect on paper but hasn't logged in for 6 months and
   a 5% response rate is, for hiring purposes, not available."* The result: the
   top-10 is driven by genuine fit, while ghosts and behavioral-twin losers are
   still decisively down-weighted.

2. **The semantic component is the Tier-5 rescue.** It compares each candidate's
   prose to the JD broken into *facets* (production retrieval; shipped
   ranking/recsys at scale; evaluation with NDCG/MRR/MAP; product-company
   engineer who codes; modern NLP/LLMs). This is what lets a candidate who
   "built a product recommender" rank highly without ever typing "RAG."

3. **Robustness and Exception Defense.** The system is designed to parse real-world unclean candidate datasets without crashing. It features:
   - Robust `_safe_int` and `_safe_float` helpers in the behavioral layer to handle non-numeric or float-as-string data fields.
   - KeyError mitigation in honeypot checks via fallback default retrievals.
   - Flexible file ingestion in `app.py` that processes standard JSON lists/arrays alongside raw JSONL files.

---

## Repository layout

```
redrob-ranker/
├── rank.py                  # entry point: candidates.jsonl -> submission.csv
├── build_embeddings.py      # OPTIONAL: pre-compute sentence-transformer vectors
├── app.py                   # Streamlit sandbox demo (upload a sample, see ranking)
├── requirements.txt
├── submission_metadata.yaml # fill team/repo/sandbox fields before submitting
├── submission.csv           # the validated top-100 output
├── sample_candidates.json   # 50-candidate sample for the sandbox demo
├── src/
│   ├── jd_spec.py           # the JD encoded as data: skill vocab, phrases, weights
│   ├── io_utils.py          # memory-safe streaming JSONL(.gz) loader
│   ├── features.py          # interpretable feature extraction per candidate
│   ├── semantic.py          # EmbeddingBackend (primary) + TfidfBackend (offline fallback)
│   ├── behavioral.py        # availability/engagement modifier + ghost gate
│   ├── honeypot.py          # impossibility checks
│   ├── scoring.py           # base_fit, gate multipliers, normalisation
│   └── reasoning.py         # deterministic, fact-grounded reasoning strings
└── tests/
    └── test_smoke.py        # contract tests: stuffer gated, honeypot caught, ghost down-weighted…
```

---

## Optional: the embedding semantic path (primary in spirit, offline by default)

The repo ships with two interchangeable semantic backends behind one interface:

* **TF-IDF** (default, `--no-embeddings`): zero downloads, deterministic, fully
  offline. This produced the submitted CSV.
* **sentence-transformers** (`all-MiniLM-L6-v2`): higher-fidelity concept
  matching. Because the rules forbid network/GPU *during ranking* but allow
  embedding **pre-computation** to run separately, you cache the vectors first:

```bash
# one-time, offline-cacheable pre-computation (may exceed the 5-min rank budget)
python build_embeddings.py --candidates ./candidates.jsonl --out data/candidate_embeddings.npy

# ranking then memory-maps the cache; still CPU-only and well under budget
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

If the cache (or the model) is absent, `rank.py` **automatically falls back to
TF-IDF** — it never blocks on a missing model or network.

---

## Run the tests / the demo

```bash
python tests/test_smoke.py        # or: python -m pytest tests/ -q
streamlit run app.py              # local sandbox at http://localhost:8501
```

---

## Compute & constraints compliance

| Constraint | This pipeline |
|---|---|
| Ranking ≤ 5 min wall-clock | ✅ ~65 s for 100k on 1 core |
| ≤ 16 GB RAM | ✅ < 2 GB (streaming loader) |
| CPU-only, no GPU at rank time | ✅ |
| No network / hosted-LLM during ranking | ✅ fully offline |
| ≤ 5 GB disk | ✅ |
| 0 honeypots in top-10, < 10% in top-100 | ✅ 0 in top-100 |

---

## AI tools

Claude (Anthropic) was used as a pair-programming and code-review assistant for
architecture discussion, drafting/refactoring modules, distribution diagnostics,
and weight tuning. No candidate data was sent to any hosted LLM, and the ranking
pipeline itself contains **no LLM calls**. See `submission_metadata.yaml`.
