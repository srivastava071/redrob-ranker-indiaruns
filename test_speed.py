import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from src.io_utils import stream_candidates
from src.features import candidate_text_for_embedding

print("Streaming candidates...")
cands = list(stream_candidates("sample_candidates.jsonl"))
texts = [candidate_text_for_embedding(c) for c in cands]
print(f"Loaded {len(texts)} candidates.")

# Test TF-IDF
t0 = time.time()
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
vectorizer.fit(texts)
vecs = vectorizer.transform(texts)
print(f"TF-IDF fit and transform: {time.time() - t0:.2f}s")

# Test MiniLM
print("Loading all-MiniLM-L6-v2...")
t0 = time.time()
model_minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print(f"MiniLM load time: {time.time() - t0:.2f}s")

t0 = time.time()
minilm_vecs = model_minilm.encode(texts, batch_size=256, show_progress_bar=False)
print(f"MiniLM encode time for {len(texts)} texts: {time.time() - t0:.2f}s")

# Test mpnet
print("Loading all-mpnet-base-v2...")
t0 = time.time()
model_mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
print(f"mpnet load time: {time.time() - t0:.2f}s")

t0 = time.time()
mpnet_vecs = model_mpnet.encode(texts, batch_size=256, show_progress_bar=False)
print(f"mpnet encode time for {len(texts)} texts: {time.time() - t0:.2f}s")
