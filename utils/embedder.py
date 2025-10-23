# utils/embedder.py

import faiss
import numpy as np
import os
import hashlib
import pickle
from sentence_transformers import SentenceTransformer

CACHE_DIR = "data/faiss_cache"

def _hash_file(file_path):
    """Generate a SHA256 hash of the file for cache uniqueness."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()[:16]


def build_faiss_index(chunks, resume_path, model_name='all-MiniLM-L6-v2'):
    """
    Create or load FAISS index for text chunks.
    - If a cached index exists for this resume hash, load it.
    - Else, compute and store new embeddings.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    file_hash = _hash_file(resume_path)

    index_path = os.path.join(CACHE_DIR, f"index_{file_hash}.faiss")
    chunks_path = os.path.join(CACHE_DIR, f"chunks_{file_hash}.pkl")

    # === Try loading from cache ===
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        print(f"[Cache] Loading existing FAISS index for resume ({file_hash})...")
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        embedder = SentenceTransformer(model_name)
        return index, embedder, chunks

    # === Else, build new index ===
    print(f"[Cache] Building new FAISS index for resume ({file_hash})...")
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save cache
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print("[Cache] FAISS index and chunks saved successfully.")
    return index, embedder, chunks
