import os, pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import TOP_K_RETRIEVAL, VECTOR_STORE_DIR
import datetime

INDEX_PATH = VECTOR_STORE_DIR
FAISS_FILE = os.path.join(INDEX_PATH, "math_mentor.index")
META_FILE = os.path.join(INDEX_PATH, "math_mentor_meta.pkl")

_model = None
_index = None
_metadata = None

def _get_faiss():
    global _model, _index, _metadata
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    if _index is None or _metadata is None:
        if os.path.exists(FAISS_FILE) and os.path.exists(META_FILE):
            _index = faiss.read_index(FAISS_FILE)
            with open(META_FILE, 'rb') as f:
                _metadata = pickle.load(f)
        else:
            _index = None
            _metadata = []
    return _index, _metadata, _model

def retrieve(query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """
    Embed query and retrieve top-k relevant chunks using FAISS.
    Returns list of dicts: {text, source, file, relevance_score}
    """
    index, metadata, model = _get_faiss()
    if index is None or index.ntotal == 0:
        return []

    embedding = model.encode(query).astype("float32").reshape(1, -1)
    distances, indices = index.search(embedding, top_k)
    
    chunks = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1 and idx < len(metadata):
            meta = metadata[idx]
            # Convert L2 distance to a pseudo-similarity score
            # A distance of 0 means identical, max distance depends on embedding
            # Euclidean distance for normalized embeddings is between 0 and 2
            sim = max(0.0, 1.0 - (dist / 4.0)) 
            chunks.append({
                "text": meta.get("text", ""),
                "source": meta.get("source", "unknown"),
                "file": meta.get("file", ""),
                "relevance_score": round(sim, 3)
            })
    return chunks
