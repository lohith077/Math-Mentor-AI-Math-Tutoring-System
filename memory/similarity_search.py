"""
FAISS-based similarity search for finding past solved problems.
Separate collection from the RAG knowledge base.
"""
import os, pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import VECTOR_STORE_DIR

INDEX_PATH = VECTOR_STORE_DIR
MEMORY_FAISS_FILE = os.path.join(INDEX_PATH, "memory.index")
MEMORY_META_FILE = os.path.join(INDEX_PATH, "memory_meta.pkl")

_model = None
_index = None
_metadata = None

def _get_memory_faiss():
    global _model, _index, _metadata
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    if _index is None or _metadata is None:
        if os.path.exists(MEMORY_FAISS_FILE) and os.path.exists(MEMORY_META_FILE):
            _index = faiss.read_index(MEMORY_FAISS_FILE)
            with open(MEMORY_META_FILE, 'rb') as f:
                _metadata = pickle.load(f)
        else:
            # Initialize empty index
            dimension = 384  # MiniLM-L6-v2 dimension
            _index = faiss.IndexFlatL2(dimension)
            _metadata = []
            os.makedirs(INDEX_PATH, exist_ok=True)
    return _index, _metadata, _model

def _save_memory_faiss():
    global _index, _metadata
    if _index is not None and _metadata is not None:
        faiss.write_index(_index, MEMORY_FAISS_FILE)
        with open(MEMORY_META_FILE, 'wb') as f:
            pickle.dump(_metadata, f)

def add_problem_to_memory(problem_id: int, problem_text: str, topic: str, solution: str):
    """Embed and store a solved problem for future similarity matching."""
    index, metadata, model = _get_memory_faiss()
    
    embedding = model.encode(problem_text).astype("float32").reshape(1, -1)
    
    # Check if we already have it to avoid duplicates
    for meta in metadata:
        if meta.get("problem_id") == problem_id:
            return

    index.add(embedding)
    metadata.append({
        "topic": topic, 
        "solution": solution[:500] if solution else "", 
        "problem_id": problem_id,
        "problem_text": problem_text
    })
    _save_memory_faiss()

def find_similar_problems(query: str, top_k: int = 3, threshold: float = 0.85) -> list[dict]:
    """
    Search memory for similar previously solved problems.
    Returns matches above similarity threshold.
    """
    index, metadata, model = _get_memory_faiss()
    if index is None or index.ntotal == 0:
        return []
        
    embedding = model.encode(query).astype("float32").reshape(1, -1)
    # Search for min(top_k, total elements)
    k = min(top_k, index.ntotal)
    if k == 0:
        return []
        
    distances, indices = index.search(embedding, k)
    
    matches = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1 and idx < len(metadata):
            meta = metadata[idx]
            # Convert L2 distance to pseudo-similarity
            similarity = max(0.0, 1.0 - (dist / 4.0))
            if similarity >= threshold:
                matches.append({
                    "problem_text": meta.get("problem_text", ""),
                    "topic": meta.get("topic"),
                    "solution": meta.get("solution"),
                    "problem_id": meta.get("problem_id"),
                    "similarity": round(similarity, 3)
                })
    return matches
