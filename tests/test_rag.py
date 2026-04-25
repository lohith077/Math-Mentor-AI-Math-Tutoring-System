import pytest
import os
from rag.indexer import chunk_text
from rag.retriever import retrieve

def test_chunk_text():
    text = "Word " * 500
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert len(chunks[0].split()) == 100

def test_retrieve():
    # Only test if index exists
    db_path = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    if os.path.exists(db_path):
        results = retrieve("quadratic formula", top_k=2)
        assert len(results) <= 2
        if len(results) > 0:
            assert "text" in results[0]
            assert "source" in results[0]
            assert "relevance_score" in results[0]
