"""
Chunk, embed, and store all knowledge base documents into FAISS index.
Run this once before starting the app: python -m rag.indexer
"""
import os, glob, json, pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_STORE_DIR
import datetime

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
INDEX_PATH = VECTOR_STORE_DIR
FAISS_FILE = os.path.join(INDEX_PATH, "math_mentor.index")
META_FILE = os.path.join(INDEX_PATH, "math_mentor_meta.pkl")

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks while keeping math blocks intact."""
    import re
    # Pattern to find math blocks to protect them
    math_pattern = r'(\$\$.*?\$\$|\\\[.*?\\\])'
    parts = re.split(math_pattern, text, flags=re.DOTALL)
    
    chunks = []
    current_chunk = ""
    
    for part in parts:
        # If part is a math block, keep it together
        if re.match(math_pattern, part, re.DOTALL):
            if len(current_chunk) + len(part) > chunk_size * 5: # Safety limit for huge blocks
                if current_chunk: chunks.append(current_chunk.strip())
                chunks.append(part.strip())
                current_chunk = ""
            else:
                current_chunk += " " + part
        else:
            # For normal text, split into words and add to current_chunk
            words = part.split()
            for word in words:
                if len(current_chunk.split()) >= chunk_size:
                    chunks.append(current_chunk.strip())
                    # Keep overlap by taking last N words
                    overlap_words = current_chunk.split()[-overlap:]
                    current_chunk = " ".join(overlap_words) + " " + word
                else:
                    current_chunk += " " + word
                    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def index_knowledge_base():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    doc_files = glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.md"))
    all_chunks, all_metadata, all_embeddings = [], [], []
    
    for doc_path in doc_files:
        topic = os.path.splitext(os.path.basename(doc_path))[0]
        with open(doc_path, "r") as f:
            text = f.read()
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk)
            all_chunks.append(chunk)
            all_metadata.append({"source": topic, "chunk_index": i, "file": os.path.basename(doc_path), "text": chunk})
            all_embeddings.append(embedding)
    
    if len(all_embeddings) == 0:
        print("No documents found.")
        return

    embeddings_np = np.array(all_embeddings).astype("float32")
    
    # Create FAISS Index
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    os.makedirs(INDEX_PATH, exist_ok=True)
    faiss.write_index(index, FAISS_FILE)
    with open(META_FILE, 'wb') as f:
        pickle.dump(all_metadata, f)
        
    print(f"Indexed {len(all_chunks)} chunks from {len(doc_files)} documents into FAISS.")

if __name__ == "__main__":
    index_knowledge_base()
