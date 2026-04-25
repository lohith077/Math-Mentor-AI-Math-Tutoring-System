# 🧠 Math Mentor — Full Implementation Prompt for Cursor / AI IDE

> **Paste this entire prompt into Cursor (or any AI IDE) as your project context.**
> Work through each phase in order. Each phase builds on the previous one.

---

## 🎯 PROJECT OVERVIEW

Build a production-grade **Math Mentor** application — a multimodal AI tutoring system that solves JEE-style math problems from image, audio, or text input. The system uses a **multi-agent LangGraph pipeline** with RAG, HITL (human-in-the-loop), memory, and a Streamlit UI.

**Final deliverable**: A fully working, deployed Streamlit app where a student can upload a photo of a math problem, speak a question, or type it — and receive a verified, step-by-step solution with source citations.

---

## 🗂️ TECH STACK (use exactly these)

| Component | Library / Tool |
|---|---|
| Language | Python 3.11+ |
| LLM Provider | Groq API |
| LLM (heavy reasoning) | `llama-3.2-90b-text-preview` via Groq |
| LLM (fast routing) | `mixtral-8x7b-32768` via Groq |
| ASR | `faster-whisper` (model: `small`) |
| OCR | `pytesseract` + `Pillow` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Store | `chromadb` |
| Math computation | `sympy` |
| Agent orchestration | `langgraph` + `langchain-core` |
| Memory | `sqlite3` (built-in) + `chromadb` |
| UI | `streamlit` |
| Env management | `python-dotenv` |

---

## 📁 PROJECT STRUCTURE

Create this exact directory structure before writing any code:

```
AIPlanet_assignment/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── config/
│   └── settings.py
├── input_handlers/
│   ├── __init__.py
│   ├── ocr_handler.py
│   ├── asr_handler.py
│   └── text_handler.py
├── agents/
│   ├── __init__.py
│   ├── state.py
│   ├── graph.py
│   ├── input_agent.py
│   ├── parser_agent.py
│   ├── intent_router_agent.py
│   ├── rag_agent.py
│   ├── helper_agent.py
│   ├── hitl_agent.py
│   ├── verifier_agent.py
│   ├── explainer_agent.py
│   └── guardrail_agent.py
├── rag/
│   ├── __init__.py
│   ├── knowledge_base/
│   │   ├── algebra.md
│   │   ├── probability.md
│   │   ├── calculus.md
│   │   ├── linear_algebra.md
│   │   ├── common_mistakes.md
│   │   └── solution_templates.md
│   ├── indexer.py
│   └── retriever.py
├── tools/
│   ├── __init__.py
│   ├── calculator.py
│   ├── sympy_solver.py
│   └── python_executor.py
├── memory/
│   ├── __init__.py
│   ├── memory_store.py
│   └── similarity_search.py
├── hitl/
│   ├── __init__.py
│   └── hitl_controller.py
└── tests/
    ├── test_input_handlers.py
    ├── test_rag.py
    ├── test_agents.py
    └── test_memory.py
```

---

## ⚙️ PHASE 1 — Config, Environment & Requirements

### `requirements.txt`
```
groq>=0.4.0
langchain-core>=0.2.0
langchain-groq>=0.1.0
langgraph>=0.1.0
chromadb>=0.5.0
sentence-transformers>=2.7.0
faster-whisper>=1.0.0
pytesseract>=0.3.10
Pillow>=10.0.0
sympy>=1.12
streamlit>=1.35.0
python-dotenv>=1.0.0
numpy>=1.26.0
soundfile>=0.12.1
```

### `.env.example`
```
GROQ_API_KEY=your_groq_api_key_here
CHROMA_PERSIST_DIR=./chroma_db
SQLITE_DB_PATH=./memory/math_mentor.db
WHISPER_MODEL=small
OCR_CONFIDENCE_THRESHOLD=70
VERIFIER_CONFIDENCE_THRESHOLD=75
```

### `config/settings.py`
```python
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./memory/math_mentor.db")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
OCR_CONFIDENCE_THRESHOLD = int(os.getenv("OCR_CONFIDENCE_THRESHOLD", 70))
VERIFIER_CONFIDENCE_THRESHOLD = int(os.getenv("VERIFIER_CONFIDENCE_THRESHOLD", 75))

# Model names
HEAVY_MODEL = "llama-3.2-90b-text-preview"
FAST_MODEL = "mixtral-8x7b-32768"

# RAG settings
TOP_K_RETRIEVAL = 5
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Scoped math topics
ALLOWED_TOPICS = ["algebra", "probability", "calculus", "linear_algebra"]
```

---

## 📚 PHASE 2 — RAG Knowledge Base

### Step 2A — Write Knowledge Base Documents

Create each file in `rag/knowledge_base/`. These are the ONLY sources RAG will cite.

**`rag/knowledge_base/algebra.md`** — Include:
- Quadratic formula with derivation
- Factoring patterns (difference of squares, perfect square trinomial, sum/diff of cubes)
- Systems of equations (substitution, elimination, matrix methods)
- Polynomial long division
- Rational root theorem
- Vieta's formulas (sum and product of roots)
- Inequalities and absolute value rules
- AM-GM inequality
- Common algebraic identities

**`rag/knowledge_base/probability.md`** — Include:
- Addition and multiplication rules
- Conditional probability (Bayes' theorem with formula and worked example)
- Permutations and combinations (nPr, nCr formulas)
- Binomial distribution (PMF, mean, variance)
- Expected value calculation
- Independent vs mutually exclusive events
- Common probability pitfalls (e.g., confusing P(A|B) with P(B|A))

**`rag/knowledge_base/calculus.md`** — Include:
- Limit definition and L'Hôpital's rule (with conditions)
- Standard limit forms (sin(x)/x, (1+1/n)^n, etc.)
- Differentiation rules: power, product, quotient, chain rule
- Standard derivatives table (sin, cos, tan, exp, ln, etc.)
- Integration by substitution and by parts
- Definite integral properties
- First and second derivative tests for optimization
- Critical point analysis

**`rag/knowledge_base/linear_algebra.md`** — Include:
- Matrix addition, multiplication rules and conditions
- Determinant calculation (2×2 and 3×3, cofactor expansion)
- Matrix inverse formula (2×2), conditions for invertibility
- Rank and nullity theorem
- Eigenvalue equation (Av = λv) and characteristic polynomial
- Dot product, cross product, magnitude formulas
- System of linear equations — consistency conditions

**`rag/knowledge_base/common_mistakes.md`** — Include:
- Cancellation errors in fractions
- Sign errors in quadratic formula
- Forgetting ± in square root solutions
- Confusing necessary vs sufficient conditions
- Domain restrictions (log, sqrt, division)
- Off-by-one in combinatorics
- Not checking if discriminant is negative
- Forgetting the constant of integration

**`rag/knowledge_base/solution_templates.md`** — Include step-by-step templates for:
- Solving a quadratic equation
- Computing conditional probability (Bayes)
- Finding maxima/minima using derivatives
- Solving a 2×2 system of equations
- Computing a definite integral by substitution

### Step 2B — Indexer

**`rag/indexer.py`**
```python
"""
Chunk, embed, and store all knowledge base documents into ChromaDB.
Run this once before starting the app: python -m rag.indexer
"""
import os, glob, chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config.settings import CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
COLLECTION_NAME = "math_mentor_kb"

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def index_knowledge_base():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    # Reset collection for fresh indexing
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)
    
    doc_files = glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.md"))
    all_chunks, all_ids, all_metadata, all_embeddings = [], [], [], []
    
    for doc_path in doc_files:
        topic = os.path.splitext(os.path.basename(doc_path))[0]
        with open(doc_path, "r") as f:
            text = f.read()
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{topic}_{i}"
            embedding = model.encode(chunk).tolist()
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadata.append({"source": topic, "chunk_index": i, "file": os.path.basename(doc_path)})
            all_embeddings.append(embedding)
    
    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadata, embeddings=all_embeddings)
    print(f"Indexed {len(all_chunks)} chunks from {len(doc_files)} documents.")

if __name__ == "__main__":
    index_knowledge_base()
```

### Step 2C — Retriever

**`rag/retriever.py`**
```python
import chromadb
from sentence_transformers import SentenceTransformer
from config.settings import CHROMA_PERSIST_DIR, TOP_K_RETRIEVAL

COLLECTION_NAME = "math_mentor_kb"
_model = None
_client = None
_collection = None

def _get_collection():
    global _model, _client, _collection
    if _collection is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = _client.get_collection(COLLECTION_NAME)
    return _collection, _model

def retrieve(query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """
    Embed query and retrieve top-k relevant chunks.
    Returns list of dicts: {text, source, score}
    """
    collection, model = _get_collection()
    embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k, include=["documents", "metadatas", "distances"])
    
    chunks = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "file": meta.get("file", ""),
            "relevance_score": round(1 - dist, 3)  # convert distance to similarity
        })
    return chunks
```

---

## 🔧 PHASE 3 — Tools

### `tools/sympy_solver.py`
```python
"""
Symbolic math solver using SymPy.
Supports: solve equations, differentiate, integrate, compute limits, matrix ops.
"""
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

def solve_equation(equation_str: str, variable: str = "x") -> dict:
    """Solve equation like '2*x**2 + 3*x - 5 = 0' for variable."""
    try:
        var = sp.Symbol(variable)
        if "=" in equation_str:
            lhs, rhs = equation_str.split("=", 1)
            eq = sp.Eq(parse_expr(lhs.strip()), parse_expr(rhs.strip()))
        else:
            eq = parse_expr(equation_str.strip())
        solutions = sp.solve(eq, var)
        return {"success": True, "solutions": [str(s) for s in solutions], "latex": [sp.latex(s) for s in solutions]}
    except Exception as e:
        return {"success": False, "error": str(e)}

def differentiate(expr_str: str, variable: str = "x", order: int = 1) -> dict:
    try:
        var = sp.Symbol(variable)
        expr = parse_expr(expr_str)
        result = sp.diff(expr, var, order)
        return {"success": True, "result": str(result), "latex": sp.latex(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def integrate(expr_str: str, variable: str = "x", lower=None, upper=None) -> dict:
    try:
        var = sp.Symbol(variable)
        expr = parse_expr(expr_str)
        if lower is not None and upper is not None:
            result = sp.integrate(expr, (var, lower, upper))
        else:
            result = sp.integrate(expr, var)
        return {"success": True, "result": str(result), "latex": sp.latex(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def compute_limit(expr_str: str, variable: str = "x", point: str = "0", direction: str = "+") -> dict:
    try:
        var = sp.Symbol(variable)
        expr = parse_expr(expr_str)
        pt = parse_expr(point)
        result = sp.limit(expr, var, pt, direction)
        return {"success": True, "result": str(result), "latex": sp.latex(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def matrix_operations(matrix_str: str, operation: str = "det") -> dict:
    """Supported operations: det, inverse, eigenvalues, rank"""
    try:
        mat = sp.Matrix(eval(matrix_str))
        if operation == "det":
            result = mat.det()
        elif operation == "inverse":
            result = mat.inv()
        elif operation == "eigenvalues":
            result = mat.eigenvals()
        elif operation == "rank":
            result = mat.rank()
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}
        return {"success": True, "result": str(result), "latex": sp.latex(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### `tools/calculator.py`
```python
"""Safe numeric calculator for arithmetic evaluation."""
import math, re

SAFE_NAMES = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
SAFE_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow})

def safe_calculate(expression: str) -> dict:
    """Evaluate a math expression safely (no exec/eval of arbitrary code)."""
    # Remove any potentially dangerous patterns
    if re.search(r"(import|exec|eval|open|os\.|sys\.|__)", expression):
        return {"success": False, "error": "Expression contains disallowed operations"}
    try:
        result = eval(expression, {"__builtins__": {}}, SAFE_NAMES)
        return {"success": True, "result": float(result) if isinstance(result, (int, float)) else str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### `tools/python_executor.py`
```python
"""
Sandboxed Python executor for complex mathematical computations.
Uses RestrictedPython-style approach with allowed builtins only.
"""
import re, math, traceback

ALLOWED_IMPORTS = {"math", "sympy", "numpy"}

def execute_math_code(code: str) -> dict:
    """
    Execute math-focused Python code in a restricted namespace.
    Only math, sympy, numpy operations allowed.
    """
    # Block dangerous patterns
    forbidden = ["import os", "import sys", "open(", "exec(", "eval(", "__import__", "subprocess"]
    for pattern in forbidden:
        if pattern in code:
            return {"success": False, "error": f"Forbidden operation detected: {pattern}"}
    
    namespace = {"math": math}
    try:
        import sympy; namespace["sympy"] = sympy
    except ImportError:
        pass
    try:
        import numpy as np; namespace["np"] = np; namespace["numpy"] = np
    except ImportError:
        pass
    
    # Capture stdout
    import io, contextlib
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(compile(code, "<math_executor>", "exec"), namespace)
        output = stdout_capture.getvalue()
        result = namespace.get("result", output.strip() if output.strip() else "Code executed successfully")
        return {"success": True, "result": str(result), "output": output}
    except Exception:
        return {"success": False, "error": traceback.format_exc(limit=3)}
```

---

## 🗄️ PHASE 4 — Memory Layer

### `memory/memory_store.py`
```python
"""
SQLite-based memory store for solved problems, OCR corrections, and user feedback.
"""
import sqlite3, json, datetime
from config.settings import SQLITE_DB_PATH

def get_connection():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS solved_problems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_type TEXT,
            original_input TEXT,
            parsed_problem TEXT,
            topic TEXT,
            solution TEXT,
            solution_steps TEXT,
            tools_used TEXT,
            verification_confidence REAL,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_id INTEGER,
            feedback_type TEXT,
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(problem_id) REFERENCES solved_problems(id)
        );
        
        CREATE TABLE IF NOT EXISTS ocr_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_text TEXT,
            corrected_text TEXT,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS hitl_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger_agent TEXT,
            original_content TEXT,
            corrected_content TEXT,
            action TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

def save_solved_problem(data: dict) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO solved_problems
        (input_type, original_input, parsed_problem, topic, solution, solution_steps, tools_used, verification_confidence, explanation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("input_type"), data.get("original_input"), 
        json.dumps(data.get("parsed_problem", {})),
        data.get("topic"), data.get("solution"),
        json.dumps(data.get("solution_steps", [])),
        json.dumps(data.get("tools_used", [])),
        data.get("verification_confidence", 0),
        data.get("explanation")
    ))
    problem_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return problem_id

def save_feedback(problem_id: int, feedback_type: str, comment: str = ""):
    conn = get_connection()
    conn.execute("INSERT INTO user_feedback (problem_id, feedback_type, comment) VALUES (?, ?, ?)",
                 (problem_id, feedback_type, comment))
    conn.commit()
    conn.close()

def save_ocr_correction(original: str, corrected: str, source: str = "user"):
    conn = get_connection()
    conn.execute("INSERT INTO ocr_corrections (original_text, corrected_text, source) VALUES (?, ?, ?)",
                 (original, corrected, source))
    conn.commit()
    conn.close()

def save_hitl_correction(trigger_agent: str, original: str, corrected: str, action: str):
    conn = get_connection()
    conn.execute("INSERT INTO hitl_corrections (trigger_agent, original_content, corrected_content, action) VALUES (?, ?, ?, ?)",
                 (trigger_agent, original, corrected, action))
    conn.commit()
    conn.close()

def get_all_solved_problems() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM solved_problems ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]
```

### `memory/similarity_search.py`
```python
"""
ChromaDB-based similarity search for finding past solved problems.
Separate collection from the RAG knowledge base.
"""
import chromadb
from sentence_transformers import SentenceTransformer
from config.settings import CHROMA_PERSIST_DIR

MEMORY_COLLECTION = "solved_problems_memory"
_client = None
_model = None
_collection = None

def _get_memory_collection():
    global _client, _model, _collection
    if _collection is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        try:
            _collection = _client.get_collection(MEMORY_COLLECTION)
        except Exception:
            _collection = _client.create_collection(MEMORY_COLLECTION)
    return _collection, _model

def add_problem_to_memory(problem_id: int, problem_text: str, topic: str, solution: str):
    """Embed and store a solved problem for future similarity matching."""
    collection, model = _get_memory_collection()
    embedding = model.encode(problem_text).tolist()
    try:
        collection.add(
            documents=[problem_text],
            ids=[str(problem_id)],
            metadatas=[{"topic": topic, "solution": solution[:500], "problem_id": problem_id}],
            embeddings=[embedding]
        )
    except Exception:
        pass  # ID might already exist on re-runs

def find_similar_problems(query: str, top_k: int = 3, threshold: float = 0.85) -> list[dict]:
    """
    Search memory for similar previously solved problems.
    Returns matches above similarity threshold.
    """
    collection, model = _get_memory_collection()
    try:
        count = collection.count()
        if count == 0:
            return []
        embedding = model.encode(query).tolist()
        results = collection.query(query_embeddings=[embedding], n_results=min(top_k, count),
                                   include=["documents", "metadatas", "distances"])
        matches = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            similarity = 1 - dist
            if similarity >= threshold:
                matches.append({
                    "problem_text": doc,
                    "topic": meta.get("topic"),
                    "solution": meta.get("solution"),
                    "problem_id": meta.get("problem_id"),
                    "similarity": round(similarity, 3)
                })
        return matches
    except Exception:
        return []
```

---

## 🤖 PHASE 5 — LangGraph State

### `agents/state.py`
```python
"""Central LangGraph state shared across all agents."""
from typing import TypedDict, Optional

class MathMentorState(TypedDict):
    # Input
    raw_input: str | bytes
    input_type: str                    # "text" | "image" | "audio"
    
    # After Input Handler
    extracted_text: str
    extraction_confidence: float        # 0-100
    
    # After Parser
    parsed_problem: dict               # {problem_text, topic, variables, constraints, needs_clarification}
    topic: str
    needs_clarification: bool
    
    # After Intent Router
    route: str                         # "solve" | "hitl"
    in_scope: bool
    
    # Memory check result — controls the fork
    memory_hit: bool                   # True = similar problem found, skip RAG+Helper
    memory_matches: list[dict]         # Retrieved similar problems
    
    # RAG Agent output
    retrieved_context: list[dict]      # [{text, source, relevance_score}]
    
    # Helper Agent output
    solution: str
    solution_steps: list[str]
    tools_used: list[str]
    
    # Verifier output
    verification_result: dict          # {is_correct, confidence, issues, notes}
    
    # Explainer output
    explanation: str                   # LaTeX-formatted step-by-step
    
    # Guardrail output
    guardrail_passed: bool
    guardrail_issues: list[str]
    
    # HITL
    hitl_required: bool
    hitl_reason: str
    hitl_trigger_agent: str
    hitl_response: dict                # {action: "approve"|"edit"|"reject", content: str}
    
    # Tracking
    agent_trace: list[dict]            # [{agent, action, timestamp}]
    problem_id: Optional[int]          # SQLite row id after saving
    feedback: dict
    
    # Error handling
    error: Optional[str]
```

---

## 🤖 PHASE 6 — All 9 Agents

### `agents/input_agent.py` — Input Handler (Rule-Based, No LLM)
```python
"""
① Input Handler Agent
Type: Rule-Based | No LLM | Tools: Whisper, Tesseract
"""
from input_handlers.ocr_handler import extract_text_from_image
from input_handlers.asr_handler import transcribe_audio
from input_handlers.text_handler import normalize_text
from agents.state import MathMentorState
from config.settings import OCR_CONFIDENCE_THRESHOLD
import datetime

def input_handler_agent(state: MathMentorState) -> MathMentorState:
    trace = state.get("agent_trace", [])
    trace.append({"agent": "InputHandler", "action": "start", "timestamp": str(datetime.datetime.now())})
    
    input_type = state["input_type"]
    raw = state["raw_input"]
    
    if input_type == "image":
        result = extract_text_from_image(raw)
        extracted = result["text"]
        confidence = result["confidence"]
    elif input_type == "audio":
        result = transcribe_audio(raw)
        extracted = result["text"]
        confidence = result["confidence"]
    else:  # text
        extracted = normalize_text(raw)
        confidence = 100.0
    
    hitl_required = confidence < OCR_CONFIDENCE_THRESHOLD
    reason = f"Extraction confidence {confidence:.1f}% < threshold {OCR_CONFIDENCE_THRESHOLD}%" if hitl_required else ""
    
    trace.append({"agent": "InputHandler", "action": f"extracted text (confidence={confidence:.1f}%)", "timestamp": str(datetime.datetime.now())})
    
    return {
        **state,
        "extracted_text": extracted,
        "extraction_confidence": confidence,
        "hitl_required": hitl_required,
        "hitl_reason": reason,
        "hitl_trigger_agent": "InputHandler" if hitl_required else "",
        "agent_trace": trace
    }
```

### `agents/parser_agent.py` — Parser (Non-ReAct, Mixtral)
```python
"""
② Parser Agent
Type: Non-ReAct | Model: Mixtral-8×7B (Groq)
"""
import json, re, datetime
from groq import Groq
from agents.state import MathMentorState
from config.settings import GROQ_API_KEY, FAST_MODEL

client = Groq(api_key=GROQ_API_KEY)

PARSER_SYSTEM = """You are a math problem parser. Given raw text (possibly from OCR or speech), extract and structure the math problem.

Output ONLY valid JSON in this exact format:
{
  "problem_text": "cleaned problem statement",
  "topic": "algebra|probability|calculus|linear_algebra|unknown",
  "variables": ["x", "y"],
  "constraints": ["x > 0"],
  "needs_clarification": false,
  "clarification_reason": ""
}

Rules:
- Fix OCR artifacts (0 vs O, 1 vs l, etc.)
- Identify the math domain
- Set needs_clarification=true if the problem is ambiguous or incomplete
- Never invent information not in the input"""

def parser_agent(state: MathMentorState) -> MathMentorState:
    if state.get("hitl_required"):
        return state  # Pause for HITL first
    
    trace = state.get("agent_trace", [])
    trace.append({"agent": "Parser", "action": "start", "timestamp": str(datetime.datetime.now())})
    
    response = client.chat.completions.create(
        model=FAST_MODEL,
        messages=[
            {"role": "system", "content": PARSER_SYSTEM},
            {"role": "user", "content": f"Parse this math problem:\n\n{state['extracted_text']}"}
        ],
        temperature=0.1,
        max_tokens=500
    )
    
    raw = response.choices[0].message.content.strip()
    # Extract JSON from response
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        parsed = json.loads(json_match.group() if json_match else raw)
    except Exception:
        parsed = {"problem_text": state["extracted_text"], "topic": "unknown",
                  "variables": [], "constraints": [], "needs_clarification": True,
                  "clarification_reason": "Could not parse problem structure"}
    
    hitl_required = parsed.get("needs_clarification", False)
    trace.append({"agent": "Parser", "action": f"parsed → topic={parsed.get('topic')}, needs_clarification={hitl_required}", "timestamp": str(datetime.datetime.now())})
    
    return {
        **state,
        "parsed_problem": parsed,
        "topic": parsed.get("topic", "unknown"),
        "needs_clarification": hitl_required,
        "hitl_required": hitl_required or state.get("hitl_required", False),
        "hitl_reason": parsed.get("clarification_reason", "") if hitl_required else state.get("hitl_reason", ""),
        "hitl_trigger_agent": "Parser" if hitl_required else state.get("hitl_trigger_agent", ""),
        "agent_trace": trace
    }
```

### `agents/intent_router_agent.py` — Intent Router (Non-ReAct, Mixtral)
```python
"""
③ Intent Router Agent
Type: Non-ReAct | Model: Mixtral-8×7B
Classifies topic, validates scope, routes to memory lookup or HITL
"""
import datetime
from agents.state import MathMentorState
from config.settings import ALLOWED_TOPICS

def intent_router_agent(state: MathMentorState) -> MathMentorState:
    if state.get("hitl_required"):
        return state
    
    trace = state.get("agent_trace", [])
    topic = state.get("topic", "unknown").lower().strip()
    
    in_scope = any(allowed in topic for allowed in ALLOWED_TOPICS) or topic in ALLOWED_TOPICS
    route = "solve" if in_scope else "hitl"
    
    hitl_required = not in_scope
    reason = f"Topic '{topic}' is outside scope. Supported: {', '.join(ALLOWED_TOPICS)}" if hitl_required else ""
    
    trace.append({
        "agent": "IntentRouter",
        "action": f"topic={topic} → {'IN scope' if in_scope else 'OUT OF scope'} → route={route}",
        "timestamp": str(datetime.datetime.now())
    })
    
    return {
        **state,
        "in_scope": in_scope,
        "route": route,
        "hitl_required": hitl_required or state.get("hitl_required", False),
        "hitl_reason": reason if hitl_required else state.get("hitl_reason", ""),
        "hitl_trigger_agent": "IntentRouter" if hitl_required else state.get("hitl_trigger_agent", ""),
        "agent_trace": trace
    }
```

### `agents/rag_agent.py` — RAG Agent (Rule-Based, No LLM)
```python
"""
Ⓐ RAG Agent — Sequential Sub-Agent #1
Type: Rule-Based | No LLM | Tools: ChromaDB + SentenceTransformer
Retrieves relevant knowledge chunks. Passes context to Helper Agent.
"""
import datetime
from rag.retriever import retrieve
from agents.state import MathMentorState

def rag_agent(state: MathMentorState) -> MathMentorState:
    trace = state.get("agent_trace", [])
    
    problem = state.get("parsed_problem", {})
    query = problem.get("problem_text", state.get("extracted_text", ""))
    
    # Enrich query with topic for better retrieval
    topic = state.get("topic", "")
    enriched_query = f"{topic}: {query}" if topic else query
    
    chunks = retrieve(enriched_query)
    
    trace.append({
        "agent": "RAGAgent",
        "action": f"retrieved {len(chunks)} chunks from knowledge base",
        "timestamp": str(datetime.datetime.now())
    })
    
    return {**state, "retrieved_context": chunks, "agent_trace": trace}
```

### `agents/helper_agent.py` — Helper Agent (ReAct, Llama-90B)
```python
"""
Ⓑ Helper Agent — Sequential Sub-Agent #2 (receives context from RAG Agent)
Type: ReAct | Model: Llama-3.2-90B | Tools: sympy_solver, calculator, python_executor
Iterative reason → tool → observe loop to compute the solution.
"""
import json, datetime
from groq import Groq
from agents.state import MathMentorState
from tools.sympy_solver import solve_equation, differentiate, integrate, compute_limit, matrix_operations
from tools.calculator import safe_calculate
from tools.python_executor import execute_math_code
from config.settings import GROQ_API_KEY, HEAVY_MODEL

client = Groq(api_key=GROQ_API_KEY)

TOOLS = {
    "sympy_solve": solve_equation,
    "sympy_diff": differentiate,
    "sympy_integrate": integrate,
    "sympy_limit": compute_limit,
    "sympy_matrix": matrix_operations,
    "calculator": safe_calculate,
    "python_exec": execute_math_code,
}

HELPER_SYSTEM = """You are a mathematical problem solver. Use the available tools to solve the problem step by step.

Available tools (call them by outputting JSON tool calls):
- sympy_solve: Solve equations. Input: {"equation_str": "...", "variable": "x"}
- sympy_diff: Differentiate. Input: {"expr_str": "...", "variable": "x", "order": 1}
- sympy_integrate: Integrate. Input: {"expr_str": "...", "variable": "x", "lower": null, "upper": null}
- sympy_limit: Compute limits. Input: {"expr_str": "...", "variable": "x", "point": "0"}
- calculator: Safe arithmetic. Input: {"expression": "2*3+4"}
- python_exec: Run math Python code. Input: {"code": "result = 2**10"}

When you need a tool, output EXACTLY:
TOOL_CALL: {"tool": "tool_name", "input": {...}}

After getting a result, continue reasoning. When done, output:
FINAL_ANSWER: your complete solution

Use the provided RAG context for formulas and templates. Show all working steps."""

def parse_tool_call(text: str):
    if "TOOL_CALL:" in text:
        try:
            json_str = text.split("TOOL_CALL:")[-1].strip()
            return json.loads(json_str)
        except Exception:
            return None
    return None

def helper_agent(state: MathMentorState) -> MathMentorState:
    trace = state.get("agent_trace", [])
    problem = state.get("parsed_problem", {})
    context_chunks = state.get("retrieved_context", [])
    
    context_text = "\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in context_chunks[:5]
    ])
    
    user_prompt = f"""Problem: {problem.get('problem_text', state.get('extracted_text', ''))}
Topic: {problem.get('topic', '')}
Variables: {problem.get('variables', [])}
Constraints: {problem.get('constraints', [])}

Relevant Knowledge Base Context:
{context_text}

Solve this problem step by step using the tools available."""

    messages = [
        {"role": "system", "content": HELPER_SYSTEM},
        {"role": "user", "content": user_prompt}
    ]
    
    solution_steps = []
    tools_used = []
    max_iterations = 8
    final_solution = ""
    
    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=HEAVY_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1500
        )
        
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        solution_steps.append(f"[Step {iteration+1}] {reply[:300]}...")
        
        tool_call = parse_tool_call(reply)
        if tool_call:
            tool_name = tool_call.get("tool")
            tool_input = tool_call.get("input", {})
            tool_fn = TOOLS.get(tool_name)
            
            if tool_fn:
                try:
                    result = tool_fn(**tool_input)
                except TypeError:
                    # Try calling with positional arg if kwargs fail
                    result = {"success": False, "error": "Invalid tool arguments"}
                
                tools_used.append(tool_name)
                tool_result_msg = f"Tool result for {tool_name}: {json.dumps(result)}"
                messages.append({"role": "user", "content": tool_result_msg})
                trace.append({"agent": "HelperAgent", "action": f"called {tool_name} → {result.get('result', result.get('error', ''))}", "timestamp": str(datetime.datetime.now())})
            else:
                messages.append({"role": "user", "content": f"Tool '{tool_name}' not found. Available: {list(TOOLS.keys())}"})
        
        elif "FINAL_ANSWER:" in reply:
            final_solution = reply.split("FINAL_ANSWER:")[-1].strip()
            break
        
        elif iteration == max_iterations - 1:
            final_solution = reply  # Use last response as solution
    
    trace.append({"agent": "HelperAgent", "action": f"solved in {len(solution_steps)} steps, tools={tools_used}", "timestamp": str(datetime.datetime.now())})
    
    return {
        **state,
        "solution": final_solution,
        "solution_steps": solution_steps,
        "tools_used": list(set(tools_used)),
        "agent_trace": trace
    }
```

### `agents/hitl_agent.py` — HITL Agent (Hybrid, Mixtral)
```python
"""
Ⓒ HITL Agent — Human-in-the-Loop
Type: Hybrid | Model: Mixtral-8×7B
Wraps errors/ambiguities as friendly messages. State is paused; Streamlit handles user response.
"""
import datetime
from groq import Groq
from agents.state import MathMentorState
from config.settings import GROQ_API_KEY, FAST_MODEL

client = Groq(api_key=GROQ_API_KEY)

def generate_friendly_message(reason: str, trigger_agent: str, content: str) -> str:
    """Convert technical error/ambiguity into user-friendly message."""
    response = client.chat.completions.create(
        model=FAST_MODEL,
        messages=[{
            "role": "user",
            "content": f"""Convert this technical issue into a friendly, helpful message for a student:
Triggered by: {trigger_agent}
Issue: {reason}
Content: {content[:500]}

Write 2-3 sentences maximum. Be warm, clear, and tell them exactly what to do. No technical jargon."""
        }],
        temperature=0.3,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def hitl_agent(state: MathMentorState) -> MathMentorState:
    trace = state.get("agent_trace", [])
    friendly_msg = generate_friendly_message(
        reason=state.get("hitl_reason", "Unclear input"),
        trigger_agent=state.get("hitl_trigger_agent", "System"),
        content=state.get("extracted_text", "")
    )
    trace.append({"agent": "HITL", "action": f"paused pipeline — reason: {state.get('hitl_reason', '')[:80]}", "timestamp": str(datetime.datetime.now())})
    return {**state, "hitl_response": {"friendly_message": friendly_msg, "status": "waiting"}, "agent_trace": trace}
```

### `agents/verifier_agent.py` — Verifier (Non-ReAct, Llama-90B)
```python
"""
④ Verifier Agent
Type: Non-ReAct | Model: Llama-3.2-90B
Checks correctness, domain/unit constraints, edge cases. Assigns confidence 0-100.
"""
import json, re, datetime
from groq import Groq
from agents.state import MathMentorState
from config.settings import GROQ_API_KEY, HEAVY_MODEL, VERIFIER_CONFIDENCE_THRESHOLD

client = Groq(api_key=GROQ_API_KEY)

VERIFIER_SYSTEM = """You are a math solution verifier. Check the given solution rigorously.

Evaluate:
1. Plug solutions back into original equation (if applicable)
2. Check domain constraints (no sqrt of negative, no log of zero/negative, no division by zero)
3. Check for edge cases
4. Verify units and dimensions are consistent
5. Check if answer makes physical/mathematical sense

Output ONLY valid JSON:
{
  "is_correct": true,
  "confidence": 85,
  "issues": [],
  "notes": "Brief verification notes",
  "plug_back_check": "x=2: 2(4)+3(2)-5=8+6-5=9≠0"
}

confidence: 0-100 integer. Be conservative — only give 90+ if you've verified by plugging in."""

def verifier_agent(state: MathMentorState) -> MathMentorState:
    trace = state.get("agent_trace", [])
    problem = state.get("parsed_problem", {})
    
    response = client.chat.completions.create(
        model=HEAVY_MODEL,
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": f"""Problem: {problem.get('problem_text', state.get('extracted_text', ''))}
Solution: {state.get('solution', '')}
Steps taken: {'; '.join(state.get('solution_steps', [])[:3])}

Verify this solution."""}
        ],
        temperature=0.1,
        max_tokens=600
    )
    
    raw = response.choices[0].message.content.strip()
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        verification = json.loads(json_match.group() if json_match else raw)
    except Exception:
        verification = {"is_correct": False, "confidence": 30, "issues": ["Could not parse verification"], "notes": raw[:200]}
    
    confidence = verification.get("confidence", 0)
    hitl_needed = confidence < VERIFIER_CONFIDENCE_THRESHOLD
    
    trace.append({"agent": "Verifier", "action": f"confidence={confidence}, is_correct={verification.get('is_correct')}", "timestamp": str(datetime.datetime.now())})
    
    return {
        **state,
        "verification_result": verification,
        "hitl_required": hitl_needed or state.get("hitl_required", False),
        "hitl_reason": f"Verifier confidence {confidence}% below threshold {VERIFIER_CONFIDENCE_THRESHOLD}%" if hitl_needed else state.get("hitl_reason", ""),
        "hitl_trigger_agent": "Verifier" if hitl_needed else state.get("hitl_trigger_agent", ""),
        "agent_trace": trace
    }
```

### `agents/explainer_agent.py` — Explainer (Non-ReAct, Llama-90B)
```python
"""
⑤ Explainer Agent
Type: Non-ReAct | Model: Llama-3.2-90B
Produces student-friendly LaTeX-formatted explanation with RAG citations.
"""
import datetime
from groq import Groq
from agents.state import MathMentorState
from config.settings import GROQ_API_KEY, HEAVY_MODEL

client = Groq(api_key=GROQ_API_KEY)

EXPLAINER_SYSTEM = """You are a patient math tutor explaining a solution to a JEE student.

Rules:
- Use LaTeX for all math expressions (wrap in $...$ for inline, $$...$$ for block)
- Break into clearly numbered steps
- Explain WHY each step works, not just what to do
- Reference the knowledge source when using a formula (e.g., "Using the quadratic formula [Source: algebra]")
- End with a "Key Takeaway" sentence
- Tone: encouraging, clear, never condescending
- Keep each step concise — one concept per step"""

def explainer_agent(state: MathMentorState) -> MathMentorState:
    trace = state.get("agent_trace", [])
    problem = state.get("parsed_problem", {})
    context = state.get("retrieved_context", [])
    
    sources_used = list(set([c["source"] for c in context]))
    
    response = client.chat.completions.create(
        model=HEAVY_MODEL,
        messages=[
            {"role": "system", "content": EXPLAINER_SYSTEM},
            {"role": "user", "content": f"""Problem: {problem.get('problem_text', '')}
Solution: {state.get('solution', '')}
Sources available: {sources_used}

Produce a clear, step-by-step explanation for a student."""}
        ],
        temperature=0.3,
        max_tokens=1200
    )
    
    explanation = response.choices[0].message.content.strip()
    trace.append({"agent": "Explainer", "action": "generated student-friendly explanation", "timestamp": str(datetime.datetime.now())})
    
    return {**state, "explanation": explanation, "agent_trace": trace}
```

### `agents/guardrail_agent.py` — Guardrail (Non-ReAct, Mixtral)
```python
"""
⑥ Guardrail Agent — Final gate before showing answer
Type: Non-ReAct | Model: Mixtral-8×7B
Enforces: domain scope, no harmful content, LaTeX well-formed, citations from RAG only.
"""
import re, datetime
from agents.state import MathMentorState
from config.settings import ALLOWED_TOPICS

def guardrail_agent(state: MathMentorState) -> MathMentorState:
    trace = state.get("agent_trace", [])
    issues = []
    
    explanation = state.get("explanation", "")
    topic = state.get("topic", "")
    verification = state.get("verification_result", {})
    context = state.get("retrieved_context", [])
    
    # Check 1: Topic is in scope
    if not any(t in topic.lower() for t in ALLOWED_TOPICS):
        issues.append(f"Topic '{topic}' outside allowed scope")
    
    # Check 2: No harmful content (basic keyword check)
    harmful_patterns = ["weapon", "explosive", "drug", "illegal", "harm", "kill"]
    full_text = (explanation + state.get("solution", "")).lower()
    if any(p in full_text for p in harmful_patterns):
        issues.append("Potentially harmful content detected")
    
    # Check 3: LaTeX basic well-formedness (check balanced $ signs)
    dollar_count = explanation.count("$")
    if dollar_count % 2 != 0:
        issues.append("LaTeX formatting issue: unbalanced $ delimiters")
    
    # Check 4: Solution exists
    if not state.get("solution") or len(state.get("solution", "").strip()) < 10:
        issues.append("Solution is empty or too short")
    
    passed = len(issues) == 0
    trace.append({"agent": "Guardrail", "action": f"{'PASSED' if passed else 'FAILED'}: {issues}", "timestamp": str(datetime.datetime.now())})
    
    return {**state, "guardrail_passed": passed, "guardrail_issues": issues, "agent_trace": trace}
```

---

## 🔀 PHASE 7 — LangGraph Wiring

### `agents/graph.py`
```python
"""
LangGraph StateGraph — wires all agents with conditional routing.

Flow:
  input_handler → parser → intent_router → memory_lookup →
    [FORK]
    memory_hit=True  → verifier → explainer → guardrail
    memory_hit=False → rag_agent → helper_agent → verifier → explainer → guardrail
    [any agent sets hitl_required=True] → hitl_agent (pipeline pauses)
"""
from langgraph.graph import StateGraph, END
from agents.state import MathMentorState
from agents.input_agent import input_handler_agent
from agents.parser_agent import parser_agent
from agents.intent_router_agent import intent_router_agent
from agents.rag_agent import rag_agent
from agents.helper_agent import helper_agent
from agents.hitl_agent import hitl_agent
from agents.verifier_agent import verifier_agent
from agents.explainer_agent import explainer_agent
from agents.guardrail_agent import guardrail_agent
from memory.similarity_search import find_similar_problems

def memory_lookup_node(state: MathMentorState) -> MathMentorState:
    """Check memory for similar solved problems before invoking RAG+Helper."""
    if state.get("hitl_required"):
        return state
    problem_text = state.get("parsed_problem", {}).get("problem_text", state.get("extracted_text", ""))
    matches = find_similar_problems(problem_text, top_k=3, threshold=0.85)
    memory_hit = len(matches) > 0
    return {
        **state,
        "memory_hit": memory_hit,
        "memory_matches": matches,
        # If cache hit, pre-fill solution from memory for verifier to check
        "solution": matches[0]["solution"] if memory_hit else state.get("solution", ""),
    }

# ── Conditional edges ──

def check_hitl(state: MathMentorState) -> str:
    return "hitl" if state.get("hitl_required") else "continue"

def route_after_memory(state: MathMentorState) -> str:
    if state.get("hitl_required"):
        return "hitl"
    return "verify" if state.get("memory_hit") else "rag"

def route_after_verify(state: MathMentorState) -> str:
    return "hitl" if state.get("hitl_required") else "explain"

def route_after_guardrail(state: MathMentorState) -> str:
    return "end" if state.get("guardrail_passed") else "hitl"

# ── Build graph ──

def build_graph():
    graph = StateGraph(MathMentorState)
    
    # Add all nodes
    graph.add_node("input_handler", input_handler_agent)
    graph.add_node("parser", parser_agent)
    graph.add_node("intent_router", intent_router_agent)
    graph.add_node("memory_lookup", memory_lookup_node)
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("helper_agent", helper_agent)
    graph.add_node("verifier", verifier_agent)
    graph.add_node("explainer", explainer_agent)
    graph.add_node("guardrail", guardrail_agent)
    graph.add_node("hitl", hitl_agent)
    
    # Entry point
    graph.set_entry_point("input_handler")
    
    # Linear pre-processing with HITL checks
    graph.add_conditional_edges("input_handler", check_hitl, {"hitl": "hitl", "continue": "parser"})
    graph.add_conditional_edges("parser", check_hitl, {"hitl": "hitl", "continue": "intent_router"})
    graph.add_conditional_edges("intent_router", check_hitl, {"hitl": "hitl", "continue": "memory_lookup"})
    
    # Memory fork — the key routing decision
    graph.add_conditional_edges("memory_lookup", route_after_memory, {
        "hitl": "hitl",
        "verify": "verifier",   # Cache hit: skip RAG + Helper
        "rag": "rag_agent"      # Cache miss: go through full pipeline
    })
    
    # RAG → Helper (sequential sub-agents, cache-miss path only)
    graph.add_edge("rag_agent", "helper_agent")
    graph.add_edge("helper_agent", "verifier")
    
    # Post-processing
    graph.add_conditional_edges("verifier", route_after_verify, {"hitl": "hitl", "explain": "explainer"})
    graph.add_edge("explainer", "guardrail")
    graph.add_conditional_edges("guardrail", route_after_guardrail, {"end": END, "hitl": "hitl"})
    
    # HITL is a terminal node (Streamlit resumes pipeline on user response)
    graph.add_edge("hitl", END)
    
    return graph.compile()

# Singleton graph instance
math_mentor_graph = build_graph()
```

---

## 📥 PHASE 8 — Input Handlers

### `input_handlers/ocr_handler.py`
```python
"""Tesseract OCR with confidence scoring."""
import pytesseract, re
from PIL import Image
import io

def extract_text_from_image(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes))
    # Preprocess: convert to grayscale
    image = image.convert("L")
    
    # Get detailed OCR data including confidence
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    text = pytesseract.image_to_string(image)
    
    # Calculate mean confidence (filter -1 values)
    confidences = [int(c) for c in data["conf"] if int(c) > 0]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return {"text": text.strip(), "confidence": round(mean_confidence, 1), "raw_data": data}
```

### `input_handlers/asr_handler.py`
```python
"""Whisper ASR using faster-whisper with math phrase normalization."""
import io, re, tempfile, os
from faster_whisper import WhisperModel
from config.settings import WHISPER_MODEL

_model = None

def get_whisper_model():
    global _model
    if _model is None:
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model

MATH_PHRASE_MAP = {
    "square root of": "sqrt(",
    "raised to the power of": "**",
    "raised to": "**",
    "to the power": "**",
    "divided by": "/",
    "times": "*",
    "plus": "+",
    "minus": "-",
    "equals": "=",
    "pi": "π",
}

def normalize_math_phrases(text: str) -> str:
    result = text.lower()
    for phrase, symbol in MATH_PHRASE_MAP.items():
        result = result.replace(phrase, symbol)
    return result

def transcribe_audio(audio_bytes: bytes) -> dict:
    model = get_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        segments, info = model.transcribe(tmp_path, beam_size=5, language="en")
        transcript = " ".join([seg.text for seg in segments]).strip()
        # Estimate confidence from avg log probability (convert to 0-100)
        confidence = min(100, max(0, round((info.all_language_probs[0][1] if hasattr(info, 'all_language_probs') else 0.85) * 100, 1)))
        normalized = normalize_math_phrases(transcript)
        return {"text": normalized, "confidence": confidence, "raw_transcript": transcript}
    finally:
        os.unlink(tmp_path)
```

### `input_handlers/text_handler.py`
```python
"""Normalize typed text input."""
import re

def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Normalize common math notation
    text = text.replace("^", "**")
    text = re.sub(r"(\d)\s*x\s*(\d)", r"\1*\2", text)
    return text
```

---

## 🎨 PHASE 9 — Streamlit UI

### `app.py`
Build the full Streamlit UI with these exact sections:

```python
"""
Math Mentor — Streamlit UI
Full implementation with all features.
"""
import streamlit as st
import tempfile, os, json
from agents.graph import math_mentor_graph
from agents.state import MathMentorState
from memory.memory_store import initialize_db, save_solved_problem, save_feedback, save_ocr_correction, save_hitl_correction
from memory.similarity_search import add_problem_to_memory
from rag.indexer import index_knowledge_base
import datetime

# ── App config ──
st.set_page_config(page_title="Math Mentor", page_icon="🧮", layout="wide")

# ── Initialize on first run ──
@st.cache_resource
def initialize():
    initialize_db()
    try:
        index_knowledge_base()
    except Exception as e:
        st.warning(f"Knowledge base indexing: {e}")
    return True

initialize()

# ── Session state ──
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "hitl_pending" not in st.session_state:
    st.session_state.hitl_pending = False

# ── Header ──
st.title("🧮 Math Mentor")
st.caption("JEE-style math problem solver · RAG + Multi-Agent · HITL")
st.divider()

# ── Input Section ──
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📥 Input")
    input_mode = st.radio("Input mode", ["Text", "Image", "Audio"], horizontal=True)
    
    raw_input = None
    input_type = input_mode.lower()
    
    if input_mode == "Text":
        raw_input = st.text_area("Type your math problem", height=140,
                                  placeholder="e.g. Solve 2x² + 3x - 5 = 0")
    elif input_mode == "Image":
        uploaded = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, caption="Uploaded image", use_column_width=True)
            raw_input = uploaded.read()
    else:  # Audio
        uploaded = st.file_uploader("Upload audio (WAV/MP3)", type=["wav", "mp3", "m4a"])
        if uploaded:
            st.audio(uploaded)
            raw_input = uploaded.read()
    
    solve_btn = st.button("🚀 Solve", type="primary", use_container_width=True, disabled=not raw_input)

# ── HITL Panel ──
if st.session_state.hitl_pending and st.session_state.pipeline_state:
    state = st.session_state.pipeline_state
    hitl_msg = state.get("hitl_response", {}).get("friendly_message", "Please review and correct the input.")
    
    st.warning(f"⚠️ **Human Review Needed** ({state.get('hitl_trigger_agent', 'System')})")
    st.info(hitl_msg)
    
    with st.expander("✏️ Edit & Continue"):
        edited_text = st.text_area("Correct the extracted text:",
                                    value=state.get("extracted_text", ""),
                                    key="hitl_edit")
        col_a, col_b, col_c = st.columns(3)
        
        if col_a.button("✅ Approve & Continue"):
            save_hitl_correction(state.get("hitl_trigger_agent", ""), state.get("extracted_text", ""), edited_text, "approve")
            new_state = {**state, "extracted_text": edited_text, "hitl_required": False,
                         "hitl_response": {"status": "approved"}}
            # Re-run from parser
            from agents.parser_agent import parser_agent
            from agents.intent_router_agent import intent_router_agent
            new_state = parser_agent(new_state)
            new_state = intent_router_agent(new_state)
            st.session_state.pipeline_state = new_state
            st.session_state.hitl_pending = False
            st.rerun()
        
        if col_b.button("❌ Reject"):
            st.session_state.hitl_pending = False
            st.session_state.pipeline_state = None
            st.rerun()
        
        if col_c.button("🔄 Re-run from scratch"):
            st.session_state.hitl_pending = False
            st.session_state.pipeline_state = None
            st.rerun()

# ── Run pipeline ──
if solve_btn and raw_input:
    initial_state: MathMentorState = {
        "raw_input": raw_input,
        "input_type": input_type,
        "extracted_text": "",
        "extraction_confidence": 100.0,
        "parsed_problem": {},
        "topic": "",
        "needs_clarification": False,
        "route": "",
        "in_scope": True,
        "memory_hit": False,
        "memory_matches": [],
        "retrieved_context": [],
        "solution": "",
        "solution_steps": [],
        "tools_used": [],
        "verification_result": {},
        "explanation": "",
        "guardrail_passed": False,
        "guardrail_issues": [],
        "hitl_required": False,
        "hitl_reason": "",
        "hitl_trigger_agent": "",
        "hitl_response": {},
        "agent_trace": [],
        "problem_id": None,
        "feedback": {},
        "error": None
    }
    
    with st.spinner("🧠 Running Math Mentor pipeline..."):
        try:
            final_state = math_mentor_graph.invoke(initial_state)
            st.session_state.pipeline_state = final_state
            
            if final_state.get("hitl_required"):
                st.session_state.hitl_pending = True
            else:
                st.session_state.hitl_pending = False
                # Save to memory
                if final_state.get("solution"):
                    problem_id = save_solved_problem({
                        "input_type": input_type,
                        "original_input": str(raw_input)[:500],
                        "parsed_problem": final_state.get("parsed_problem", {}),
                        "topic": final_state.get("topic", ""),
                        "solution": final_state.get("solution", ""),
                        "solution_steps": final_state.get("solution_steps", []),
                        "tools_used": final_state.get("tools_used", []),
                        "verification_confidence": final_state.get("verification_result", {}).get("confidence", 0),
                        "explanation": final_state.get("explanation", "")
                    })
                    problem_text = final_state.get("parsed_problem", {}).get("problem_text", "")
                    add_problem_to_memory(problem_id, problem_text, final_state.get("topic",""), final_state.get("solution",""))
            st.rerun()
        except Exception as e:
            st.error(f"Pipeline error: {e}")

# ── Results ──
with col_result:
    state = st.session_state.pipeline_state
    if state and not st.session_state.hitl_pending:
        
        # Extracted text preview
        if state.get("extracted_text"):
            with st.expander("📋 Extracted Input", expanded=False):
                st.text(state["extracted_text"])
                conf = state.get("extraction_confidence", 100)
                st.progress(int(conf)/100, text=f"Extraction confidence: {conf:.1f}%")
        
        # Memory reuse notice
        if state.get("memory_hit"):
            st.info(f"♻️ Found similar solved problem in memory (similarity: {state['memory_matches'][0]['similarity']})")
        
        # Confidence indicator
        verif = state.get("verification_result", {})
        if verif:
            conf = verif.get("confidence", 0)
            color = "green" if conf >= 80 else "orange" if conf >= 60 else "red"
            st.metric("Verifier Confidence", f"{conf}%", delta="✅ Correct" if verif.get("is_correct") else "⚠️ Issues found")
        
        # Guardrail status
        if not state.get("guardrail_passed"):
            st.error(f"⛔ Guardrail issues: {', '.join(state.get('guardrail_issues', []))}")
        
        # Final explanation
        if state.get("explanation"):
            st.subheader("📖 Solution")
            st.markdown(state["explanation"])
        
        # RAG Sources
        if state.get("retrieved_context"):
            with st.expander(f"📚 Knowledge Sources ({len(state['retrieved_context'])} retrieved)"):
                for chunk in state["retrieved_context"]:
                    st.caption(f"**[{chunk['source']}]** — relevance: {chunk.get('relevance_score', 0):.3f}")
                    st.text(chunk["text"][:300] + "...")
                    st.divider()
        
        # Agent trace
        with st.expander("🔍 Agent Trace"):
            for step in state.get("agent_trace", []):
                st.text(f"[{step['agent']}] {step['action']}")
        
        # Tools used
        if state.get("tools_used"):
            st.caption(f"🔧 Tools used: {', '.join(state['tools_used'])}")
        
        # Feedback buttons
        st.divider()
        st.subheader("📣 Feedback")
        col_f1, col_f2, col_f3 = st.columns(3)
        
        problem_id = state.get("problem_id")
        if col_f1.button("✅ Correct"):
            if problem_id:
                save_feedback(problem_id, "correct")
            st.success("Thanks!")
        
        if col_f2.button("❌ Incorrect"):
            comment = st.text_input("What was wrong?", key="feedback_comment")
            if problem_id:
                save_feedback(problem_id, "incorrect", comment)
            st.info("Feedback saved.")
        
        if col_f3.button("🔄 Re-check"):
            if state:
                new_state = {**state, "hitl_required": True, "hitl_reason": "User requested re-check",
                             "hitl_trigger_agent": "User"}
                from agents.hitl_agent import hitl_agent
                new_state = hitl_agent(new_state)
                st.session_state.pipeline_state = new_state
                st.session_state.hitl_pending = True
                st.rerun()
```

---

## 🧪 PHASE 10 — Tests

Write tests in `tests/`. Use `pytest`. Cover:

**`tests/test_input_handlers.py`**
- Test OCR on a sample math image (create a synthetic PIL image with text)
- Test ASR returns a transcript dict with `text` and `confidence` keys
- Test text normalization (^ → **, whitespace cleanup)

**`tests/test_rag.py`**
- Test indexer creates ChromaDB collection
- Test retriever returns list of dicts with `text`, `source`, `relevance_score`
- Test retrieval with known algebra query returns algebra.md chunks

**`tests/test_agents.py`**
- Test parser_agent with simple clean text input
- Test intent_router routes algebra → in_scope=True
- Test intent_router routes "cooking recipe" → in_scope=False → hitl_required=True
- Test helper_agent with simple equation (mock Groq response if needed)
- Test verifier_agent returns dict with `confidence` and `is_correct` keys
- Test guardrail catches empty solution

**`tests/test_memory.py`**
- Test initialize_db creates tables
- Test save_solved_problem returns an integer ID
- Test find_similar_problems returns [] on empty collection
- Test add_problem_to_memory + find_similar_problems finds it

---

## 🚀 PHASE 11 — README & Deployment

### `README.md` Must Include:
1. Architecture diagram (embed the Mermaid version)
2. Setup instructions:
   ```bash
   git clone <repo>
   cd AIPlanet_assignment
   pip install -r requirements.txt
   cp .env.example .env
   # Add your GROQ_API_KEY to .env
   python -m rag.indexer
   streamlit run app.py
   ```
3. Environment variables table
4. How to get a free Groq API key
5. Demo video link
6. Evaluation summary

### Deployment (Streamlit Cloud):
1. Push to GitHub
2. Go to share.streamlit.io → New app → connect repo
3. Set `GROQ_API_KEY` in Streamlit secrets
4. Set main file: `app.py`
5. Click Deploy

---

## ✅ BUILD ORDER (follow exactly)

```
Phase 1  → config/settings.py + .env.example + requirements.txt
Phase 2  → Knowledge base docs → rag/indexer.py → rag/retriever.py  [run: python -m rag.indexer]
Phase 3  → tools/ (sympy_solver, calculator, python_executor)
Phase 4  → memory/memory_store.py → memory/similarity_search.py  [run: initialize_db()]
Phase 5  → agents/state.py
Phase 6  → All 9 agents (input → parser → router → rag → helper → hitl → verifier → explainer → guardrail)
Phase 7  → agents/graph.py  [test: invoke with a simple text input]
Phase 8  → input_handlers/ (ocr, asr, text)
Phase 9  → app.py  [run: streamlit run app.py]
Phase 10 → tests/  [run: pytest tests/]
Phase 11 → README.md + deploy
```

---

## 🔑 KEY IMPLEMENTATION RULES

1. **Memory fork is the most critical routing logic** — `memory_hit: bool` in state determines whether RAG+Helper runs at all. Get this right first.
2. **RAG Agent never generates text** — it only retrieves. No LLM call in rag_agent.py.
3. **Helper Agent is the only ReAct agent** — it's the only one with a reason→tool→observe loop.
4. **HITL pauses the graph** — after `hitl_agent` runs, the graph ends. Streamlit resumes by calling specific agents directly.
5. **All agents must update `agent_trace`** — this powers the UI trace panel.
6. **Never hallucinate citations** — Explainer only cites sources present in `retrieved_context`.
7. **Guardrail is the last gate** — nothing reaches the UI without passing guardrail.
8. **SQLite + ChromaDB are separate concerns** — SQLite = full records, ChromaDB = embeddings for similarity.
9. **Use `@st.cache_resource` for model loading** — Whisper and SentenceTransformer should load once.
10. **Groq API key is the only external dependency** — everything else runs locally.

---

## 📦 DELIVERABLES CHECKLIST

- [ ] GitHub repo with clean commit history
- [ ] `.env.example` with all variables documented
- [ ] `README.md` with architecture diagram + setup instructions
- [ ] Working Streamlit app (all 3 input modes functional)
- [ ] RAG knowledge base (6 markdown docs, indexed into ChromaDB)
- [ ] All 9 agents implemented and wired in LangGraph
- [ ] Memory layer (SQLite + ChromaDB) persisting between sessions
- [ ] HITL triggering correctly from 4 different agents
- [ ] Feedback buttons saving to SQLite
- [ ] Agent trace visible in UI
- [ ] All tests passing (`pytest tests/`)
- [ ] Deployed app link (Streamlit Cloud / HuggingFace Spaces)
- [ ] 3–5 min demo video showing: image→solution, audio→solution, HITL in action, memory reuse
