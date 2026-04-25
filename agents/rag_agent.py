"""
Ⓐ RAG Agent — Sequential Sub-Agent #1
Type: Rule-Based | No LLM | Tools: ChromaDB + SentenceTransformer
Retrieves relevant knowledge chunks. Passes context to Helper Agent.
"""
import datetime
from langsmith import traceable
from rag.retriever import retrieve
from agents.state import MathMentorState

@traceable(run_type="chain", name="RAG Agent")
def rag_agent(state: MathMentorState) -> MathMentorState:
    print(f"[DEBUG] 📚 RAG Agent starting...")
    start_time = datetime.datetime.now()
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
    
    duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"[DEBUG] 📚 RAG Agent finished in {duration:.2f}s")
    
    return {**state, "retrieved_context": chunks, "agent_trace": trace}
