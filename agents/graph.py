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
from langsmith import traceable
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
from agents.followup_agent import followup_agent
from memory.similarity_search import find_similar_problems

@traceable(run_type="chain", name="Memory Lookup Node")
def memory_lookup_node(state: MathMentorState) -> MathMentorState:
    """Check memory for similar solved problems before invoking RAG+Helper."""
    if state.get("hitl_required"):
        return state
    problem_text = state.get("parsed_problem", {}).get("problem_text", state.get("extracted_text", ""))
    
    # User optimization: top_k=1 because in math, similar is not good enough
    matches = find_similar_problems(problem_text, top_k=1, threshold=0.85)
    
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

def check_followup_and_hitl(state: MathMentorState) -> str:
    if state.get("is_followup"):
        return "followup"
    return "hitl" if state.get("hitl_required") else "continue"

def route_after_memory(state: MathMentorState) -> str:
    # Removed HITL check here as per user request (logic handled earlier)
    return "verify" if state.get("memory_hit") else "rag"

def route_after_verify(state: MathMentorState) -> str:
    if state.get("hitl_required"):
        return "hitl"
    
    # User Optimization: Cache Fallback
    # If we had a memory hit but the verifier found it incorrect, 
    # don't give up—fall back to the RAG+Helper pipeline.
    verif = state.get("verification_result", {})
    if state.get("memory_hit") and not verif.get("is_correct"):
        print(f"[DEBUG] 🔄 Cache hit was mathematically incorrect. Falling back to RAG.")
        return "fallback_to_rag"
        
    return "explain"

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
    graph.add_node("followup", followup_agent)
    
    # Entry point
    graph.set_entry_point("input_handler")
    
    # Conditional Entry: If it's a followup, bypass processing and go to Followup Agent
    graph.add_conditional_edges("input_handler", check_followup_and_hitl, {
        "followup": "followup",
        "hitl": "hitl", 
        "continue": "parser"
    })
    graph.add_conditional_edges("parser", check_hitl, {"hitl": "hitl", "continue": "intent_router"})
    graph.add_conditional_edges("intent_router", check_hitl, {"hitl": "hitl", "continue": "memory_lookup"})
    
    # Memory fork — the key routing decision
    graph.add_conditional_edges("memory_lookup", route_after_memory, {
        "verify": "verifier",   # Cache hit: skip RAG + Helper
        "rag": "rag_agent"      # Cache miss: go through full pipeline
    })
    
    # RAG → Helper (sequential sub-agents, cache-miss path only)
    graph.add_edge("rag_agent", "helper_agent")
    graph.add_edge("helper_agent", "verifier")
    
    # Post-processing
    graph.add_conditional_edges("verifier", route_after_verify, {
        "hitl": "hitl", 
        "explain": "explainer",
        "fallback_to_rag": "rag_agent"  # Cache hit failed verification
    })
    graph.add_edge("explainer", "guardrail")
    graph.add_conditional_edges("guardrail", route_after_guardrail, {"end": END, "hitl": "hitl"})
    
    # Followup is a terminal node in the graph (resumes on next user interaction)
    graph.add_edge("followup", END)
    
    # HITL is a terminal node
    graph.add_edge("hitl", END)
    
    return graph.compile()

# Singleton graph instance with explicit name for tracing
math_mentor_graph = build_graph()
math_mentor_graph.name = "Math Mentor StateGraph"
