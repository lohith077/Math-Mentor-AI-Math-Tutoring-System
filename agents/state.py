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
    semantic_confidence: float          # 0-1 from LLM parsing quality
    
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
    hitl_friendly_message: str
    hitl_options: list[str]
    hitl_trigger_agent: str
    hitl_response: dict                # {status: "waiting"|"approved"|"rejected", friendly_message, options}
    
    # Tracking
    agent_trace: list[dict]            # [{agent, action, timestamp}]
    problem_id: Optional[int]          # SQLite row id after saving
    # Follow-up Chat
    chat_history: list[tuple[str, str]] # [(role, content)]
    is_followup: bool
    followup_query: str
    followup_suggestions: list[str]
    
    # Error handling
    error: Optional[str]
