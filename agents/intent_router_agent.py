"""
③ Intent Router Agent
Type: Logic Node (Pure Python) | No LLM
Classifies topic, validates scope, routes to memory lookup or HITL
"""
import datetime
from langsmith import traceable
from agents.state import MathMentorState
from config.settings import ALLOWED_TOPICS

@traceable(run_type="chain", name="Intent Router Agent")
def intent_router_agent(state: MathMentorState) -> MathMentorState:
    if state.get("hitl_required"):
        return state
    
    print(f"[DEBUG] 🚦 Intent Router Agent starting...")
    start_time = datetime.datetime.now()
    trace = state.get("agent_trace", [])
    topic = state.get("topic", "unknown").lower().strip()
    
    in_scope = any(allowed in topic for allowed in ALLOWED_TOPICS) or topic in ALLOWED_TOPICS
    is_approved = state.get("hitl_response", {}).get("status") == "approved"
    hitl_required = not in_scope
    
    if is_approved and hitl_required:
        print("[DEBUG] 🚦 Intent Router Agent: Bypassing scope check because user manually approved.")
        hitl_required = False
        in_scope = True
        route = "solve"
    else:
        route = "solve" if in_scope else "hitl"

    reason = f"Topic '{topic}' is outside scope. Supported: {', '.join(ALLOWED_TOPICS)}" if hitl_required else ""
    
    trace.append({
        "agent": "IntentRouter",
        "action": f"topic={topic} → {'IN scope' if in_scope else 'OUT OF scope'} → route={route} (Approved: {is_approved})",
        "timestamp": str(datetime.datetime.now())
    })
    
    duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"[DEBUG] 🚦 Intent Router Agent finished in {duration:.2f}s")

    return {
        **state,
        "in_scope": in_scope,
        "route": route,
        "hitl_required": hitl_required or state.get("hitl_required", False),
        "hitl_reason": reason if hitl_required else state.get("hitl_reason", ""),
        "hitl_friendly_message": f"I noticed this problem is about {topic}, which is currently outside my core JEE syllabus. Would you like me to try solving it anyway, or do you have another math problem?" if hitl_required else state.get("hitl_friendly_message", ""),
        "hitl_options": ["Try solving it anyway", "Ask a different problem"] if hitl_required else state.get("hitl_options", []),
        "hitl_trigger_agent": "IntentRouter" if hitl_required else state.get("hitl_trigger_agent", ""),
        "agent_trace": trace
    }
