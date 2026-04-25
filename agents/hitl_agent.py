"""
Ⓒ HITL Agent — Human-in-the-Loop
Type: Hybrid | Model: Mixtral-8×7B
Wraps errors/ambiguities as friendly messages. State is paused; Streamlit handles user response.
"""
import datetime
from groq import Groq
from langsmith import traceable
from agents.state import MathMentorState
from config.settings import GROQ_API_KEY, FAST_MODEL

client = Groq(api_key=GROQ_API_KEY)

@traceable(run_type="llm", name="HITL Friendly Message LLM")
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

Write 2-3 sentences maximum. Be warm, clear, and tell them exactly what to do. Mention that they can provide feedback and that our support team or a live math tutor will connect with them to resolve any doubts. No technical jargon."""
        }],
        temperature=0.3,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

@traceable(run_type="chain", name="HITL Agent")
def hitl_agent(state: MathMentorState) -> MathMentorState:
    print(f"[DEBUG] 🚨 HITL Agent starting...")
    start_time = datetime.datetime.now()
    trace = state.get("agent_trace", [])
    
    # Use existing friendly message if available (from Guardrail/Parser/IntentRouter)
    friendly_msg = state.get("hitl_friendly_message")
    options = state.get("hitl_options", [])
    
    if not friendly_msg:
        # Fallback to general generation
        friendly_msg = generate_friendly_message(
            reason=state.get("hitl_reason", "Unclear input"),
            trigger_agent=state.get("hitl_trigger_agent", "System"),
            content=state.get("extracted_text", "")
        )
    
    # QUIZ MODE: If options exist, append them to the message for visual guide (Markdown list)
    if options and isinstance(options, list):
        friendly_msg += "\n\n**Please choose or correct based on these possibilities:**\n"
        for opt in options:
            friendly_msg += f"- {opt}\n"

    trace.append({"agent": "HITL", "action": f"paused pipeline — reason: {state.get('hitl_reason', '')[:80]}", "timestamp": str(datetime.datetime.now())})
    
    duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"[DEBUG] 🚨 HITL Agent finished in {duration:.2f}s")
    
    return {
        **state, 
        "hitl_response": {
            "friendly_message": friendly_msg, 
            "status": "waiting",
            "options": options
        },
        "agent_trace": trace
    }
