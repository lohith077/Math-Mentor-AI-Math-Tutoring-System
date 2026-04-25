"""
⑥ Guardrail Agent — Final gate before showing answer
Type: LLM-Based | Model: Llama-3.1-8b-instant
Enforces: domain scope, accuracy, safety, and helpfulness.
"""
import json, datetime, re
from groq import Groq
from langsmith import traceable
from agents.state import MathMentorState
from agents.groq_utils import groq_completion_with_fallback
from config.settings import GROQ_API_KEY, FAST_MODEL, FAST_MODEL_BACKUPS, ALLOWED_TOPICS

client = Groq(api_key=GROQ_API_KEY)

GUARDRAIL_SYSTEM = f"""You are the 'Quality & Safety Controller' for Math Mentor. 
Your job is to audit the final explanation before the student sees it.

### Audit Criteria:
1. **Scope**: Is it math-related and within JEE topics ({', '.join(ALLOWED_TOPICS)})?
2. **Safety**: No harmful, illegal, or irrelevant content.
3. **Clarity**: Is the explanation concise and correct?
4. **LaTeX**: Is the math properly formatted ($..$ or $$..$$)?

### Interaction Style:
If you find an issue, don't just say 'failed'. Provide:
- A clear reason.
- A 'Conversational Clarification': A friendly prompt for the user (like GPT/Claude).
- 'Quiz Options' (Optional): If the problem is ambiguous, provide 2-3 likely options to help the user clarify quickly.

Output EXACTLY this JSON format:
{{
  "passed": true/false,
  "issues": ["..."],
  "friendly_clarification": "Supportive message asking for more info...",
  "quiz_options": ["Option A", "Option B"] or null
}}"""

@traceable(run_type="chain", name="Guardrail Agent")
def guardrail_agent(state: MathMentorState) -> MathMentorState:
    if state.get("guardrail_passed"):
        return state
    
    print(f"[DEBUG] 🛡️ Guardrail Agent starting (LLM)...")
    start_time = datetime.datetime.now()
    trace = state.get("agent_trace", [])
    
    explanation = state.get("explanation", "")
    solution = state.get("solution", "")
    topic = state.get("topic", "")
    
    # Use fallback wrapper
    response, actual_model = groq_completion_with_fallback(
        client=client,
        model=FAST_MODEL,
        backups=FAST_MODEL_BACKUPS,
        messages=[
            {"role": "system", "content": GUARDRAIL_SYSTEM},
            {"role": "user", "content": f"Topic: {topic}\nSolution: {solution}\nExplanation: {explanation}"}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
    except:
        result = {"passed": False, "issues": ["Guardrail parse error"], "friendly_clarification": "I'm having trouble verifying this. Can you try rephrasing?"}
    
    passed = result.get("passed", False)
    is_approved = state.get("hitl_response", {}).get("status") == "approved"
    
    if is_approved and not passed:
        print("[DEBUG] 🛡️ Guardrail Agent: Bypassing quality check because user manually approved.")
        passed = True
        result["passed"] = True

    trace.append({"agent": "Guardrail", "action": f"{'PASSED' if passed else 'FAILED'}: {result.get('issues')} (Approved: {is_approved}, Model: {actual_model if 'actual_model' in locals() else 'Existing'})", "timestamp": str(datetime.datetime.now())})
    
    # Store the smart clarification for the HITL agent to use
    new_state = {
        **state, 
        "guardrail_passed": passed, 
        "guardrail_issues": result.get("issues", []),
        "hitl_required": not passed or state.get("hitl_required", False),
        "hitl_reason": "; ".join(result.get("issues", [])) if not passed else state.get("hitl_reason", ""),
        "hitl_trigger_agent": "Guardrail" if not passed else state.get("hitl_trigger_agent", ""),
        "agent_trace": trace,
        # Pass the smart suggestions forward
        "hitl_options": result.get("quiz_options")
    }
    
    # Pre-populate the friendly message so HITL agent has it
    if not passed:
        new_state["hitl_friendly_message"] = result.get("friendly_clarification", "I've detected some quality issues with the explanation. Would you like me to try again or review the current version?")
        new_state["hitl_options"] = result.get("quiz_options", ["Try again", "Show anyway"])

    duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"[DEBUG] 🛡️ Guardrail Agent finished in {duration:.2f}s")
    return new_state
