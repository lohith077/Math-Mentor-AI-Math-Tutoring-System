"""
④ Verifier Agent
Type: Non-ReAct | Model: GPT-OSS-120B
Checks correctness, domain/unit constraints, edge cases. Assigns confidence 0-100.
"""
import json, re, datetime
from groq import Groq
from langsmith import traceable
from agents.state import MathMentorState
from agents.groq_utils import groq_completion_with_fallback
from config.settings import GROQ_API_KEY, REASONING_MODEL, REASONING_MODEL_BACKUPS, VERIFIER_CONFIDENCE_THRESHOLD

client = Groq(api_key=GROQ_API_KEY)

VERIFIER_SYSTEM = """You are a math solution verifier. Your goal is to be a "Devils Advocate" and try to prove the solution wrong.

### STEP-BY-STEP REASONING (Implicit CoT)
For every verification, you must follow this internal logic:
1.  **Plug-back**: Take the final answer (e.g., $x=2$) and substitute it back into the original problem. Does it hold?
2.  **Constraint Check**: Check domain (log(x) where x>0, denominators != 0, sqrt(x) where x>=0).
3.  **Path Audit**: Read the steps taken. Is there a logical leap or sign error?
4.  **Unit/Dimension**: Are the units correct for the physical context?

### FORMAT
Output ONLY a JSON object in this exact format:
{
  "thought_process": "Your step-by-step internal reasoning here (CoT)",
  "is_correct": true,
  "confidence": 0-100,
  "issues": ["List of logical errors found"],
  "notes": "Brief summary",
  "plug_back_check": "Example of substitution: 2(4)+3(2)-5=9 != 0"
}

### SCORING
- **Confidence 80-100**: PASS. Verified by substitution; no logical leaps.
- **Confidence 75-80**: Plausible Hit / Possible Issues. (TRIGGER HITL)
- **Confidence < 75**: TRIGGER HITL. Critical errors, logical gaps, or missing info.
"""

@traceable(run_type="chain", name="Verifier Agent")
def verifier_agent(state: MathMentorState) -> MathMentorState:
    print(f"[DEBUG] ✅ Verifier Agent starting...")
    start_time = datetime.datetime.now()
    trace = state.get("agent_trace", [])
    problem = state.get("parsed_problem", {})
    
    # Check if we already have a verification for this specific solution
    existing_verification = state.get("verification_result", {})
    if existing_verification and state.get("solution"):
        print(f"[DEBUG] ✅ Verifier Agent: Using existing verification.")
        verification = existing_verification
    else:
        # Use fallback wrapper
        response, actual_model = groq_completion_with_fallback(
            client=client,
            model=REASONING_MODEL,
            backups=REASONING_MODEL_BACKUPS,
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
    is_approved = state.get("hitl_response", {}).get("status") == "approved"
    hitl_needed = confidence < VERIFIER_CONFIDENCE_THRESHOLD
    
    if is_approved and hitl_needed:
        print("[DEBUG] ✅ Verifier Agent: Bypassing confidence check because user manually approved.")
        hitl_needed = False
        verification["is_correct"] = True # Force success for routing
    
    trace.append({"agent": "Verifier", "action": f"confidence={confidence}, is_correct={verification.get('is_correct')}, model={actual_model if 'actual_model' in locals() else 'Existing'}", "timestamp": str(datetime.datetime.now())})
    
    duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"[DEBUG] ✅ Verifier Agent finished in {duration:.2f}s")
    
    # User Optimization: Cache Fallback Reset
    # If this was a memory hit but it failed verification, we MUST clear the 
    # solution and steps so the Helper Agent knows to solve it from scratch.
    final_solution = state.get("solution", "")
    final_steps = state.get("solution_steps", [])
    if state.get("memory_hit") and not verification.get("is_correct"):
        print(f"[DEBUG] ✅ Verifier: Clearing incorrect cache solution for fallback.")
        final_solution = ""
        final_steps = []

    return {
        **state,
        "verification_result": verification,
        "solution": final_solution,
        "solution_steps": final_steps,
        "hitl_required": hitl_needed or state.get("hitl_required", False),
        "hitl_reason": f"Verifier confidence {confidence}% below threshold {VERIFIER_CONFIDENCE_THRESHOLD}%" if hitl_needed else state.get("hitl_reason", ""),
        "hitl_friendly_message": verification.get("notes", "I'm having a little trouble being 100% sure about this solution. Would you like to review the steps or try rephrasing?") if hitl_needed else state.get("hitl_friendly_message", ""),
        "hitl_options": ["Show solution anyway", "Try re-solving"] if hitl_needed else state.get("hitl_options", []),
        "hitl_trigger_agent": "Verifier" if hitl_needed else state.get("hitl_trigger_agent", ""),
        "agent_trace": trace
    }
