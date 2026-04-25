"""
② Parser Agent
Type: Non-ReAct | Model: Qwen-3-32B (Groq)
"""
import json, re, datetime
from groq import Groq
from langsmith import traceable
from agents.state import MathMentorState
from agents.groq_utils import groq_completion_with_fallback
from config.settings import GROQ_API_KEY, FAST_MODEL, FAST_MODEL_BACKUPS

client = Groq(api_key=GROQ_API_KEY)

PARSER_SYSTEM = """You are an expert Math Problem Parser with a tutor persona. Your goal is to convert messy raw text into a structured JSON, and if the input is unclear, suggest ways to fix it like Claude or GPT would.

### FORMAT
Output ONLY a JSON object:
{
  "problem_text": "A clean, LaTeX-formatted version",
  "topic": "algebra | probability | calculus | linear_algebra",
  "variables": ["x", "y"],
  "constraints": [],
  "semantic_confidence": 0.0 to 1.0,
  "needs_clarification": true/false,
  "clarification_reason": "Technical reason for failure",
  "friendly_clarification": "A supportive, conversational message for the student (e.g., 'I see a quadratic here, but is the constant term 5 or 6?')",
  "quiz_options": ["Option A", "Option B"] or null
}

### RULES
1. **Multi-Modal Cleaning**:
   - **OCR Fixes**: 'V'/'v' → 'sqrt', 'fe'/'S' → 'integral', etc.
   - **ASR (Speech) Fixes**: "power of" → '^', "pi" → 'π', etc.
2. **Interactive Clarification**:
   - If a core element is missing, set `needs_clarification: true`.
   - **Friendly Clarification**: Write a message that sounds like a helpful tutor, not an error message.
   - **Quiz Mode**: If there are 2-3 likely interpretations of the messy input, list them in `quiz_options`.
3. **JEE Standards**: Use standard symbols and LaTeX for all math.

### EXAMPLES
Input: "find the root of x^2 - 4"
Output: {
  "problem_text": "Find the roots of $x^2 - 4 = 0$",
  "topic": "algebra",
  "variables": ["x"],
  "constraints": [],
  "semantic_confidence": 0.9,
  "needs_clarification": false,
  "clarification_reason": "",
  "friendly_clarification": "",
  "quiz_options": null
}

Input: "derive sin ?" 
Output: {
  "problem_text": "Differentiate $\sin(?)$",
  "topic": "calculus",
  "variables": ["x"],
  "constraints": [],
  "semantic_confidence": 0.4,
  "needs_clarification": true,
  "clarification_reason": "Missing variable/argument for sine function",
  "friendly_clarification": "I see you want to differentiate a sine function, but I'm not sure what the variable is. Did you mean sin(x) or sin(theta)?",
  "quiz_options": ["Differentiate sin(x)", "Differentiate sin(theta)"]
}"""

@traceable(run_type="chain", name="Parser Agent")
def parser_agent(state: MathMentorState) -> MathMentorState:
    if state.get("hitl_required"):
        return state  # Pause for HITL first
    
    print(f"[DEBUG] 🔍 Parser Agent starting...")
    start_time = datetime.datetime.now()
    trace = state.get("agent_trace", [])
    
    # Check if we already have a parsed problem for this exact text
    existing_parsed = state.get("parsed_problem", {})
    if existing_parsed and existing_parsed.get("problem_text") == state.get("extracted_text"):
        print(f"[DEBUG] 🔍 Parser Agent: Using existing parsed problem.")
        parsed = existing_parsed
    else:
        trace.append({"agent": "Parser", "action": "start", "timestamp": str(datetime.datetime.now())})
        
        # Use fallback wrapper
        response, actual_model = groq_completion_with_fallback(
            client=client,
            model=FAST_MODEL,
            backups=FAST_MODEL_BACKUPS,
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
            parsed = {
                "problem_text": state["extracted_text"], 
                "topic": "unknown",
                "variables": [], 
                "constraints": [], 
                "needs_clarification": True,
                "clarification_reason": "I had trouble understanding the mathematical structure of this problem.",
                "friendly_clarification": "I've captured the text, but I'm having a little trouble parsing the mathematical symbols and structure. Could you please double-check if there are any typos or missing terms in the text below?",
                "quiz_options": []
            }
    
    # Logic to bypass ambiguity check if user already approved/corrected in HITL
    is_approved = state.get("hitl_response", {}).get("status") == "approved"
    
    hitl_required = parsed.get("needs_clarification", False)
    
    if is_approved and hitl_required:
        print("[DEBUG] 🔍 Parser Agent: Bypassing ambiguity check because user manually approved/edited.")
        hitl_required = False
        parsed["needs_clarification"] = False # Update inside dict for consistency

    trace.append({"agent": "Parser", "action": f"parsed → topic={parsed.get('topic')}, model={actual_model if 'actual_model' in locals() else 'Existing'}, needs_clarification={hitl_required} (Approved: {is_approved})", "timestamp": str(datetime.datetime.now())})
    
    duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"[DEBUG] 🔍 Parser Agent finished in {duration:.2f}s")
    
    return {
        **state,
        "parsed_problem": parsed,
        "topic": parsed.get("topic", "unknown"),
        "needs_clarification": hitl_required,
        "semantic_confidence": parsed.get("semantic_confidence", 0.5),
        "hitl_required": hitl_required or state.get("hitl_required", False),
        "hitl_reason": parsed.get("clarification_reason", "") if hitl_required else state.get("hitl_reason", ""),
        "hitl_friendly_message": parsed.get("friendly_clarification", "") if hitl_required else state.get("hitl_friendly_message", ""),
        "hitl_options": parsed.get("quiz_options", []) if hitl_required else state.get("hitl_options", []),
        "hitl_trigger_agent": "Parser" if hitl_required else state.get("hitl_trigger_agent", ""),
        "agent_trace": trace
    }
