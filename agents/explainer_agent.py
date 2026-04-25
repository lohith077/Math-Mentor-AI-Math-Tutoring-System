"""
⑤ Explainer Agent
Type: Non-ReAct | Model: Qwen-3-32B
Produces student-friendly LaTeX-formatted explanation with RAG citations.
"""
import datetime
from groq import Groq
from langsmith import traceable
from agents.state import MathMentorState
from agents.groq_utils import groq_completion_with_fallback
from config.settings import GROQ_API_KEY, MATH_MODEL, MATH_MODEL_BACKUPS

client = Groq(api_key=GROQ_API_KEY)

EXPLAINER_SYSTEM = """You are a Master JEE Mathematics Tutor. Your goal is to provide conceptual depth and exam-solving speed.

### Structure of your response:
1. **Conceptual Approach**: Start with a 1-2 sentence high-level strategy (e.g., "We will use the Graphical method here to avoid complex calculation").
2. **Step-by-Step Solution**:
   - Use numbered steps.
   - Use LaTeX for ALL math (wrap in $...$ for inline, $$...$$ for blocks).
   - Explain the "WHY": If you use a formula, explain why it applies here.
   - **Citations**: Reference the RAG context using "[Source: Topic]" if relevant.
3. **💡 JEE Exam Tip / Shortcut**: A specific tip on how to solve this faster or a property that simplifies the calculation (e.g., "Property of symmetric matrices allows us to...").
4. **⚠️ Common Trap**: Warn the student about a specific mistake they might make (e.g., "Don't forget to check if the denominator becomes zero at x=2").
5. **Key Takeaway**: A final conceptual summary.

### Tone:
Professional, highly logical, and encouraging. Focus on "Conceptual Clarity" over "Rote Calculation"."""

@traceable(run_type="chain", name="Explainer Agent")
def explainer_agent(state: MathMentorState) -> MathMentorState:
    print(f"[DEBUG] 📝 Explainer Agent starting...")
    start_time = datetime.datetime.now()
    trace = state.get("agent_trace", [])
    problem = state.get("parsed_problem", {})
    context = state.get("retrieved_context", [])
    
    sources_used = list(set([c["source"] for c in context]))
    
    # Check if we already have an explanation
    if state.get("explanation") and state.get("solution"):
        print(f"[DEBUG] 📝 Explainer Agent: Using existing explanation.")
        explanation = state["explanation"]
    else:
        # Use fallback wrapper
        response, actual_model = groq_completion_with_fallback(
            client=client,
            model=MATH_MODEL,
            backups=MATH_MODEL_BACKUPS,
            messages=[
                {"role": "system", "content": EXPLAINER_SYSTEM},
                {"role": "user", "content": f"""Problem: {problem.get('problem_text', '')}
Solution: {state.get('solution', '')}
Sources available: {sources_used}

Produce a clear, step-by-step explanation for a student."""}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        explanation = response.choices[0].message.content.strip()
    trace.append({"agent": "Explainer", "action": f"solution explained using model={actual_model if 'actual_model' in locals() else 'Existing'}", "timestamp": str(datetime.datetime.now())})
    
    duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"[DEBUG] 📝 Explainer Agent finished in {duration:.2f}s")
    
    return {**state, "explanation": explanation, "agent_trace": trace}
