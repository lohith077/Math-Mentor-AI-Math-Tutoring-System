"""
⑦ Followup Agent — Conversations after the solution
Type: LLM-Based | Model: Qwen-3-32B
Handles clarifications, logic deep-dives, and suggests next steps.
"""
import json, datetime
from groq import Groq
from langsmith import traceable
from agents.state import MathMentorState
from agents.groq_utils import groq_completion_with_fallback
from config.settings import GROQ_API_KEY, FAST_MODEL, FAST_MODEL_BACKUPS
from memory.similarity_search import find_similar_problems

client = Groq(api_key=GROQ_API_KEY)

FOLLOWUP_SYSTEM = """You are an expert Math Mentor. The student has just received a solution and may have follow-up questions.

### YOUR TASKS:
1. **Answer**: Be encouraging, patient, and precise. Use LaTeX for math.
2. **Context**: Use the previous problem, solution, and explanation to provide specific answers.
3. **Suggestions**: ALWAYS provide exactly 5 suggested follow-up questions at the end of your response.

### SUGGESTION CATEGORIES (Prioritize these):
- Deep dive into formulas/identities used.
- More detailed, step-by-step explanation.
- Logical flow overview.
- "Try a similar problem" (Include this if relevant).
- JEE shortcuts or common pitfalls.

### FORMAT:
Output ONLY a JSON object:
{
  "answer": "Your friendly, LaTeX-heavy explanation...",
  "suggestions": ["Question 1", "Question 2", "Question 3", "Question 4", "Question 5"],
  "action_required": "none" | "find_similar"
}"""

@traceable(run_type="chain", name="Followup Agent")
def followup_agent(state: MathMentorState) -> MathMentorState:
    print(f"[DEBUG] 🎓 Followup Agent starting...")
    start_time = datetime.datetime.now()
    trace = state.get("agent_trace", [])
    
    query = state.get("followup_query", "")
    history = state.get("chat_history", [])
    
    # Special Trigger: Similar Problem Search
    if "similar" in query.lower():
        print(f"[DEBUG] 🎓 Followup Agent: Triggering Similarity Search.")
        matches = find_similar_problems(state.get("extracted_text", ""), top_k=2)
        if matches:
            ans = "I found some similar problems in our archive! Check these out:\n\n"
            for m in matches:
                ans += f"**Problem**: {m['problem_text']}\n**Solution Snippet**: {m['solution'][:200]}...\n\n"
            ans += "Would you like me to solve one of these for you?"
            suggestions = ["Solve the first one", "Solve the second one", "Explain the logic flow of this one", "What formulas are common here?", "Back to original problem"]
            return {
                **state,
                "chat_history": history + [("user", query), ("assistant", ans)],
                "followup_suggestions": suggestions,
                "agent_trace": trace + [{"agent": "Followup", "action": "Retrieved similar problems", "timestamp": str(datetime.datetime.now())}]
            }

    # Standard Conversation
    context = f"""
    Original Problem: {state.get('extracted_text')}
    Solved Result: {state.get('solution')}
    Detailed Explanation: {state.get('explanation')}
    """
    
    messages = [
        {"role": "system", "content": FOLLOWUP_SYSTEM},
        {"role": "user", "content": f"Context: {context}\n\nStudent Question: {query}"}
    ]
    
    # Use fallback wrapper
    response, actual_model = groq_completion_with_fallback(
        client=client,
        model=FAST_MODEL,
        backups=FAST_MODEL_BACKUPS,
        messages=messages,
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
    except:
        result = {
            "answer": "I'm sorry, I had trouble processing that. Can you rephrase?",
            "suggestions": ["Explain the last solution again", "Show formulas used", "Logical flow", "Try similar", " JEE Tips"],
            "action_required": "none"
        }
    
    new_history = history + [("user", query), ("assistant", result.get("answer", ""))]
    
    trace.append({"agent": "Followup", "action": f"answered: {query[:50]}... using model={actual_model if 'actual_model' in locals() else 'Existing'}", "timestamp": str(datetime.datetime.now())})
    
    return {
        **state,
        "chat_history": new_history,
        "followup_suggestions": result.get("suggestions", []),
        "agent_trace": trace
    }
