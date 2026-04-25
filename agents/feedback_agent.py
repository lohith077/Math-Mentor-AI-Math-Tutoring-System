"""
⑦ Feedback Agent
Type: Non-ReAct | Model: Groq Compound Mini
Analyzes user feedback to categorize sentiment and identify actionable knowledge gaps.
"""
import json, datetime
from groq import Groq
from langsmith import traceable
from agents.groq_utils import groq_completion_with_fallback
from config.settings import GROQ_API_KEY, MINI_MODEL, MINI_MODEL_BACKUPS

client = Groq(api_key=GROQ_API_KEY)

FEEDBACK_SYSTEM = """You are a 'Math Mentor Feedback Analyst'. 
Your goal is to analyze user feedback on a solved math problem.

### Categories:
- **Technical**: Errors in the math, calculation, or final answer.
- **Explanation**: The math is right, but the explanation is confusing or dry.
- **Scope**: The system solved it, but user wanted something else.
- **Stylistic**: Formatting or tone preferences.
- **General**: Generic "thanks" or "not helpful".

### Sentiment:
- Positive, Neutral, Negative.

### Actionable:
- Is this something a human can fix by adding a new RAG document or updating the solver? (true/false)

Output ONLY JSON:
{
  "category": "Technical | Explanation | Scope | Stylistic | General",
  "sentiment": "Positive | Neutral | Negative",
  "is_actionable": true/false,
  "lesson_learned": "Short note for the developers on how to improve."
}"""

@traceable(run_type="chain", name="Feedback Agent")
def feedback_agent(comment: str, problem_text: str, solution: str) -> dict:
    """Analyze a user comment in the context of the problem solved."""
    print(f"[DEBUG] 📊 Feedback Agent analyzing...")
    
    # Use fallback wrapper
    response, actual_model = groq_completion_with_fallback(
        client=client,
        model=MINI_MODEL,
        backups=MINI_MODEL_BACKUPS,
        messages=[
            {"role": "system", "content": FEEDBACK_SYSTEM},
            {"role": "user", "content": f"Problem: {problem_text}\nSolution: {solution}\nUser Feedback: {comment}"}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {
            "category": "General",
            "sentiment": "Neutral",
            "is_actionable": False,
            "lesson_learned": "Could not parse feedback analysis."
        }
