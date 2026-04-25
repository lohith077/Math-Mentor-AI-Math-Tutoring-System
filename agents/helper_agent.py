"""
Ⓑ Helper Agent — Sequential Sub-Agent #2 (receives context from RAG Agent)
Type: ReAct | Model: Qwen-3-32B | Tools: sympy_solver, calculator, python_executor
Iterative reason → tool → observe loop to compute the solution.
"""
import json, datetime
from groq import Groq
from langsmith import traceable
from agents.state import MathMentorState
from agents.groq_utils import groq_completion_with_fallback
from tools.sympy_solver import solve_equation, differentiate, integrate, compute_limit, matrix_operations
from tools.calculator import safe_calculate
from tools.python_executor import execute_math_code
from config.settings import GROQ_API_KEY, MATH_MODEL, MATH_MODEL_BACKUPS

client = Groq(api_key=GROQ_API_KEY)

TOOLS = {
    "sympy_solve": solve_equation,
    "sympy_diff": differentiate,
    "sympy_integrate": integrate,
    "sympy_limit": compute_limit,
    "sympy_matrix": matrix_operations,
    "calculator": safe_calculate,
    "python_exec": execute_math_code,
}

HELPER_SYSTEM = """You are a Senior Mathematics Professor specialized in JEE Advanced. Your goal is to provide perfectly accurate, step-by-step mathematical solutions.

### TOOL BELT (Output JSON for tool calls):
- sympy_solve: For exact algebraic solutions. Input: {"equation_str": "...", "variable": "x"}
- sympy_diff: For symbolic differentiation. Input: {"expr_str": "...", "variable": "x", "order": 1}
- sympy_integrate: For exact integration. Input: {"expr_str": "...", "variable": "x", "lower": null, "upper": null}
- sympy_limit: For symbolic limits. Input: {"expr_str": "...", "variable": "x", "point": "0"}
- calculator: For high-precision arithmetic only. Input: {"expression": "..."}
- python_exec: For complex logic, custom algorithms, or numerical simulations. Input: {"code": "..."}

### OPERATIONAL PROTOCOL:
1. **Analytic Phase**: First, present a brief "Plan" based on the problem and the provided RAG knowledge.
2. **Execution Phase**: Use tools sequentially. Do not guess complex calculations.
3. **Consistency**: Use LaTeX (e.g., $x^2$) for ALL mathematical expressions.
4. **Finality**: Once the result is derived, provide a comprehensive summary.

### OUTPUT FORMAT:
- Tool Call: `TOOL_CALL: {"tool": "tool_name", "input": {...}}`
- Conclusion: `FINAL_ANSWER: [Step-by-step summary + LaTeX Final Result]`

Always prioritize standard identities and shortcuts provided in the RAG Context to ensure JEE-style elegance."""

def parse_tool_call(text: str):
    if "TOOL_CALL:" in text:
        try:
            json_str = text.split("TOOL_CALL:")[-1].strip()
            return json.loads(json_str)
        except Exception:
            return None
    return None

@traceable(run_type="chain", name="Helper Agent")
def helper_agent(state: MathMentorState) -> MathMentorState:
    print(f"[DEBUG] 🤖 Helper Agent starting ReAct loop...")
    start_time = datetime.datetime.now()
    trace = state.get("agent_trace", [])
    problem = state.get("parsed_problem", {})
    context_chunks = state.get("retrieved_context", [])
    
    context_text = "\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in context_chunks[:5]
    ])
    
    user_prompt = f"""Problem: {problem.get('problem_text', state.get('extracted_text', ''))}
Topic: {problem.get('topic', '')}
Variables: {problem.get('variables', [])}
Constraints: {problem.get('constraints', [])}

Relevant Knowledge Base Context:
{context_text}

Solve this problem step by step using the tools available."""

    messages = [
        {"role": "system", "content": HELPER_SYSTEM},
        {"role": "user", "content": user_prompt}
    ]
    
    solution_steps = state.get("solution_steps", [])
    tools_used = state.get("tools_used", [])
    max_iterations = 8
    final_solution = state.get("solution", "")
    
    # If we already have a solution, skip the ReAct loop
    # (Unless we want to force re-run, but for performance, skip is better on resume)
    if final_solution and solution_steps:
        print(f"[DEBUG] 🤖 Helper Agent: Using existing solution.")
    else:
        for iteration in range(max_iterations):
            # Use fallback wrapper
            response, actual_model = groq_completion_with_fallback(
                client=client,
                model=MATH_MODEL,
                backups=MATH_MODEL_BACKUPS,
                messages=messages,
                temperature=0.1,
                max_tokens=1500
            )
            
            reply = response.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})
            solution_steps.append(f"[Step {iteration+1}] {reply[:300]}...")
            
            tool_call = parse_tool_call(reply)
            if tool_call:
                tool_name = tool_call.get("tool")
                tool_input = tool_call.get("input", {})
                tool_fn = TOOLS.get(tool_name)
                
                if tool_fn:
                    try:
                        result = tool_fn(**tool_input)
                    except TypeError:
                        # Try calling with positional arg if kwargs fail
                        result = {"success": False, "error": "Invalid tool arguments"}
                    
                    tools_used.append(tool_name)
                    tool_result_msg = f"Tool result for {tool_name}: {json.dumps(result)}"
                    messages.append({"role": "user", "content": tool_result_msg})
                    trace.append({"agent": "HelperAgent", "action": f"called {tool_name} → {result.get('result', result.get('error', ''))}", "timestamp": str(datetime.datetime.now())})
                else:
                    messages.append({"role": "user", "content": f"Tool '{tool_name}' not found. Available: {list(TOOLS.keys())}"})
            
            elif "FINAL_ANSWER:" in reply:
                final_solution = reply.split("FINAL_ANSWER:")[-1].strip()
                break
            
            else:
                # If no tool call and no FINAL_ANSWER tag, assume the reply is the answer
                final_solution = reply
                break
    
    # Final check: if final_solution is still empty but we have steps, summarize them
    if not final_solution and solution_steps:
        final_solution = f"Solved in {len(solution_steps)} steps. Final result derived in the trace."
    
    trace.append({"agent": "HelperAgent", "action": f"solved in {len(solution_steps)} steps, tools={tools_used}, model={actual_model if 'actual_model' in locals() else 'Existing'}", "timestamp": str(datetime.datetime.now())})
    
    duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"[DEBUG] 🤖 Helper Agent finished in {duration:.2f}s ({len(solution_steps)} iterations)")
    print(f"[RAW HELPER RESPONSE] {final_solution[:500]}...")
    
    return {
        **state,
        "solution": final_solution,
        "solution_steps": solution_steps,
        "tools_used": list(set(tools_used)),
        "agent_trace": trace
    }
