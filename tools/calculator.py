"""Safe numeric calculator for arithmetic evaluation."""
import math, re

SAFE_NAMES = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
SAFE_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow})

def safe_calculate(expression: str) -> dict:
    """Evaluate a math expression safely (no exec/eval of arbitrary code)."""
    # Remove any potentially dangerous patterns
    if re.search(r"(import|exec|eval|open|os\.|sys\.|__)", expression):
        return {"success": False, "error": "Expression contains disallowed operations"}
    try:
        result = eval(expression, {"__builtins__": {}}, SAFE_NAMES)
        return {"success": True, "result": float(result) if isinstance(result, (int, float)) else str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}
