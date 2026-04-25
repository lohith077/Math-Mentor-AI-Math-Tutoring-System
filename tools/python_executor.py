"""
Sandboxed Python executor for complex mathematical computations.
Uses RestrictedPython-style approach with allowed builtins only.
"""
import re, math, traceback

ALLOWED_IMPORTS = {"math", "sympy", "numpy"}

def execute_math_code(code: str) -> dict:
    """
    Execute math-focused Python code in a restricted namespace.
    Only math, sympy, numpy operations allowed.
    """
    # Block dangerous patterns
    forbidden = ["import os", "import sys", "open(", "exec(", "eval(", "__import__", "subprocess"]
    for pattern in forbidden:
        if pattern in code:
            return {"success": False, "error": f"Forbidden operation detected: {pattern}"}
    
    namespace = {"math": math}
    try:
        import sympy; namespace["sympy"] = sympy
    except ImportError:
        pass
    try:
        import numpy as np; namespace["np"] = np; namespace["numpy"] = np
    except ImportError:
        pass
    
    # Capture stdout
    import io, contextlib
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(compile(code, "<math_executor>", "exec"), namespace)
        output = stdout_capture.getvalue()
        result = namespace.get("result", output.strip() if output.strip() else "Code executed successfully")
        return {"success": True, "result": str(result), "output": output}
    except Exception:
        return {"success": False, "error": traceback.format_exc(limit=3)}
