"""
Symbolic math solver using SymPy.
Supports: solve equations, differentiate, integrate, compute limits, matrix ops.
"""
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

def solve_equation(equation_str: str, variable: str = "x") -> dict:
    """Solve equation like '2*x**2 + 3*x - 5 = 0' for variable."""
    try:
        var = sp.Symbol(variable)
        if "=" in equation_str:
            lhs, rhs = equation_str.split("=", 1)
            eq = sp.Eq(parse_expr(lhs.strip()), parse_expr(rhs.strip()))
        else:
            eq = parse_expr(equation_str.strip())
        solutions = sp.solve(eq, var)
        return {"success": True, "solutions": [str(s) for s in solutions], "latex": [sp.latex(s) for s in solutions]}
    except Exception as e:
        return {"success": False, "error": str(e)}

def differentiate(expr_str: str, variable: str = "x", order: int = 1) -> dict:
    try:
        var = sp.Symbol(variable)
        expr = parse_expr(expr_str)
        result = sp.diff(expr, var, order)
        return {"success": True, "result": str(result), "latex": sp.latex(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def integrate(expr_str: str, variable: str = "x", lower=None, upper=None) -> dict:
    try:
        var = sp.Symbol(variable)
        expr = parse_expr(expr_str)
        if lower is not None and upper is not None:
            result = sp.integrate(expr, (var, lower, upper))
        else:
            result = sp.integrate(expr, var)
        return {"success": True, "result": str(result), "latex": sp.latex(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def compute_limit(expr_str: str, variable: str = "x", point: str = "0", direction: str = "+") -> dict:
    try:
        var = sp.Symbol(variable)
        expr = parse_expr(expr_str)
        pt = parse_expr(point)
        result = sp.limit(expr, var, pt, direction)
        return {"success": True, "result": str(result), "latex": sp.latex(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def matrix_operations(matrix_str: str, operation: str = "det") -> dict:
    """Supported operations: det, inverse, eigenvalues, rank"""
    try:
        mat = sp.Matrix(eval(matrix_str))
        if operation == "det":
            result = mat.det()
        elif operation == "inverse":
            result = mat.inv()
        elif operation == "eigenvalues":
            result = mat.eigenvals()
        elif operation == "rank":
            result = mat.rank()
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}
        return {"success": True, "result": str(result), "latex": sp.latex(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}
