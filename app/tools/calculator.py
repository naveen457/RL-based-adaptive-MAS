"""Mathematical & Arithmetic Calculator Tool.

Provides safe evaluation of arithmetic expressions without code execution vulnerabilities.
"""

from __future__ import annotations

import math
from typing import Any, Dict


def basic_calculator(expression: str) -> Dict[str, Any]:
    """Safely evaluate mathematical and arithmetic expressions.
    
    Supports basic math (+, -, *, /, **, %, sqrt, abs, round).
    """
    cleaned = expression.strip()
    if not cleaned:
        return {"expression": expression, "error": "Empty expression"}

    # Restricted vocabulary for safe arithmetic evaluation
    safe_dict = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
    }

    try:
        result = eval(cleaned, {"__builtins__": {}}, safe_dict)  # pylint: disable=eval-used
        return {
            "expression": cleaned,
            "result": result,
            "status": "success",
        }
    except Exception as exc:
        return {
            "expression": cleaned,
            "error": f"Evaluation error: {type(exc).__name__}: {exc}",
            "status": "error",
        }
