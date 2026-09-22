"""Dedicated Tool-Calling Node and Tool Implementations.

This module provides a dedicated node for executing tools (e.g., web search).
Additional tools can be added to this same node in the future without creating
separate graph nodes for each tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

import httpx


# ---------------------------------------------------------------------------
# Tool Models & Results
# ---------------------------------------------------------------------------

class ToolCallRequest(BaseModel):
    """Structured request to call a tool."""

    tool_name: str = Field(description="Name of the tool to execute.")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool.")


class ToolExecutionOutput(BaseModel):
    """Structured output from tool execution."""

    tool_name: str
    status: str = "success"  # "success" | "error" | "skipped"
    result: Any = None
    error: Optional[str] = None

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


# ---------------------------------------------------------------------------
# Tool Imports (from app.tools)
# ---------------------------------------------------------------------------

from app.tools import (
    arxiv_search,
    basic_calculator,
    duckduckgo_html_search,
    get_current_date,
    tavily_search,
    web_search,
)



# ---------------------------------------------------------------------------
# Dedicated Tool Executor Node
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Dedicated node responsible for executing external tools.
    
    Acts as the single execution gateway for tool calls (web search, calculator,
    arXiv paper search, code execution, etc.), keeping cognitive reasoning agents
    decoupled from direct tool execution mechanics.
    """

    def __init__(self, tools: Optional[Dict[str, Callable[..., Any]]] = None) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {
            "web_search": web_search,
            "calculator": basic_calculator,
            "get_current_date": get_current_date,
            "arxiv_search": arxiv_search,
        }
        if tools:
            self._tools.update(tools)



    @property
    def registered_tools(self) -> List[str]:
        """Names of all tools currently registered on this node."""
        return sorted(list(self._tools.keys()))

    def register_tool(self, name: str, func: Callable[..., Any]) -> None:
        """Add a new tool to this tool executor node."""
        self._tools[name] = func

    def execute(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> ToolExecutionOutput:
        """Execute a specific tool with the provided arguments."""
        if tool_name not in self._tools:
            return ToolExecutionOutput(
                tool_name=tool_name,
                status="error",
                error=f"Tool '{tool_name}' is not registered on this node. Available tools: {self.registered_tools}"
            )

        args = arguments or {}
        try:
            res = self._tools[tool_name](**args)
            return ToolExecutionOutput(
                tool_name=tool_name,
                status="success",
                result=res
            )
        except Exception as exc:
            return ToolExecutionOutput(
                tool_name=tool_name,
                status="error",
                error=f"Tool execution failed: {type(exc).__name__}: {exc}"
            )

    def execute_calls(self, tool_calls: List[Dict[str, Any]]) -> List[ToolExecutionOutput]:
        """Execute a list of tool call dictionaries."""
        outputs = []
        for call in tool_calls:
            name = call.get("tool_name") or call.get("name", "")
            args = call.get("arguments") or call.get("args", {})
            outputs.append(self.execute(name, args))
        return outputs
