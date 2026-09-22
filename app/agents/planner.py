from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config.settings import settings


# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------

class PlannerOutput(BaseModel):
    """Structured planning result for a user task."""

    task_understanding: str = Field(
        description="Concise restatement of what the user is asking for and the goal."
    )
    required_capabilities: List[str] = Field(
        default_factory=list,
        description="Capabilities needed to complete the task (e.g. research, coding, math, data analysis).",
    )
    steps: List[str] = Field(
        default_factory=list,
        description="Ordered steps to complete the task. Each step is a short imperative sentence.",
    )
    requires_research: bool = Field(
        default=False,
        description="Whether the task requires in-depth conceptual synthesis, background context analysis, or structured report writing (do NOT set true for live web search queries).",
    )
    requires_coding: bool = Field(
        default=False,
        description="Whether the task requires writing or modifying code.",
    )
    requires_verification: bool = Field(
        default=False,
        description="Whether the result should be verified/critiqued before finalising.",
    )
    requires_tools: bool = Field(
        default=False,
        description="Whether the task requires live external tools (e.g. web_search for real-time/current/trending events, or calculator for math).",
    )
    tools_needed: List[str] = Field(
        default_factory=list,
        description="List of specific tool names needed from tool_executor (e.g. ['web_search'], ['calculator']).",
    )
    selected_agents: List[str] = Field(
        default_factory=list,
        description="List of agent IDs selected from the registry to execute this plan.",
    )
    estimated_complexity: str = Field(
        default="medium",
        description="Rough complexity estimate: trivial | low | medium | high | unknown.",
    )


# ---------------------------------------------------------------------------
# Dynamic System Prompt Generator
# ---------------------------------------------------------------------------

def get_planner_system_prompt(registry: Optional[Any] = None) -> str:
    """Generate dynamic planner system prompt derived completely from the dynamic registry using loops."""
    try:
        from app.agents.registry import default_registry
        reg = registry or default_registry
    except Exception:
        reg = None

    lines = [
        "You are the Planner agent in an adaptive multi-agent system.",
        "Your role is to decompose the user's task into a minimal, precise, and executable plan",
        "that downstream specialist agents and tools can execute. Do not execute any task steps yourself.",
        "",
        "### Available System Capabilities and Registered Nodes:",
    ]

    cognitive_agents = []
    tool_nodes = []

    if reg is not None:
        for agent in reg.list_agents():
            if agent.agent_id == "planner":
                continue
            if agent.node_type == "tool_executor" or bool(agent.tools):
                tool_nodes.append(agent)
            else:
                cognitive_agents.append(agent)

    if tool_nodes:
        lines.append("\n#### Tool-Calling Nodes & Registered External Tools:")
        for node in tool_nodes:
            lines.append(f"- **{node.agent_id}** ({node.role}): {node.description}")
            if node.capabilities:
                lines.append(f"  * Node capabilities: {', '.join(node.capabilities)}")
            if node.tools:
                lines.append("  * Available Tools on this node:")
                for tool in node.tools:
                    param_str = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
                    lines.append(f"    - `{tool.tool_name}({param_str})`: {tool.description}")

    if cognitive_agents:
        lines.append("\n#### Specialist Cognitive Agents:")
        for agent in cognitive_agents:
            lines.append(f"- **{agent.agent_id}** ({agent.role}): {agent.description}")
            if agent.capabilities:
                lines.append(f"  * Capabilities provided: {', '.join(agent.capabilities)}")

    lines.extend([
        "",
        "### Dynamic Planning Directives:",
        "1. Minimal Sufficient Composition (Parsimony):",
        "   - Select ONLY the agents whose roles or capabilities are strictly needed for the task in `selected_agents`.",
        "   - Do not include specialist agents whose capabilities are irrelevant to prevent over-engineering waste.",
        "2. Multi-Tool & Synergistic Composition:",
        "   - Review all registered tools listed above.",
        "   - When a task benefits from multiple tools (e.g. establishing context, querying data, performing computation),",
        "     list ALL relevant tools in `tools_needed`, set `requires_tools: true`, and include the host tool node in `selected_agents`.",
        "   - Tools execute sequentially with accumulated context passed forward.",
        "3. Conversational & Context Continuity:",
        "   - If prior conversation history is provided, incorporate context from previous turns.",
        "   - For simple conversational greetings or acknowledgments without task work, keep the plan minimal without specialist nodes.",
        "4. Output Requirements:",
        "   - Output a single JSON object conforming to the schema without any markdown formatting or text outside the JSON.",
    ])
    return "\n".join(lines)


SYSTEM_PROMPT = get_planner_system_prompt()



# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

@dataclass
class Planner:
    """Planner agent backed by an NVIDIA NIM LLM.

    Designed to be stateless and reusable, so it can later be wrapped as a
    LangGraph node without modification.
    """

    model: ChatOpenAI
    structured_llm: ChatOpenAI

    @classmethod
    def from_settings(
        cls,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Planner:
        """Construct a Planner from settings (or explicit overrides for testing).

        When *api_key* is explicitly passed as ``None`` (not an empty string),
        the check falls back to the configured settings so missing-config tests
        can target the settings path directly.
        """
        resolved_model = model if model is not None else settings.model
        resolved_base_url = base_url if base_url is not None else settings.base_url

        # Explicit None means 'use settings'; an empty string or real key is used as-is.
        if api_key is None:
            resolved_api_key = settings.api_key
        else:
            resolved_api_key = api_key

        if not resolved_api_key:
            raise ValueError(
                "NVIDIA API key is not configured. "
                "Set NVIDIA_API_KEY in your environment or .env file."
            )
        if not resolved_model:
            raise ValueError(
                "NVIDIA_MODEL is not configured. "
                "Set NVIDIA_MODEL in your environment or .env file."
            )

        llm = ChatOpenAI(
            model=resolved_model,
            openai_api_key=resolved_api_key,
            openai_api_base=resolved_base_url,
            temperature=0.0,
            max_tokens=settings.max_tokens,
        )

        structured_llm = llm.with_structured_output(
            PlannerOutput,
            method="function_calling",
            include_raw=False,
        )

        return cls(model=llm, structured_llm=structured_llm)

    def plan(
        self,
        task: str,
        conversation_history: Optional[str] = None,
    ) -> PlannerOutput:
        """ Decompose *task* into a structured plan.

        Args:
            task: The user's natural-language task description.
            conversation_history: Optional multi-turn conversation history.


        Returns:
            PlannerOutput with the decomposition.
        """
        import time

        parts = []
        if conversation_history:
            parts.append(f"Conversation Context & Thread History:\n{conversation_history}")
        parts.append(f"Current task to plan:\n{task}")

        user_content = "\n\n".join(parts)
        sys_prompt = get_planner_system_prompt()
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        for attempt in range(3):
            try:
                res = self.structured_llm.invoke(messages)
                if res is not None:
                    return res
                if conversation_history:
                    # Retry with only the task, stripping conversation history
                    res = self.structured_llm.invoke([
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": f"Current task to plan:\n{task}"},
                    ])
                    if res is not None:
                        return res
            except Exception as e:
                err_str = str(e).lower()
                if ("rate_limit" in err_str or "429" in err_str or "tokens per minute" in err_str) and attempt < 2:
                    print("  [Rate Limit] Replenishing tokens, waiting 5s...")
                    time.sleep(5)
                    continue
                if attempt == 2:
                    break

        # Guaranteed non-None fallback if model produced None or failed structured decoding
        task_lower = task.lower()
        is_tool = any(w in task_lower for w in ["search", "find", "who", "population", "calculate", "date", "time", "difference"])
        is_code = any(w in task_lower for w in ["code", "python", "function", "program", "script"])
        is_verify = any(w in task_lower for w in ["verify", "check", "critique", "validate", "review"])
        tools = []
        if any(w in task_lower for w in ["search", "population", "find", "who"]):
            tools.append("web_search")
        if any(w in task_lower for w in ["calculate", "difference", "sum", "math"]):
            tools.append("calculator")
        if any(w in task_lower for w in ["date", "time", "current"]):
            tools.append("get_current_date")

        req_caps = ["planning"]
        if is_tool:
            req_caps.extend(["tool_use", "web_search"])
        if is_code:
            req_caps.append("coding")
        if is_verify:
            req_caps.append("verification")

        return PlannerOutput(
            task_understanding=task,
            required_capabilities=list(dict.fromkeys(req_caps)),
            steps=[f"Process and answer: {task}"],
            requires_research=False,
            requires_coding=is_code,
            requires_verification=is_verify,
            requires_tools=is_tool,
            tools_needed=tools,
            selected_agents=["tool_executor"] if is_tool else [],
            estimated_complexity="medium",
        )


# ---------------------------------------------------------------------------
# Convenience helper for manual / scripted use
# ---------------------------------------------------------------------------

def create_plan(task: str) -> PlannerOutput:
    """One-shot helper that constructs a Planner and plans *task*.

    Intended for scripts and manual invocation. For repeated use, construct a
    ``Planner`` once and call ``planner.plan(...)``.
    """
    planner = Planner.from_settings()
    return planner.plan(task)
