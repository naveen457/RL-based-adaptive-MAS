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
    """Generate dynamic planner system prompt containing all currently registered nodes and tools."""
    try:
        from app.agents.registry import default_registry
        reg = registry or default_registry
        registry_summary = reg.to_prompt_summary()
    except Exception:
        registry_summary = ""

    return f"""\
You are the Planner agent in an adaptive multi-agent system.

Your job is to decompose the user's task into a clear, executable plan that downstream
agent nodes can follow. Do not execute any steps yourself — only produce the plan.

{registry_summary}

ROUTING AND TOOL SELECTION RULES:
1. DISTINGUISHING TOOL_EXECUTOR VS RESEARCHER & TOOL SELECTION:
   - "web_search":
     * If the task asks for "current trends", "current events", "trending topics", "latest updates", "recent news", "live info", or real-time facts:
       - You MUST set requires_tools: true.
       - You MUST include "tool_executor" in selected_agents.
       - You MUST include "web_search" in tools_needed.
       - Include "web_search" and "tool_use" in required_capabilities.
     * CRITICAL RULE: Queries about "current trends", "current events", or "latest news" do NOT require the calendar date. Do NOT include "get_current_date" for these queries.
   - "get_current_date":
     * ONLY use when the task explicitly asks what calendar date, day of the week, month, year, or clock time it is right now (e.g. "what is today's date?", "what time is it?", "tell me today's date").
     * Set requires_tools: true, selected_agents: ["planner", "tool_executor", "finalizer"], tools_needed: ["get_current_date"].
   - "calculator":
     * ONLY use if the task requires mathematical or arithmetic calculations.
     * Set requires_tools: true, include "tool_executor" in selected_agents, include "calculator" in tools_needed.

   - Only set requires_research: true if deep conceptual synthesis or in-depth document writing is needed in addition to or instead of live search.
2. If the task requires writing or modifying code: set requires_coding: true and include "coder" in selected_agents.
3. If the task requires quality critique/verification: set requires_verification: true and include "critic" in selected_agents.
4. For simple greetings or conversational replies: requires_research: false, requires_coding: false, requires_verification: false, requires_tools: false, and selected_agents: ["planner", "finalizer"].
5. Always list the exact agent IDs needed in selected_agents.
6. Provide ordered imperative steps and complexity estimate.

Output a single JSON object matching the schema. Do not include any extra text,
explanations, or commentary outside the JSON object.
"""

SYSTEM_PROMPT = get_planner_system_prompt()



# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

@dataclass
class Planner:
    """Planner agent backed by an OpenRouter LLM.

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
                "OpenRouter API key is not configured. "
                "Set OPENROUTER_API_KEY in your environment or .env file."
            )
        if not resolved_model:
            raise ValueError(
                "OPENROUTER_MODEL is not configured. "
                "Set OPENROUTER_MODEL in your environment or .env file."
            )

        llm = ChatOpenAI(
            model=resolved_model,
            openai_api_key=resolved_api_key,
            openai_api_base=resolved_base_url,
            temperature=0.0,
        )

        structured_llm = llm.with_structured_output(
            PlannerOutput,
            method="function_calling",
            include_raw=False,
        )

        return cls(model=llm, structured_llm=structured_llm)

    def plan(self, task: str) -> PlannerOutput:
        """ Decompose *task* into a structured plan.

        Args:
            task: The user's natural-language task description.

        Returns:
            PlannerOutput with the decomposition.
        """
        sys_prompt = get_planner_system_prompt()
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": task},
        ]
        return self.structured_llm.invoke(messages)


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
