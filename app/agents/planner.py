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
specialist agents and tools can execute. Do not execute any task steps yourself — only produce the structured plan.

{registry_summary}

PLANNING & ROUTING GUIDELINES:
1. EXTERNAL TOOLS & CAPABILITIES:
   - Tool Execution (`tool_executor`):
     * Set `requires_tools: true` and include `"tool_executor"` in `selected_agents` whenever the task needs external live data, real-time factual lookups, current calendar/clock time, or calculation.
     * Populate `tools_needed` with the specific tool name(s) from the registry:
       - `"get_current_date"`: For current date, time, weekday, month, or year queries.
       - `"calculator"`: For arithmetic, mathematical calculations, or numerical evaluations.
       - `"web_search"`: For live web search, current news, recent developments, real-world factual information, or external search retrieval.
     * Include `"tool_use"` and specific tool names (e.g. `"web_search"`) in `required_capabilities`.

2. SPECIALIST COGNITIVE AGENTS:
   - `"researcher"`: Set `requires_research: true` and include `"researcher"` in `selected_agents` when the task requires in-depth conceptual synthesis, background context analysis, or structured report writing.
   - `"coder"`: Set `requires_coding: true` and include `"coder"` in `selected_agents` when the task involves writing, explaining, modifying, or debugging programming code.
   - `"critic"`: Set `requires_verification: true` and include `"critic"` in `selected_agents` when the task benefits from rigorous verification, code review, or quality critique.
   - `"finalizer"`: Always included in `selected_agents` to synthesize the final polished response.

3. CONVERSATIONAL & CHIT-CHAT INPUTS:
   - For basic conversational greetings or acknowledgments (e.g. "hi", "hello", "thank you"):
     Set all specialized capability flags to false (`requires_research: false`, `requires_coding: false`, `requires_verification: false`, `requires_tools: false`) and `selected_agents: ["planner", "finalizer"]`.

4. CONVERSATIONAL CONTINUITY & CONTEXT:
   - If prior conversation context is provided, consider previous turns (user identity, past questions, referenced tools or code) when planning and decomposing the current task.

5. OUTPUT REQUIREMENTS:
   - List the exact minimal agent IDs needed in `selected_agents`.
   - Provide concise, ordered imperative steps and an estimated complexity.
   - Output a single JSON object conforming to the schema without any markdown formatting or commentary outside the JSON.
"""

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
                return self.structured_llm.invoke(messages)
            except Exception as e:
                err_str = str(e).lower()
                if ("rate_limit" in err_str or "429" in err_str or "tokens per minute" in err_str) and attempt < 2:
                    print("  [Rate Limit] Replenishing tokens, waiting 5s...")
                    time.sleep(5)
                    continue
                raise


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
