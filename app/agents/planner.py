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
        description="Whether the task requires gathering external information before acting.",
    )
    requires_coding: bool = Field(
        default=False,
        description="Whether the task requires writing or modifying code.",
    )
    requires_verification: bool = Field(
        default=False,
        description="Whether the result should be verified/critiqued before finalising.",
    )
    estimated_complexity: str = Field(
        default="medium",
        description="Rough complexity estimate: trivial | low | medium | high | unknown.",
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the Planner agent in an adaptive multi-agent system.

Your job is to decompose the user's task into a clear, executable plan that later
agent nodes can follow. Do not execute any steps yourself — only produce the plan.

Think carefully about:
- What the user actually wants (restate it concisely).
- Which capabilities are needed (research, coding, math, data analysis, etc.).
- The ordered sequence of steps to complete the task.
- Whether the task requires external research, code, or verification/criticism.
- A rough complexity estimate (trivial, low, medium, high, or unknown).

Output a single JSON object matching the schema. Do not include any extra text,
explanations, or commentary outside the JSON object.
"""


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
        resolved_model = model if model is not None else settings.openrouter_model
        resolved_base_url = base_url if base_url is not None else settings.openrouter_base_url

        # Explicit None means 'use settings'; an empty string or real key is used as-is.
        if api_key is None:
            resolved_api_key = settings.openrouter_api_key
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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
