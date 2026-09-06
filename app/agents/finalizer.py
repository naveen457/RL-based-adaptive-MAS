from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config.settings import settings


# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------

class FinalizerOutput(BaseModel):
    """Structured final answer synthesizing information from other agents."""

    final_answer: str = Field(
        description="The final synthesized answer to the original task.",
    )
    key_points: List[str] = Field(
        default_factory=list,
        description="The key points or takeaways from the final answer.",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Limitations, caveats, or things the final answer does not cover.",
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the Finalizer agent in an adaptive multi-agent system.

Your job is to synthesize the supplied information into a clear, final answer to
the original task.

You are given:
- The original task.
- Relevant outputs or context from other agents (research findings, code,
  critique, etc.).

Rules:
- Synthesize the supplied information; do not invent new research findings.
- Do NOT claim that tools, searches, or other agents were used when they were
  not actually provided in the context.
- If the supplied information is insufficient, say so in the limitations.
- Produce a final answer that a user could read directly.

Output a single JSON object matching the schema. Do not include any extra text,
explanations, or commentary outside the JSON object.
"""


# ---------------------------------------------------------------------------
# Finalizer
# ---------------------------------------------------------------------------

@dataclass
class Finalizer:
    """Finalizer agent backed by an OpenRouter LLM.

    Stateless and reusable, suitable for later use as a LangGraph node.
    """

    model: ChatOpenAI
    structured_llm: ChatOpenAI

    @classmethod
    def from_settings(
        cls,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Finalizer:
        """Construct a Finalizer from settings (or explicit overrides for testing)."""
        resolved_model = model if model is not None else settings.openrouter_model
        resolved_base_url = base_url if base_url is not None else settings.openrouter_base_url

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
            FinalizerOutput,
            method="function_calling",
            include_raw=False,
        )

        return cls(model=llm, structured_llm=structured_llm)

    def finalize(
        self,
        original_task: str,
        supporting_info: str,
    ) -> FinalizerOutput:
        """Synthesize *supporting_info* into a final answer for *original_task*.

        Args:
            original_task: The original task or question.
            supporting_info: Relevant outputs/context from other agents.

        Returns:
            FinalizerOutput with the final answer, key points, and limitations.
        """
        user_content = (
            f"Original task:\n{original_task}\n\n"
            f"Supporting information from other agents:\n{supporting_info}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self.structured_llm.invoke(messages)


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def create_final_answer(
    original_task: str,
    supporting_info: str,
) -> FinalizerOutput:
    """One-shot helper that constructs a Finalizer and produces a final answer."""
    finalizer = Finalizer.from_settings()
    return finalizer.finalize(original_task=original_task, supporting_info=supporting_info)
