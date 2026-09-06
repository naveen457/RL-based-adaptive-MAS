from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config.settings import settings


# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------

class CoderOutput(BaseModel):
    """Structured coding result for a programming task."""

    approach: str = Field(
        description="High-level description of the approach taken to solve the task.",
    )
    code: str = Field(
        description="The implementation code. Provide a complete, runnable solution where practical.",
    )
    explanation: str = Field(
        description="Explanation of how the code works and why the approach was chosen.",
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Assumptions made while writing the code (e.g. input constraints, environment).",
    )
    testing_notes: List[str] = Field(
        default_factory=list,
        description="Suggestions for testing the code, including edge cases and example usage.",
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the Coder agent in an adaptive multi-agent system.

Your job is to write code that solves the given programming task and return a
structured result.

Important constraints:
- Do NOT execute, run, or test the generated code yourself.
- The code field must contain the actual implementation (not just a description).
- Provide a complete, runnable solution where practical.
- Clearly state any assumptions about inputs, environment, or dependencies.
- Give concrete testing notes, including edge cases and example usage.

Output a single JSON object matching the schema. Do not include any extra text,
explanations, or commentary outside the JSON object.
"""


# ---------------------------------------------------------------------------
# Coder
# ---------------------------------------------------------------------------

@dataclass
class Coder:
    """Coder agent backed by an OpenRouter LLM.

    Stateless and reusable, suitable for later use as a LangGraph node.
    Does NOT execute generated code.
    """

    model: ChatOpenAI
    structured_llm: ChatOpenAI

    @classmethod
    def from_settings(
        cls,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Coder:
        """Construct a Coder from settings (or explicit overrides for testing)."""
        resolved_model = model if model is not None else settings.model
        resolved_base_url = base_url if base_url is not None else settings.base_url

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
            CoderOutput,
            method="function_calling",
            include_raw=False,
        )

        return cls(model=llm, structured_llm=structured_llm)

    def code(self, task: str) -> CoderOutput:
        """Write code to solve *task* and return a structured result.

        Args:
            task: The programming task description.

        Returns:
            CoderOutput with implementation, explanation, and testing notes.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        return self.structured_llm.invoke(messages)


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def create_code_solution(task: str) -> CoderOutput:
    """One-shot helper that constructs a Coder and solves *task*."""
    coder = Coder.from_settings()
    return coder.code(task)
