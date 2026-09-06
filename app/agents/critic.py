from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from app.config.settings import settings


# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------

class CriticOutput(BaseModel):
    """Structured review of an output against a task or context."""

    overall_assessment: str = Field(
        description="Overall assessment of the output: what is good, what is wrong, and how serious the issues are.",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Specific problems found in the output (correctness, logic, quality, unsupported claims, etc.).",
    )
    missing_requirements: List[str] = Field(
        default_factory=list,
        description="Requirements from the original task/context that the output failed to address.",
    )
    corrections: List[str] = Field(
        default_factory=list,
        description="Concrete corrections or suggested fixes for the issues found.",
    )
    verification_status: str = Field(
        description="One of: 'correct', 'partially_correct', 'incorrect', 'unclear'. "
        "A concise verdict on the correctness of the output.",
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the Critic agent in an adaptive multi-agent system.

Your job is to critically review a piece of output against an original task or
context and return a structured review.

You are given:
- The original task or context.
- The output that needs reviewing.

Evaluate the output for:
- Correctness: is the information factually accurate?
- Completeness: does it address all parts of the original task?
- Logical problems: contradictions, non-sequiturs, flawed reasoning.
- Unsupported claims: statements made without evidence or justification.
- Quality issues: clarity, structure, tone, missing detail.

Be specific and useful. Do not just say "good" or "bad". For each issue,
describe what is wrong and, where possible, what the correction should be.

For verification_status use exactly one of: 'correct', 'partially_correct',
'incorrect', 'unclear'.

Output a single JSON object matching the schema. Do not include any extra text,
explanations, or commentary outside the JSON object.
"""


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

@dataclass
class Critic:
    """Critic agent backed by an OpenRouter LLM.

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
    ) -> Critic:
        """Construct a Critic from settings (or explicit overrides for testing)."""
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
            CriticOutput,
            method="json_schema",
            include_raw=False,
        )

        return cls(model=llm, structured_llm=structured_llm)

    def review(
        self,
        original_task: str,
        output_to_review: str,
    ) -> CriticOutput:
        """Review *output_to_review* against *original_task*.

        Args:
            original_task: The original task, question, or context.
            output_to_review: The output that should be reviewed.

        Returns:
            CriticOutput with assessment, issues, and corrections.
        """
        user_content = (
            f"Original task / context:\n{original_task}\n\n"
            f"Output to review:\n{output_to_review}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self.structured_llm.invoke(messages)


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def create_critic_review(
    original_task: str,
    output_to_review: str,
) -> CriticOutput:
    """One-shot helper that constructs a Critic and reviews *output_to_review*."""
    critic = Critic.from_settings()
    return critic.review(original_task=original_task, output_to_review=output_to_review)
