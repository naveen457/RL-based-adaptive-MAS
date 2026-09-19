from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    quality_score: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Quality and completeness score from 0.0 to 1.0 assessing if the output adequately satisfies the user task requirements.",
    )
    retry_target_node: Optional[str] = Field(
        default=None,
        description="Target node to redirect to if quality_score is below threshold: 'tool_executor', 'coder', or 'researcher'.",
    )
    suggested_tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Specific follow-up tool calls if redirecting to tool_executor, e.g. [{'tool_name': 'web_search', 'arguments': {'query': '...'}}].",
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

Scoring and Redirection:
- quality_score: Rate the overall output from 0.0 to 1.0.
  * Score >= 0.75: The output is sufficiently accurate and complete to proceed to final answer synthesis without looping.
  * Score < 0.75: The output has critical flaws, missing data, or unmet task criteria that require an upstream node to retry.
- If quality_score < 0.75:
  * Specify retry_target_node as one of: 'tool_executor', 'coder', 'researcher'.
  * If redirecting to 'tool_executor', provide the exact tool name and arguments in suggested_tool_calls (e.g. [{"tool_name": "web_search", "arguments": {"query": "..."}}]).

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
    """Critic agent backed by an NVIDIA NIM LLM.

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
        resolved_model = model if model is not None else settings.model
        resolved_base_url = base_url if base_url is not None else settings.base_url

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
            CriticOutput,
            method="function_calling",
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
        import time

        user_content = (
            f"Original task / context:\n{original_task}\n\n"
            f"Output to review:\n{output_to_review}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
# Convenience helper
# ---------------------------------------------------------------------------

def create_critic_review(
    original_task: str,
    output_to_review: str,
) -> CriticOutput:
    """One-shot helper that constructs a Critic and reviews *output_to_review*."""
    critic = Critic.from_settings()
    return critic.review(original_task=original_task, output_to_review=output_to_review)
