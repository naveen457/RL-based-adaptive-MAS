from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config.settings import settings


# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------

class ResearcherOutput(BaseModel):
    """Structured research result for a given task or question."""

    research_question: str = Field(
        description="The research question or task being addressed, restated clearly.",
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Key findings or information relevant to the research question.",
    )
    sources_or_evidence: List[str] = Field(
        default_factory=list,
        description="Sources, evidence, or cited information that supports the findings. "
        "When no external search tool is available, note that the information is drawn "
        "from general knowledge or supplied context, not from a live web search.",
    )
    uncertainties: List[str] = Field(
        default_factory=list,
        description="Things that are uncertain, unverified, or would benefit from further research.",
    )
    requires_further_research: bool = Field(
        default=False,
        description="Whether the topic clearly needs additional research beyond what is provided here.",
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the Researcher agent in an adaptive multi-agent system.

Your job is to research the given task or question and return a structured report.

Important constraints:
- You do NOT have access to a live web-search tool.
- Do NOT claim that you searched the internet or fetched live pages.
- Where information comes from your training knowledge or from supplied context,
  say so clearly in the sources_or_evidence field.
- Distinguish well-established facts from speculation.
- List any uncertainties or areas that genuinely require further research.

Output a single JSON object matching the schema. Do not include any extra text,
explanations, or commentary outside the JSON object.
"""


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------

@dataclass
class Researcher:
    """Researcher agent backed by an OpenRouter LLM.

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
    ) -> Researcher:
        """Construct a Researcher from settings (or explicit overrides for testing)."""
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
            ResearcherOutput,
            method="function_calling",
            include_raw=False,
        )

        return cls(model=llm, structured_llm=structured_llm)

    def research(self, task: str) -> ResearcherOutput:
        """Research *task* and return a structured report.

        Args:
            task: The research question or task description.

        Returns:
            ResearcherOutput with findings and evidence.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        return self.structured_llm.invoke(messages)


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def create_research_report(task: str) -> ResearcherOutput:
    """One-shot helper that constructs a Researcher and researches *task*."""
    researcher = Researcher.from_settings()
    return researcher.research(task)
