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

def get_finalizer_system_prompt(registry: Optional[Any] = None) -> str:
    """Generate dynamic finalizer system prompt containing all registered specialist capabilities and tools."""
    try:
        from app.agents.registry import default_registry
        reg = registry or default_registry
        registry_summary = reg.to_prompt_summary()
        agent_names = [f"'{a.agent_id}'" for a in reg.list_agents()]
        tool_names = []
        for a in reg.list_agents():
            for t in a.tools:
                tool_names.append(f"'{t.tool_name}'")
        agents_str = ", ".join(sorted(agent_names))
        tools_str = ", ".join(sorted(tool_names))
    except Exception:
        registry_summary = ""
        agents_str = "registered agents"
        tools_str = "registered tools"

    return f"""\
You are the Finalizer agent in an adaptive multi-agent system.

Your job is to synthesize all available context into a clear, comprehensive, and accurate final answer to the original task.

{registry_summary}

You are provided with:
- The original user task.
- Context and outputs from upstream specialist agents and external tools (if invoked).
- Conversation context & thread history from prior dialogue turns (if available).

Guidelines:
1. Answer the user's task directly, accurately, and thoroughly.
2. Ground your response in the provided tool and agent outputs. If tool results (such as live date/time, search results, or calculations) are present in the supporting information, integrate those factual results into your answer.
3. Maintain conversational continuity across multi-turn interactions. If prior conversation history includes the user's name, previous preferences, questions, or context, directly incorporate and acknowledge it to personalize your answer.
4. If the user asks about available tools, system capabilities, or what this system can do:
   - Accurately describe the multi-agent system and its currently registered tools ({tools_str}) and specialist agents ({agents_str}) dynamically present in the registry.
   - Note: Structured schema formatters (such as FinalizerOutput or ResearcherOutput) are internal response data models, not tools.
5. If no external tools were invoked, answer using your comprehensive knowledge without inventing false citations or claiming external tools were used.
6. Provide clear, well-structured explanations with actionable key points.

Output a single JSON object matching the schema. Do not include any extra text, explanations, or commentary outside the JSON object.
"""

SYSTEM_PROMPT = get_finalizer_system_prompt()


# ---------------------------------------------------------------------------
# Finalizer
# ---------------------------------------------------------------------------

@dataclass
class Finalizer:
    """Finalizer agent backed by an NVIDIA NIM LLM.

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
            FinalizerOutput,
            method="function_calling",
            include_raw=False,
        )

        return cls(model=llm, structured_llm=structured_llm)

    def finalize(
        self,
        original_task: str,
        supporting_info: str,
        conversation_history: Optional[str] = None,
    ) -> FinalizerOutput:
        """Synthesize *supporting_info* into a final answer for *original_task*.

        Args:
            original_task: The original task or question.
            supporting_info: Relevant outputs/context from other agents.
            conversation_history: Optional multi-turn dialog history.

        Returns:
            FinalizerOutput with the final answer, key points, and limitations.
        """
        import time

        parts = []
        if conversation_history:
            parts.append(f"Conversation Context & Thread History:\n{conversation_history}")
        parts.append(f"Original task:\n{original_task}")
        if supporting_info:
            parts.append(f"Supporting information from other agents:\n{supporting_info}")

        user_content = "\n\n".join(parts)
        sys_prompt = get_finalizer_system_prompt()
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
                    # Retry without conversation history
                    res = self.structured_llm.invoke([
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": f"Original task:\n{original_task}\n\nSupporting information from other agents:\n{supporting_info}"},
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

        # Fallback if structured output produced None
        ans = supporting_info if supporting_info else f"Completed response for: {original_task}"
        return FinalizerOutput(
            final_answer=ans,
            key_points=[ans[:200]],
            limitations=[],
        )


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
