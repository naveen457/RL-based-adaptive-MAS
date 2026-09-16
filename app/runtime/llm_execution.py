"""Execution provider that runs the existing configured LLM-backed agents."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol

from pydantic import BaseModel, Field

from app.agents.coder import Coder
from app.agents.critic import Critic
from app.agents.finalizer import Finalizer
from app.agents.planner import PlannerOutput
from app.agents.researcher import Researcher
from app.architecture.models import MASArchitecture


class AgentExecutionRecord(BaseModel):
    """One observed output from an existing runtime agent."""

    agent_id: str
    invoked: bool
    status: str
    output: Any = None


class LLMExecutionResult(BaseModel):
    """Outputs produced by the selected existing LLM-backed agents."""

    execution_trace: list[AgentExecutionRecord] = Field(default_factory=list)
    final_response: Any = None


class AgentFactory(Protocol):
    def __call__(self) -> Any: ...


class ExistingLLMAgentExecutor:
    """Run existing agent implementations according to adapted architecture."""

    def __init__(
        self,
        *,
        researcher_factory: AgentFactory = Researcher.from_settings,
        coder_factory: AgentFactory = Coder.from_settings,
        critic_factory: AgentFactory = Critic.from_settings,
        finalizer_factory: AgentFactory = Finalizer.from_settings,
    ) -> None:
        self.researcher_factory = researcher_factory
        self.coder_factory = coder_factory
        self.critic_factory = critic_factory
        self.finalizer_factory = finalizer_factory

    def execute(
        self,
        task: str,
        planner_output: PlannerOutput,
        architecture: MASArchitecture,
    ) -> LLMExecutionResult:
        """Execute only active agents required by the PlannerOutput."""

        active_agents = architecture.active_agent_ids
        trace = [
            AgentExecutionRecord(
                agent_id="planner",
                invoked=True,
                status="completed",
                output=planner_output.model_dump(mode="json"),
            )
        ]
        supporting_outputs: Dict[str, Any] = {}

        research_output = None
        if planner_output.requires_research and "researcher" in active_agents:
            research_output = self.researcher_factory().research(task)
            supporting_outputs["researcher"] = research_output
            trace.append(self._completed("researcher", research_output))
        else:
            trace.append(self._not_invoked("researcher"))

        coder_output = None
        if planner_output.requires_coding and "coder" in active_agents:
            coder_output = self.coder_factory().code(task)
            supporting_outputs["coder"] = coder_output
            trace.append(self._completed("coder", coder_output))
        else:
            trace.append(self._not_invoked("coder"))

        critic_output = None
        if planner_output.requires_verification and "critic" in active_agents:
            review_context = self._supporting_text(supporting_outputs)
            critic_output = self.critic_factory().review(
                original_task=task,
                output_to_review=review_context,
            )
            supporting_outputs["critic"] = critic_output
            trace.append(self._completed("critic", critic_output))
        else:
            trace.append(self._not_invoked("critic"))

        final_response = None
        if "finalizer" in active_agents:
            finalizer_context = self._supporting_text(supporting_outputs)
            final_response = self.finalizer_factory().finalize(
                original_task=task,
                supporting_info=finalizer_context,
            )
            trace.append(self._completed("finalizer", final_response))
        else:
            trace.append(self._not_invoked("finalizer"))

        return LLMExecutionResult(
            execution_trace=trace,
            final_response=self._serialize(final_response),
        )

    def node_handlers(
        self,
        task: str,
        planner_output: PlannerOutput,
        event_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
        """Create LangGraph node handlers backed by the existing agent classes."""

        def event(name: str, agent_id: str) -> None:
            if event_callback is not None:
                event_callback(name, agent_id)

        def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
            event("started", "planner")
            event("completed", "planner")
            return {"planner_output": planner_output}

        def researcher_node(state: Dict[str, Any]) -> Dict[str, Any]:
            event("started", "researcher")
            output = self.researcher_factory().research(task)
            event("completed", "researcher")
            return {"research_output": output}

        def coder_node(state: Dict[str, Any]) -> Dict[str, Any]:
            event("started", "coder")
            output = self.coder_factory().code(task)
            event("completed", "coder")
            return {"coder_output": output}

        def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
            event("started", "critic")
            outputs = {}
            if state.get("research_output") is not None:
                outputs["researcher"] = state["research_output"]
            if state.get("coder_output") is not None:
                outputs["coder"] = state["coder_output"]
            output = self.critic_factory().review(
                original_task=task,
                output_to_review=self._supporting_text(outputs),
            )
            event("completed", "critic")
            return {"critic_output": output}

        def finalizer_node(state: Dict[str, Any]) -> Dict[str, Any]:
            event("started", "finalizer")
            outputs = {}
            for key, agent_id in (
                ("research_output", "researcher"),
                ("coder_output", "coder"),
                ("critic_output", "critic"),
            ):
                if state.get(key) is not None:
                    outputs[agent_id] = state[key]
            output = self.finalizer_factory().finalize(
                original_task=task,
                supporting_info=self._supporting_text(outputs),
            )
            event("completed", "finalizer")
            return {"final_answer": output}

        return {
            "planner": planner_node,
            "researcher": researcher_node,
            "coder": coder_node,
            "critic": critic_node,
            "finalizer": finalizer_node,
        }

    @staticmethod
    def _completed(agent_id: str, output: Any) -> AgentExecutionRecord:
        return AgentExecutionRecord(
            agent_id=agent_id,
            invoked=True,
            status="completed",
            output=ExistingLLMAgentExecutor._serialize(output),
        )

    @staticmethod
    def _not_invoked(agent_id: str) -> AgentExecutionRecord:
        return AgentExecutionRecord(
            agent_id=agent_id,
            invoked=False,
            status="not_invoked",
            output=None,
        )

    @staticmethod
    def _serialize(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        return value

    @classmethod
    def _supporting_text(cls, outputs: Dict[str, Any]) -> str:
        parts = []
        for agent_id, output in outputs.items():
            serialized = cls._serialize(output)
            parts.append(f"{agent_id} output:\n{serialized}")
        return "\n\n".join(parts) if parts else "(no intermediate outputs)"
