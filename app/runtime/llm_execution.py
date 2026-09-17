"""Execution provider that runs the existing configured LLM-backed agents."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional, Protocol

from pydantic import BaseModel, Field

from app.agents.coder import Coder
from app.agents.critic import Critic
from app.agents.finalizer import Finalizer
from app.agents.planner import PlannerOutput
from app.agents.researcher import Researcher
from app.agents.tool_executor import ToolExecutor
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
        tool_executor_factory: Optional[Callable[[], ToolExecutor]] = None,
    ) -> None:
        self.researcher_factory = researcher_factory
        self.coder_factory = coder_factory
        self.critic_factory = critic_factory
        self.finalizer_factory = finalizer_factory
        self.tool_executor_factory = tool_executor_factory or (lambda: ToolExecutor())

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

        # 1. Execute tools first if tool need is detected
        requires_tools = getattr(planner_output, "requires_tools", False)
        tools_needed = list(getattr(planner_output, "tools_needed", []) or [])
        selected_agents = list(getattr(planner_output, "selected_agents", []) or [])
        req_caps = set(planner_output.required_capabilities or [])
        query_lower = task.lower()
        temporal_keywords = ["current", "trending", "latest", "today", "live", "recent", "events in"]
        query_needs_search = any(k in query_lower for k in temporal_keywords)

        has_tool_need = (
            "tool_executor" in active_agents
            or "tool_executor" in selected_agents
            or requires_tools
            or query_needs_search
            or bool(tools_needed)
            or bool(req_caps & {"tool_use", "web_search", "external_api", "tools"})
        )

        tool_outputs = {}
        if has_tool_need:
            executor = self.tool_executor_factory()
            # Determine which tools to invoke
            if not tools_needed:
                if query_needs_search or requires_tools or ("web_search" in req_caps):
                    tools_needed.append("web_search")

            # Check for explicit date/time query
            explicit_date_keywords = [
                "today's date", "what is today", "what is the date",
                "what's the date", "what day is it", "today date",
                "current time", "what time is it", "what day is today"
            ]
            is_explicit_date_query = any(k in query_lower for k in explicit_date_keywords)
            if is_explicit_date_query and "get_current_date" not in tools_needed:
                tools_needed.append("get_current_date")

            # If web_search is needed and the user did NOT explicitly ask for today's date/time,
            # prune get_current_date to prevent an unnecessary second loop
            if not is_explicit_date_query and "web_search" in tools_needed:
                tools_needed = [t for t in tools_needed if t not in {"get_current_date", "date", "time"}]

            for t_name in tools_needed:
                if t_name in {"get_current_date", "date", "time"}:
                    res = executor.execute("get_current_date", {})
                    tool_outputs["get_current_date"] = res.serialize()
                elif t_name == "calculator":
                    res = executor.execute("calculator", {"expression": task})
                    tool_outputs["calculator"] = res.serialize()
                elif t_name == "web_search":
                    res = executor.execute("web_search", {"query": task})
                    tool_outputs["web_search"] = res.serialize()
                elif t_name in executor.registered_tools:
                    res = executor.execute(t_name, {"query": task})
                    tool_outputs[t_name] = res.serialize()

            # If no specific tools were matched but search is needed
            if not tool_outputs and (query_needs_search or requires_tools):
                fallback_query = optimize_search_query(task)
                res = executor.execute("web_search", {"query": fallback_query})
                tool_outputs["web_search"] = res.serialize()


            supporting_outputs["tool_executor"] = tool_outputs
            trace.append(self._completed("tool_executor", tool_outputs))
        elif "tool_executor" in active_agents:
            trace.append(self._not_invoked("tool_executor"))


        # 2. Researcher execution (receives tool output context if available)
        research_output = None
        if planner_output.requires_research and "researcher" in active_agents:
            tool_ctx = json.dumps(tool_outputs) if tool_outputs else None
            researcher_inst = self.researcher_factory()
            try:
                research_output = researcher_inst.research(task, supporting_context=tool_ctx)
            except TypeError:
                research_output = researcher_inst.research(task)

            supporting_outputs["researcher"] = research_output
            trace.append(self._completed("researcher", research_output))
        else:
            trace.append(self._not_invoked("researcher"))

        # 3. Coder execution
        coder_output = None
        if planner_output.requires_coding and "coder" in active_agents:
            coder_output = self.coder_factory().code(task)
            supporting_outputs["coder"] = coder_output
            trace.append(self._completed("coder", coder_output))
        else:
            trace.append(self._not_invoked("coder"))

        # 4. Critic execution
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

        def tool_executor_node(state: Dict[str, Any]) -> Dict[str, Any]:
            event("started", "tool_executor")
            tool_calls = state.get("tool_calls")
            executor = self.tool_executor_factory()
            if tool_calls:
                results = [r.serialize() for r in executor.execute_calls(tool_calls)]
            else:
                results = [executor.execute("web_search", {"query": task}).serialize()]
            event("completed", "tool_executor")
            return {"tool_output": results}

        def finalizer_node(state: Dict[str, Any]) -> Dict[str, Any]:
            event("started", "finalizer")
            outputs = {}
            for key, agent_id in (
                ("research_output", "researcher"),
                ("coder_output", "coder"),
                ("critic_output", "critic"),
                ("tool_output", "tool_executor"),
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
            "tool_executor": tool_executor_node,
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
