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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from app.memory.store import format_history_for_prompt


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
        conversation_history: Optional[str] = None,
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
    def _resolve_tools_to_run(self, task: str, requested_tools: list[str]) -> list[str]:
        """Dynamically resolve and expand tools needed for the task, ensuring multi-tool synergies."""
        tools = list(requested_tools or [])
        if not tools:
            tools = ["web_search"]

        q_lower = task.lower()
        is_pure_trend = any(t in q_lower for t in ["trend", "treand"]) and not any(
            w in q_lower for w in ["today", "year", "date", "time", "clock", "now", "recent", "latest", "summit"]
        )

        # Prune date only for pure generic trend searches without temporal indicator
        if is_pure_trend:
            tools = [t for t in tools if t not in {"get_current_date", "date", "time"}]
        else:
            # Temporal grounding: if search is requested or query has temporal intent, include get_current_date
            temporal_signals = ["today", "current", "latest", "recent", "now", "date", "time", "year", "summit", "events", "this year"]
            if any(w in q_lower for w in temporal_signals):
                if not any(t in tools for t in ["get_current_date", "date", "time"]):
                    tools.append("get_current_date")

        # News / search grounding: if date is requested and query also asks for news/events/search
        if any(w in q_lower for w in ["news", "search", "summit", "happened", "event", "headline", "outcome", "what is", "tell me about"]):
            if "web_search" not in tools:
                tools.append("web_search")

        # Computational grounding: if query contains math/calculation intent
        if any(w in q_lower for w in ["calculate", "math", "sum", "average", "multiply", "divide", "%", "compute", "difference"]):
            if "calculator" not in tools:
                tools.append("calculator")

        # Priority sort: temporal tools (1) -> search tools (2) -> calculator (3)
        priority_map = {
            "get_current_date": 1,
            "date": 1,
            "time": 1,
            "web_search": 2,
            "retriever": 2,
            "calculator": 3,
        }
        return sorted(list(dict.fromkeys(tools)), key=lambda t: priority_map.get(t, 2))

    def _execute_tool_workflow(
        self,
        task: str,
        requested_tools: list[str],
        executor: Any,
        explicit_tool_calls: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Execute a multi-tool workflow with tool-to-tool communication and dynamic cyclic looping."""
        if explicit_tool_calls:
            results = [r.serialize() for r in executor.execute_calls(explicit_tool_calls)]
            outputs = {r.get("tool_name", f"tool_{i}"): r for i, r in enumerate(results)}
            return outputs, results

        tools_to_run = self._resolve_tools_to_run(task, requested_tools)
        tool_outputs: dict[str, Any] = {}
        serialized_results: list[dict[str, Any]] = []
        accumulated_tool_context: dict[str, Any] = {}
        executed_tool_set: set[str] = set()

        max_iterations = 3
        iteration = 0

        while tools_to_run and iteration < max_iterations:
            iteration += 1
            t_name = tools_to_run.pop(0)
            if t_name in executed_tool_set:
                continue
            executed_tool_set.add(t_name)

            if t_name in {"get_current_date", "date", "time"}:
                res = executor.execute("get_current_date", {})
                ser = res.serialize()
                tool_outputs["get_current_date"] = ser
                serialized_results.append(ser)
                if res.status == "success" and isinstance(res.result, dict):
                    accumulated_tool_context["current_date"] = res.result.get("current_date")
                    accumulated_tool_context["year"] = res.result.get("year")
            elif t_name == "calculator":
                calc_expr = task
                res = executor.execute("calculator", {"expression": calc_expr})
                ser = res.serialize()
                tool_outputs["calculator"] = ser
                serialized_results.append(ser)
                if res.status == "success":
                    accumulated_tool_context["calculation_result"] = res.result
            elif t_name == "web_search":
                search_query = task
                # Tool-to-tool context communication: enrich query with discovered live date if temporal query
                if "current_date" in accumulated_tool_context:
                    curr_yr = str(accumulated_tool_context.get("year", ""))
                    temporal_signals = ["latest", "current", "recent", "today", "now", "summit", "events", "this year"]
                    if any(w in task.lower() for w in temporal_signals):
                        if curr_yr and curr_yr not in search_query:
                            search_query = f"{task} {curr_yr}"
                res = executor.execute("web_search", {"query": search_query})
                ser = res.serialize()
                tool_outputs["web_search"] = ser
                serialized_results.append(ser)
                if res.status == "success":
                    accumulated_tool_context["search_findings"] = res.result
            elif t_name in executor.registered_tools:
                res = executor.execute(t_name, {"query": task, "context": accumulated_tool_context})
                ser = res.serialize()
                tool_outputs[t_name] = ser
                serialized_results.append(ser)
            else:
                res = executor.execute("web_search", {"query": task})
                ser = res.serialize()
                tool_outputs["web_search"] = ser
                serialized_results.append(ser)

        return tool_outputs, serialized_results

    def execute(
        self,
        task: str,
        planner_output: PlannerOutput,
        architecture: MASArchitecture,
        conversation_history: Optional[str] = None,
    ) -> LLMExecutionResult:
        """Run the active agents in topological order and collect their outputs."""

        active_agents = set(architecture.active_agent_ids)
        supporting_outputs: Dict[str, Any] = {}
        trace: list[AgentExecutionRecord] = []

        # 1. Tool execution (dedicated tool_executor node)
        requires_tools = getattr(planner_output, "requires_tools", False)
        tools_needed = list(getattr(planner_output, "tools_needed", []) or [])
        selected_agents = list(getattr(planner_output, "selected_agents", []) or [])
        req_caps = set(planner_output.required_capabilities or [])

        has_tool_need = (
            "tool_executor" in active_agents
            or "tool_executor" in selected_agents
            or requires_tools
            or bool(tools_needed)
            or bool(req_caps & {"tool_use", "web_search", "external_api", "tools"})
        )

        tool_outputs = {}
        if has_tool_need:
            executor = self.tool_executor_factory()
            tool_outputs, _ = self._execute_tool_workflow(task, tools_needed, executor)
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
            finalizer_inst = self.finalizer_factory()
            try:
                final_response = finalizer_inst.finalize(
                    original_task=task,
                    supporting_info=finalizer_context,
                    conversation_history=conversation_history,
                )
            except TypeError:
                final_response = finalizer_inst.finalize(
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
            return {
                "planner_output": planner_output,
            }

        def researcher_node(state: Dict[str, Any]) -> Dict[str, Any]:
            event("started", "researcher")
            res_agent = self.researcher_factory()
            if state.get("tool_output"):
                supporting = json.dumps(state["tool_output"], default=str)
                try:
                    output = res_agent.research(task, supporting_context=supporting)
                except TypeError:
                    output = res_agent.research(task)
            else:
                output = res_agent.research(task)
            event("completed", "researcher")
            return {
                "research_output": output,
            }

        def coder_node(state: Dict[str, Any]) -> Dict[str, Any]:
            event("started", "coder")
            output = self.coder_factory().code(task)
            event("completed", "coder")
            return {
                "coder_output": output,
            }

        def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
            event("started", "critic")
            outputs = {}
            if state.get("research_output") is not None:
                outputs["researcher"] = state["research_output"]
            if state.get("coder_output") is not None:
                outputs["coder"] = state["coder_output"]
            if state.get("tool_output") is not None:
                outputs["tool_executor"] = state["tool_output"]
            output = self.critic_factory().review(
                original_task=task,
                output_to_review=self._supporting_text(outputs),
            )
            event("completed", "critic")
            suggested_calls = getattr(output, "suggested_tool_calls", []) or []

            res_dict: Dict[str, Any] = {
                "critic_output": output,
            }
            if suggested_calls:
                res_dict["tool_calls"] = suggested_calls
            return res_dict

        def tool_executor_node(state: Dict[str, Any]) -> Dict[str, Any]:
            event("started", "tool_executor")
            tool_calls = state.get("tool_calls")
            executor = self.tool_executor_factory()
            tools_to_run = list(getattr(planner_output, "tools_needed", []) or [])
            _, results = self._execute_tool_workflow(
                task,
                tools_to_run,
                executor,
                explicit_tool_calls=tool_calls,
            )
            event("completed", "tool_executor")

            prev_output = state.get("tool_output")
            if prev_output and tool_calls:
                if isinstance(prev_output, list):
                    merged_output = list(prev_output) + results
                elif isinstance(prev_output, dict):
                    merged_output = [prev_output] + results
                else:
                    merged_output = results
            else:
                merged_output = results

            return {
                "tool_output": merged_output,
                "tool_calls": [],
            }

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

            all_msgs = state.get("messages", [])
            history_str = format_history_for_prompt(all_msgs) if all_msgs else None

            finalizer_inst = self.finalizer_factory()
            try:
                output = finalizer_inst.finalize(
                    original_task=task,
                    supporting_info=self._supporting_text(outputs),
                    conversation_history=history_str,
                )
            except TypeError:
                output = finalizer_inst.finalize(
                    original_task=task,
                    supporting_info=self._supporting_text(outputs),
                )
            event("completed", "finalizer")
            final_text = getattr(output, "final_answer", str(output))
            return {
                "final_answer": output,
                "messages": [AIMessage(content=final_text)],
            }

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
