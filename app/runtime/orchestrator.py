"""Runtime path connecting PlannerOutput to adaptive architecture execution."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol

from pydantic import BaseModel, Field

from app.agents.planner import Planner, PlannerOutput
from app.architecture.llm_adapter import LLMArchitectureAdapter
from app.architecture.manager import ArchitectureManager
from app.graph.adaptive_integration import AdaptiveWorkflowAdapter
from app.graph.dynamic_builder import DynamicGraphBuilder, DynamicGraphMetadata
from app.graph.state import MASState
from app.runtime.llm_execution import (
    AgentExecutionRecord,
    ExistingLLMAgentExecutor,
    LLMExecutionResult,
)


class PlannerRunner(Protocol):
    model: Any

    def plan(self, task: str) -> PlannerOutput: ...


class RuntimeExecutionResult(BaseModel):
    """Planner, adaptation, and final workflow result for one user task."""

    user_task: str
    planner_output: Dict[str, Any]
    initial_active_agents: list[str]
    architecture_decision: Dict[str, Any]
    proposed_actions: list[Dict[str, Any]] = Field(default_factory=list)
    accepted_actions: list[Dict[str, Any]] = Field(default_factory=list)
    rejected_actions: list[Dict[str, Any]] = Field(default_factory=list)
    final_active_agents: list[str]
    architecture_version: int
    architecture_changed: bool
    langgraph_compatible: bool
    final_architecture: Dict[str, Any]
    agents_actually_invoked: list[str] = Field(default_factory=list)
    execution_trace: list[AgentExecutionRecord] = Field(default_factory=list)
    execution_consistency_errors: list[str] = Field(default_factory=list)
    final_response: Any = None

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def to_dict(self) -> Dict[str, Any]:
        return self.serialize()


class RuntimeTraceEvent(BaseModel):
    """Ordered event emitted by the dynamic graph runtime."""

    sequence: int
    event: str
    node: Optional[str] = None
    architecture_version: int
    architecture_stage: str
    status: str

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class DynamicGraphExecutionResult(BaseModel):
    """Inspectable result of one dynamically compiled graph execution."""

    task: str
    planner_output: Dict[str, Any]
    architecture_decision: Dict[str, Any]
    initial_architecture: Dict[str, Any]
    final_architecture: Dict[str, Any]
    graph_nodes: list[str]
    graph_edges: list[dict[str, str]]
    active_agents: list[str]
    architecture_version: int
    architecture_actions: list[Dict[str, Any]] = Field(default_factory=list)
    compiled: bool
    graph_representation: str
    execution_trace: list[RuntimeTraceEvent] = Field(default_factory=list)
    agents_invoked: list[str] = Field(default_factory=list)
    final_response: Any = None

    # Mid-execution adaptation metadata
    architecture_versions: list[int] = Field(default_factory=list)
    architecture_before_reassessment: Optional[Dict[str, Any]] = None
    architecture_after_reassessment: Optional[Dict[str, Any]] = None
    triggering_agent: Optional[str] = None
    context_that_triggered_reassessment: Optional[str] = None
    reassessment_actions_proposed: list[Dict[str, Any]] = Field(default_factory=list)
    reassessment_actions_accepted: list[Dict[str, Any]] = Field(default_factory=list)
    reassessment_error: Optional[str] = None
    graph_before: Optional[Dict[str, Any]] = None
    graph_after: Optional[Dict[str, Any]] = None
    execution_path_before: list[str] = Field(default_factory=list)
    execution_path_after: list[str] = Field(default_factory=list)
    actual_execution_path: list[str] = Field(default_factory=list)

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class AdaptiveRuntimeOrchestrator:
    """Execute the existing MAS after PlannerOutput-driven adaptation.

    The injected workflow runner is primarily for deterministic tests. In the
    normal path, the existing AdaptiveWorkflowAdapter invokes the protected
    baseline workflow unchanged.
    """

    def __init__(
        self,
        *,
        planner: PlannerRunner,
        architecture_adapter: Optional[LLMArchitectureAdapter] = None,
        initial_manager: Optional[ArchitectureManager] = None,
        workflow_runner: Optional[Callable[[str], MASState]] = None,
        agent_executor: Optional[ExistingLLMAgentExecutor] = None,
    ) -> None:
        self.planner = planner
        manager = initial_manager or ArchitectureManager.create_default_architecture()
        self.workflow_adapter = AdaptiveWorkflowAdapter(manager=manager)
        self.architecture_adapter = architecture_adapter or LLMArchitectureAdapter(
            planner.model,
            workflow_adapter=self.workflow_adapter,
        )
        if architecture_adapter is not None:
            self.workflow_adapter = architecture_adapter.workflow_adapter
        self.workflow_runner = workflow_runner
        self.agent_executor = agent_executor

    @classmethod
    def from_settings(cls) -> "AdaptiveRuntimeOrchestrator":
        """Construct the runtime using the existing configured Planner/client."""

        return cls(
            planner=Planner.from_settings(),
            agent_executor=ExistingLLMAgentExecutor(),
        )

    def run(self, user_task: str) -> RuntimeExecutionResult:
        """Plan, adapt, execute the existing workflow, and collect metadata."""

        planner_output = self.planner.plan(user_task)
        initial = self.architecture_adapter.workflow_adapter.current_architecture()
        adaptation = self.architecture_adapter.adapt_from_planner_output(planner_output)
        final = self.architecture_adapter.workflow_adapter.current_architecture()

        if self.agent_executor is not None:
            execution = self.agent_executor.execute(
                user_task,
                planner_output,
                final,
            )
            execution_trace = execution.execution_trace
            final_response = execution.final_response
        else:
            if self.workflow_runner is not None:
                state = self.workflow_runner(user_task)
            else:
                state = (
                    self.architecture_adapter.workflow_adapter.run_baseline_workflow(
                        user_task
                    )
                )
            final_response = (
                state.get("final_answer") if isinstance(state, dict) else None
            )
            if isinstance(final_response, BaseModel):
                final_response = final_response.model_dump(mode="json")
            execution_trace = self._observe_execution(state)
        invoked_agents = [
            record.agent_id for record in execution_trace if record.invoked
        ]
        consistency_errors = self._check_execution_consistency(
            final_active_agents=final.active_agent_ids,
            planner_output=planner_output,
            invoked_agents=invoked_agents,
        )

        return RuntimeExecutionResult(
            user_task=user_task,
            planner_output=adaptation.planner_output,
            initial_active_agents=sorted(initial.active_agent_ids),
            architecture_decision=adaptation.architecture_decision,
            proposed_actions=adaptation.parsed_actions,
            accepted_actions=adaptation.valid_actions,
            rejected_actions=[item.serialize() for item in adaptation.rejected_actions],
            final_active_agents=sorted(final.active_agent_ids),
            architecture_version=self.architecture_adapter.workflow_adapter.current_version(),
            architecture_changed=initial.serialize() != final.serialize(),
            langgraph_compatible=adaptation.langgraph_compatible,
            final_architecture=final.serialize(),
            agents_actually_invoked=invoked_agents,
            execution_trace=execution_trace,
            execution_consistency_errors=consistency_errors,
            final_response=final_response,
        )

    def run_dynamic(
        self,
        user_task: str,
        *,
        runtime_adaptation: bool = False,
    ) -> DynamicGraphExecutionResult:
        """Adapt architecture, compile a graph from it, and execute that graph.

        When runtime_adaptation is True, the orchestrator compiles v0 up to the
        reassessment boundary, invokes v0 via LangGraph, inspects intermediate agent output,
        reassesses the architecture using the existing LLMArchitectureAdapter, advances
        architecture version to v1, compiles graph v1 starting at the first pending node,
        and invokes v1 via LangGraph with the accumulated state.
        """
        sequence = 0
        events: list[RuntimeTraceEvent] = []

        def emit(
            event: str,
            *,
            node: Optional[str] = None,
            version: int = 0,
            stage: str = "ADAPTED",
            status: str = "completed",
        ) -> None:
            nonlocal sequence
            sequence += 1
            events.append(
                RuntimeTraceEvent(
                    sequence=sequence,
                    event=event,
                    node=node,
                    architecture_version=version,
                    architecture_stage=stage,
                    status=status,
                )
            )

        emit("planner.started", node="planner", stage="ORIGINAL", status="started")
        planner_output = self.planner.plan(user_task)
        emit("planner.completed", node="planner", stage="ORIGINAL")
        initial = self.architecture_adapter.workflow_adapter.current_architecture()
        version_before = self.architecture_adapter.workflow_adapter.current_version()
        emit(
            "architecture_adaptation.started", version=version_before, status="started"
        )
        adaptation = self.architecture_adapter.adapt_from_planner_output(planner_output)
        final = self.architecture_adapter.workflow_adapter.current_architecture()
        version = self.architecture_adapter.workflow_adapter.current_version()
        emit("architecture_adaptation.completed", version=version)
        if initial.serialize() != final.serialize():
            emit("architecture.changed", version=version)

        executor = self.agent_executor or ExistingLLMAgentExecutor()

        if not runtime_adaptation:
            emit("graph.build.started", version=version, stage="ADAPTED", status="started")
            handlers = executor.node_handlers(
                user_task,
                planner_output,
                event_callback=lambda status, node: emit(
                    f"agent.{node}.{status}",
                    node=node,
                    version=version,
                    stage="COMPILED",
                ),
            )
            emit(
                "graph.compile.started", version=version, stage="COMPILED", status="started"
            )
            built = DynamicGraphBuilder().build(
                final,
                planner_output,
                handlers,
                architecture_version=version,
                architecture_actions=adaptation.valid_actions,
            )
            emit("graph.compile.completed", version=version, stage="COMPILED")
            state = built.compiled_graph.invoke({"original_task": user_task})
            emit("workflow.completed", version=version, stage="ACTUAL")
            final_response = state.get("final_answer") if isinstance(state, dict) else None
            if isinstance(final_response, BaseModel):
                final_response = final_response.model_dump(mode="json")
            invoked = []
            for event in events:
                if event.event.endswith(".started") and event.node is not None:
                    if event.node not in invoked:
                        invoked.append(event.node)
            return DynamicGraphExecutionResult(
                task=user_task,
                planner_output=adaptation.planner_output,
                architecture_decision=adaptation.architecture_decision,
                initial_architecture=initial.serialize(),
                final_architecture=final.serialize(),
                graph_nodes=built.metadata.graph_nodes,
                graph_edges=built.metadata.graph_edges,
                active_agents=built.metadata.active_agents,
                architecture_version=version,
                architecture_actions=adaptation.valid_actions,
                compiled=built.metadata.compiled,
                graph_representation=built.metadata.graph_representation,
                execution_trace=events,
                agents_invoked=invoked,
                final_response=final_response,
                actual_execution_path=invoked,
            )

        # ------------------------------------------------------------------
        # Runtime / During-Execution Architecture Adaptation
        # ------------------------------------------------------------------
        emit("architecture.v0.created", version=version, stage="ADAPTED")

        # Compile Graph v0 up to the reassessment boundary (excluding finalizer)
        handlers_v0 = executor.node_handlers(
            user_task,
            planner_output,
            event_callback=lambda status, node: emit(
                f"agent.{node}.{status}",
                node=node,
                version=version,
                stage="COMPILED",
            ),
        )
        built_v0 = DynamicGraphBuilder().build(
            final,
            planner_output,
            handlers_v0,
            architecture_version=version,
            architecture_actions=adaptation.valid_actions,
            include_finalizer=False,
        )
        emit("graph.v0.compiled", version=version, stage="COMPILED")

        # Invoke Graph v0 via LangGraph
        state_v0 = built_v0.compiled_graph.invoke({"original_task": user_task})

        # Capture intermediate context and triggering agent
        research_output = state_v0.get("research_output")
        coder_output = state_v0.get("coder_output")
        context_parts = []
        triggering_agent = None
        executed_in_v0 = {"planner"}

        if research_output:
            context_parts.append(f"research_output:\n{research_output}")
            triggering_agent = "researcher"
            executed_in_v0.add("researcher")
        if coder_output:
            context_parts.append(f"coder_output:\n{coder_output}")
            triggering_agent = "coder"
            executed_in_v0.add("coder")
        context_str = "\n\n".join(context_parts)

        # Record execution path before reassessment
        path_before = [
            event.node
            for event in events
            if event.event.endswith(".started") and event.node is not None
        ]
        unique_path_before = []
        for n in path_before:
            if n not in unique_path_before:
                unique_path_before.append(n)

        emit("architecture.reassessment.started", version=version, stage="REASSESSMENT", status="started")
        reassessment = None
        reassessment_error = None
        try:
            reassessment = self.architecture_adapter.reassess_from_context(
                user_task,
                context_str,
                final,
            )
        except Exception as exc:
            reassessment_error = str(exc)
        emit("architecture.reassessment.completed", version=version, stage="REASSESSMENT")

        arch_v1 = self.architecture_adapter.workflow_adapter.current_architecture()
        version_v1 = self.architecture_adapter.workflow_adapter.current_version()
        arch_changed = bool(
            reassessment
            and reassessment.valid_actions
            and ((version_v1 > version) or (arch_v1.serialize() != final.serialize()))
        )

        graph_before_meta = {
            "graph_nodes": built_v0.metadata.graph_nodes,
            "graph_edges": built_v0.metadata.graph_edges,
            "graph_representation": built_v0.metadata.graph_representation,
        }

        if arch_changed:
            emit("architecture.v1.created", version=version_v1, stage="ADAPTED")
            v1_active_agents = set(arch_v1.active_agent_ids)
            updated_plan = planner_output.model_copy(
                update={
                    "requires_coding": "coder" in v1_active_agents,
                    "requires_verification": "critic" in v1_active_agents,
                }
            )
            handlers_v1 = executor.node_handlers(
                user_task,
                updated_plan,
                event_callback=lambda status, node: emit(
                    f"agent.{node}.{status}",
                    node=node,
                    version=version_v1,
                    stage="COMPILED",
                ),
            )
            if "coder" in v1_active_agents and "coder" not in executed_in_v0:
                entry_node = "coder"
            elif "critic" in v1_active_agents and "critic" not in executed_in_v0:
                entry_node = "critic"
            else:
                entry_node = "finalizer"

            built_v1 = DynamicGraphBuilder().build(
                arch_v1,
                updated_plan,
                handlers_v1,
                architecture_version=version_v1,
                architecture_actions=reassessment.valid_actions if reassessment else [],
                entry_point=entry_node,
                excluded_nodes=executed_in_v0,
                include_finalizer=True,
            )
            emit("graph.v1.compiled", version=version_v1, stage="COMPILED")

            # Invoke Graph v1 via LangGraph with accumulated state
            final_state = built_v1.compiled_graph.invoke(state_v0)
            emit("workflow.completed", version=version_v1, stage="ACTUAL")

            final_built = built_v1
            final_arch = arch_v1
            final_version = version_v1
        else:
            # Continue with remaining nodes on current architecture
            built_continue = DynamicGraphBuilder().build(
                final,
                planner_output,
                handlers_v0,
                architecture_version=version,
                entry_point="finalizer",
                excluded_nodes=executed_in_v0,
                include_finalizer=True,
            )
            final_state = built_continue.compiled_graph.invoke(state_v0)
            emit("workflow.completed", version=version, stage="ACTUAL")

            final_built = built_continue
            final_arch = final
            final_version = version

        final_response = final_state.get("final_answer") if isinstance(final_state, dict) else None
        if isinstance(final_response, BaseModel):
            final_response = final_response.model_dump(mode="json")

        invoked = []
        for event in events:
            if event.event.endswith(".started") and event.node is not None:
                if event.node not in invoked:
                    invoked.append(event.node)

        reassessment_completed_found = False
        path_after = []
        for event in events:
            if event.event == "architecture.reassessment.completed":
                reassessment_completed_found = True
                continue
            if reassessment_completed_found and event.event.endswith(".started") and event.node is not None:
                if event.node not in path_after:
                    path_after.append(event.node)

        graph_after_meta = {
            "graph_nodes": final_built.metadata.graph_nodes,
            "graph_edges": final_built.metadata.graph_edges,
            "graph_representation": final_built.metadata.graph_representation,
        }

        all_nodes = built_v0.metadata.graph_nodes + [
            n for n in final_built.metadata.graph_nodes if n not in built_v0.metadata.graph_nodes
        ]
        all_edges = built_v0.metadata.graph_edges + [
            e for e in final_built.metadata.graph_edges if e not in built_v0.metadata.graph_edges
        ]

        return DynamicGraphExecutionResult(
            task=user_task,
            planner_output=adaptation.planner_output,
            architecture_decision=adaptation.architecture_decision,
            initial_architecture=initial.serialize(),
            final_architecture=final_arch.serialize(),
            graph_nodes=all_nodes,
            graph_edges=all_edges,
            active_agents=final_built.metadata.active_agents,
            architecture_version=final_version,
            architecture_actions=(
                adaptation.valid_actions + (reassessment.valid_actions if reassessment else [])
            ),
            compiled=True,
            graph_representation=final_built.metadata.graph_representation,
            execution_trace=events,
            agents_invoked=invoked,
            final_response=final_response,
            architecture_versions=[version, final_version] if arch_changed else [version],
            architecture_before_reassessment=final.serialize(),
            architecture_after_reassessment=final_arch.serialize(),
            triggering_agent=triggering_agent,
            context_that_triggered_reassessment=context_str,
            reassessment_actions_proposed=reassessment.parsed_actions if reassessment else [],
            reassessment_actions_accepted=reassessment.valid_actions if reassessment else [],
            reassessment_error=reassessment_error,
            graph_before=graph_before_meta,
            graph_after=graph_after_meta,
            execution_path_before=unique_path_before,
            execution_path_after=path_after,
            actual_execution_path=invoked,
        )

    @staticmethod
    def _observe_execution(state: Any) -> list[AgentExecutionRecord]:
        """Observe existing workflow outputs without invoking agents again."""

        state_dict = state if isinstance(state, dict) else {}
        output_keys = {
            "planner": "planner_output",
            "researcher": "research_output",
            "coder": "coder_output",
            "critic": "critic_output",
            "finalizer": "final_answer",
        }
        records: list[AgentExecutionRecord] = []
        for agent_id, output_key in output_keys.items():
            output = state_dict.get(output_key)
            invoked = output is not None
            if invoked and isinstance(output, BaseModel):
                output = output.model_dump(mode="json")
            records.append(
                AgentExecutionRecord(
                    agent_id=agent_id,
                    invoked=invoked,
                    status="completed" if invoked else "not_invoked",
                    output=output,
                )
            )
        return records

    @staticmethod
    def _check_execution_consistency(
        *,
        final_active_agents: set[str],
        planner_output: PlannerOutput,
        invoked_agents: list[str],
    ) -> list[str]:
        """Report mismatches between architecture state and observed workflow output."""

        errors: list[str] = []
        for agent_id in invoked_agents:
            if agent_id != "planner" and agent_id not in final_active_agents:
                errors.append(f"inactive agent '{agent_id}' produced workflow output")

        required_by_plan = []
        if planner_output.requires_research:
            required_by_plan.append("researcher")
        if planner_output.requires_coding:
            required_by_plan.append("coder")
        if planner_output.requires_verification:
            required_by_plan.append("critic")
        for agent_id in required_by_plan:
            if agent_id in final_active_agents and agent_id not in invoked_agents:
                errors.append(
                    f"active required agent '{agent_id}' was not observed in workflow output"
                )
        return errors


def run_adaptive_runtime(user_task: str) -> RuntimeExecutionResult:
    """Run the production planner-to-adaptive-architecture runtime path."""

    return AdaptiveRuntimeOrchestrator.from_settings().run(user_task)
