"""Controlled multi-task dynamic-graph evaluation.

Proves that different task requirements produce different compiled
LangGraph architectures and execution paths from the SAME initial MAS
architecture, using the existing pipeline:

PlannerOutput -> LLMArchitectureAdapter -> ArchitectureManager
-> AdaptiveArchitecture -> AdaptiveWorkflowAdapter -> DynamicGraphBuilder
-> graph.compile()

Constraints honored:
- app/graph/workflow.py is not modified (it is not even imported here).
- Existing agent implementations are reused unchanged via
  ExistingLLMAgentExecutor factory injection.
- RL/meta-RL algorithms and LLM/provider configuration are untouched;
  the live entry point reuses the existing settings-backed clients.

Experiment controls:
1. Every task starts from a freshly constructed default baseline
   architecture (ArchitectureManager.create_default_architecture()), so
   architecture state is reset between tasks and one task's changes can
   never influence another task.
2. The compiled graph is never hard-coded: it is produced at runtime by
   DynamicGraphBuilder.build() from the adapted architecture and the
   planner's requirements.
3. PNG rendering is never required; optional image export is best-effort
   and failure-tolerant.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from pydantic import BaseModel, Field

from app.agents.planner import Planner, PlannerOutput
from app.architecture.llm_adapter import (
    LLMArchitectureAdapter,
    LLMClient,
)
from app.architecture.manager import ArchitectureManager
from app.config.settings import settings
from app.graph.adaptive_integration import AdaptiveWorkflowAdapter
from app.graph.dynamic_builder import DynamicGraphBuilder
from app.graph.state import MASState
from app.runtime.llm_execution import ExistingLLMAgentExecutor
from app.runtime.orchestrator import AdaptiveRuntimeOrchestrator

# ---------------------------------------------------------------------------
# Controlled task definitions (task TEXT only; never expected graphs)
# ---------------------------------------------------------------------------

TASK_A_CONVERSATION = "Hello, how are you?"
TASK_B_RESEARCH = (
    "Research retrieval-augmented generation and summarize the main approaches."
)
TASK_C_CODING = (
    "Write a Java solution to find the second largest element in an array."
)
TASK_D_CODING_VERIFICATION = (
    "Write a Java solution to find the second largest element in an array and "
    "verify the edge cases."
)

EXPERIMENT_TASKS: Sequence[Tuple[str, str]] = (
    ("conversation", TASK_A_CONVERSATION),
    ("research", TASK_B_RESEARCH),
    ("coding", TASK_C_CODING),
    ("coding_verification", TASK_D_CODING_VERIFICATION),
)

BASELINE_TASK_NAME = "conversation"


class PlannerRunner(Protocol):
    """Minimal contract for the existing Planner (mirrors the runtime)."""

    model: Any

    def plan(self, task: str) -> PlannerOutput: ...


class PlannerRunnerFactory(Protocol):
    def __call__(self, *, model: Any) -> PlannerRunner: ...


# ---------------------------------------------------------------------------
# Per-task result
# ---------------------------------------------------------------------------


class Step36TaskResult(BaseModel):
    """Complete inspection record for one task in the Step 36 experiment."""

    task_name: str
    task: str

    # Planning stage
    planner_output: Dict[str, Any]
    required_capabilities: List[str] = Field(default_factory=list)

    # Architecture adaptation stage
    architecture_decision: Dict[str, Any]
    proposed_actions: List[Dict[str, Any]] = Field(default_factory=list)
    accepted_actions: List[Dict[str, Any]] = Field(default_factory=list)
    rejected_actions: List[Dict[str, Any]] = Field(default_factory=list)
    initial_active_agents: List[str] = Field(default_factory=list)
    final_active_agents: List[str] = Field(default_factory=list)
    architecture_changed: bool = False
    architecture_version: int = 0
    baseline_architecture_id: str = ""

    # Compiled graph stage
    graph_nodes: List[str] = Field(default_factory=list)
    graph_edges: List[Dict[str, str]] = Field(default_factory=list)
    graph_representation: str = ""
    compiled: bool = False

    # Execution stage
    execution_path: List[str] = Field(default_factory=list)
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    final_response: Any = None

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def to_dict(self) -> Dict[str, Any]:
        return self.serialize()


# ---------------------------------------------------------------------------
# Architecture-difference metric
# ---------------------------------------------------------------------------


class ArchitectureDifferenceMetric(BaseModel):
    """Architecture-difference metrics for one task versus the baseline."""

    task_name: str
    active_agent_count: int
    graph_node_count: int
    graph_edge_count: int
    graph_signature: str
    differs_from_baseline: bool
    baseline_graph_signature: str
    differs_from_other_tasks: List[str] = Field(default_factory=list)

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


class Step36ComparisonReport(BaseModel):
    """Cross-task comparison that makes architectural differences obvious."""

    tasks: List[str] = Field(default_factory=list)
    required_capabilities: Dict[str, List[str]] = Field(default_factory=dict)
    actions: Dict[str, List[str]] = Field(default_factory=dict)
    final_agents: Dict[str, List[str]] = Field(default_factory=dict)
    compiled_edges: Dict[str, List[str]] = Field(default_factory=dict)
    execution_paths: Dict[str, List[str]] = Field(default_factory=dict)
    graph_signatures: Dict[str, str] = Field(default_factory=dict)
    differences: List[ArchitectureDifferenceMetric] = Field(default_factory=list)
    all_graphs_differ_from_baseline: bool = False
    all_graphs_differ_from_each_other: bool = False
    conclusion: str = ""

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def render_table(self) -> str:
        """Render the required comparison table as aligned text."""
        headers = [
            "Task",
            "Required capabilities",
            "Actions",
            "Final agents",
            "Compiled edges",
            "Execution path",
        ]
        rows: List[List[str]] = []
        for name in self.tasks:
            rows.append(
                [
                    name,
                    ", ".join(self.required_capabilities.get(name, [])) or "(none)",
                    ", ".join(self.actions.get(name, [])) or "(none)",
                    ", ".join(self.final_agents.get(name, [])) or "(none)",
                    " | ".join(self.compiled_edges.get(name, [])) or "(none)",
                    " -> ".join(self.execution_paths.get(name, [])) or "(none)",
                ]
            )

        widths = [
            max(
                [len(headers[col])] + [len(row[col]) for row in rows]
            )
            for col in range(len(headers))
        ]
        separator = "-+-".join("-" * width for width in widths)
        lines = [
            " | ".join(
                headers[col].ljust(widths[col]) for col in range(len(headers))
            ),
            separator,
        ]
        for row in rows:
            lines.append(
                " | ".join(row[col].ljust(widths[col]) for col in range(len(row)))
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full experiment result
# ---------------------------------------------------------------------------


class Step36ExperimentResult(BaseModel):
    """Complete result of the controlled multi-task Step 36 experiment."""

    tasks: List[Step36TaskResult] = Field(default_factory=list)
    baseline_signature: str = ""
    comparison: Optional[Step36ComparisonReport] = None

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def demonstrated(self) -> bool:
        """Return True when the target claim of Step 36 was demonstrated."""
        return bool(
            self.comparison
            and self.comparison.all_graphs_differ_from_baseline
            and self.comparison.all_graphs_differ_from_each_other
            and all(item.compiled for item in self.tasks)
            and len(self.tasks) >= 2
        )


# ---------------------------------------------------------------------------
# Deterministic fakes (tests / demo only)
# ---------------------------------------------------------------------------


class FakeDecisionLLM:
    """Deterministic LLM client that replays a canned architecture decision."""

    def __init__(self, decision: Dict[str, Any]) -> None:
        self.decision = decision
        self.model = self

    def invoke(self, messages: Any) -> Any:
        class _Response:
            content = self.decision

        return _Response()


class FakePlanner:
    """Deterministic planner returning a canned PlannerOutput for any task."""

    def __init__(self, output: PlannerOutput, client: Any = None) -> None:
        self.output = output
        self.model = client if client is not None else FakeDecisionLLM({})

    def plan(self, task: str) -> PlannerOutput:
        return self.output


class _FakeResearcher:
    def research(self, task: str):
        from app.agents.researcher import ResearcherOutput

        return ResearcherOutput(
            research_question=task,
            findings=["deterministic research finding for step 36"],
            sources_or_evidence=["general knowledge (no live search)"],
        )


class _FakeCoder:
    def code(self, task: str):
        from app.agents.coder import CoderOutput

        return CoderOutput(
            approach="deterministic approach for step 36",
            code="public class Solution { /* deterministic step 36 code */ }",
            explanation="deterministic explanation for step 36",
        )


class _FakeCritic:
    def review(self, original_task: str, output_to_review: str):
        from app.agents.critic import CriticOutput

        return CriticOutput(
            overall_assessment="deterministic review for step 36",
            verification_status="correct",
        )


class _FakeFinalizer:
    def finalize(self, original_task: str, supporting_info: str):
        from app.agents.finalizer import FinalizerOutput

        return FinalizerOutput(
            final_answer="deterministic final answer for step 36",
            key_points=[supporting_info],
            limitations=[],
        )


def fake_agent_executor() -> ExistingLLMAgentExecutor:
    """Executor wired to deterministic fake agents (no LLM/network calls)."""
    return ExistingLLMAgentExecutor(
        researcher_factory=_FakeResearcher,
        coder_factory=_FakeCoder,
        critic_factory=_FakeCritic,
        finalizer_factory=_FakeFinalizer,
    )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def graph_signature(graph_edges: List[Dict[str, str]]) -> str:
    """Return a deterministic textual signature of a compiled graph topology."""
    normalized = sorted((edge["source"], edge["target"]) for edge in graph_edges)
    return ";".join(f"{source}->{target}" for source, target in normalized)


def execution_path_from_trace(execution_trace: List[Dict[str, Any]]) -> List[str]:
    """Derive the actual node execution path from a dynamic runtime trace."""
    path: List[str] = []
    for event in execution_trace:
        node = event.get("node")
        event_name = str(event.get("event", ""))
        if event_name == "workflow.completed":
            break
        if node is not None and event_name.endswith(".started"):
            if node not in path:
                path.append(node)
    return path


def _active_agent_ids_from_serialized(architecture: Dict[str, Any]) -> List[str]:
    """Extract sorted active agent ids from a serialized MASArchitecture."""
    return sorted(
        agent["agent_id"]
        for agent in architecture.get("agents", [])
        if agent.get("active")
    )


def _split_actions(
    decision: Dict[str, Any], accepted: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split proposed decision actions into accepted and rejected lists."""
    accepted_keys = {str(action) for action in accepted}
    proposed = list(decision.get("actions") or [])
    accepted_matched = [action for action in proposed if str(action) in accepted_keys]
    rejected = [action for action in proposed if str(action) not in accepted_keys]
    return accepted_matched, rejected


def _task_result_from_dynamic(
    task_name: str,
    task: str,
    dynamic: Any,
    decision: Dict[str, Any],
) -> Step36TaskResult:
    """Build the Step 36 per-task record from a dynamic execution result."""
    trace = [event.serialize() for event in dynamic.execution_trace]
    accepted, rejected = _split_actions(decision, dynamic.architecture_actions)
    return Step36TaskResult(
        task_name=task_name,
        task=task,
        planner_output=dynamic.planner_output,
        required_capabilities=list(
            dynamic.planner_output.get("required_capabilities") or []
        ),
        architecture_decision=decision,
        proposed_actions=list(decision.get("actions") or []),
        accepted_actions=accepted,
        rejected_actions=rejected,
        initial_active_agents=_active_agent_ids_from_serialized(
            dynamic.initial_architecture
        ),
        final_active_agents=dynamic.active_agents,
        architecture_changed=(
            dynamic.initial_architecture != dynamic.final_architecture
            or bool(accepted)
            or dynamic.architecture_version > 0
        ),
        architecture_version=dynamic.architecture_version,
        baseline_architecture_id=str(
            dynamic.initial_architecture.get("architecture_id", "")
        ),
        graph_nodes=dynamic.graph_nodes,
        graph_edges=dynamic.graph_edges,
        graph_representation=dynamic.graph_representation,
        compiled=dynamic.compiled,
        execution_path=execution_path_from_trace(trace),
        execution_trace=trace,
        final_response=dynamic.final_response,
    )


def _orchestrator_for_task(
    planner_runner: PlannerRunner,
    decision_client: LLMClient,
    agent_executor: Optional[ExistingLLMAgentExecutor],
) -> Tuple[AdaptiveRuntimeOrchestrator, AdaptiveWorkflowAdapter, int]:
    """Build a fresh per-task runtime on a newly reset baseline architecture."""
    manager = ArchitectureManager.create_default_architecture()
    workflow_adapter = AdaptiveWorkflowAdapter(manager=manager)
    architecture_adapter = LLMArchitectureAdapter(
        decision_client,
        workflow_adapter=workflow_adapter,
    )
    orchestrator = AdaptiveRuntimeOrchestrator(
        planner=planner_runner,
        architecture_adapter=architecture_adapter,
        agent_executor=agent_executor,
    )
    return orchestrator, workflow_adapter, workflow_adapter.current_version()


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_step36_task(
    task_name: str,
    task: str,
    *,
    planner_runner: PlannerRunner,
    decision_client: LLMClient,
    agent_executor: Optional[ExistingLLMAgentExecutor] = None,
) -> Step36TaskResult:
    """Run ONE task end to end on a freshly reset baseline architecture.

    The compiled graph is produced exclusively by the existing chain:
    PlannerOutput -> LLMArchitectureAdapter -> ArchitectureManager
    -> AdaptiveArchitecture -> AdaptiveWorkflowAdapter -> DynamicGraphBuilder
    -> graph.compile()
    """

    # Experiment control 1: SAME controlled baseline for every task. A new
    # manager/adaptive stack is created per task so state cannot leak.
    orchestrator, workflow_adapter, version_before = _orchestrator_for_task(
        planner_runner, decision_client, agent_executor
    )
    baseline = workflow_adapter.current_architecture()

    dynamic = orchestrator.run_dynamic(task)

    # Defensive control checks: every task must start from the unmodified
    # default baseline at version 0.
    if dynamic.initial_architecture.get("architecture_id") != baseline.architecture_id:
        raise AssertionError(
            "Step 36 control violated: task did not start from the default "
            f"baseline architecture (got {dynamic.initial_architecture.get('architecture_id')})"
        )
    if version_before != 0:
        raise AssertionError(
            "Step 36 control violated: baseline architecture version must "
            f"start at 0 (got {version_before})"
        )

    return _task_result_from_dynamic(task_name, task, dynamic, dynamic.architecture_decision)


def run_step36_experiment(
    *,
    planner_outputs: Optional[Dict[str, PlannerOutput]] = None,
    decisions: Optional[Dict[str, Dict[str, Any]]] = None,
    agent_executor: Optional[ExistingLLMAgentExecutor] = None,
    planner_factory: Optional[PlannerRunnerFactory] = None,
    decision_client: Optional[LLMClient] = None,
    tasks: Optional[Sequence[Tuple[str, str]]] = None,
) -> Step36ExperimentResult:
    """Run the controlled multi-task Step 36 experiment.

    Deterministic usage (tests / demo): pass ``planner_outputs`` and
    ``decisions`` keyed by task name. Fake planner/LLM instances are
    constructed per task, so no network access happens.

    Live usage: pass ``planner_factory`` (e.g. the existing
    ``Planner.from_settings``) and a settings-backed ``decision_client``.
    """

    selected_tasks = list(tasks if tasks is not None else EXPERIMENT_TASKS)
    deterministic = planner_outputs is not None and decisions is not None
    if not deterministic and planner_factory is None:
        raise ValueError(
            "run_step36_experiment requires either deterministic inputs "
            "(planner_outputs + decisions) or a live planner_factory"
        )

    results: List[Step36TaskResult] = []
    for task_name, task in selected_tasks:
        if deterministic:
            client: LLMClient = FakeDecisionLLM(decisions[task_name])  # type: ignore[index]
            runner: PlannerRunner = FakePlanner(planner_outputs[task_name], client)  # type: ignore[index]
            executor = agent_executor
        else:
            client = decision_client  # type: ignore[assignment]
            runner = planner_factory(model=client)  # type: ignore[misc]
            executor = agent_executor

        results.append(
            run_step36_task(
                task_name,
                task,
                planner_runner=runner,
                decision_client=client,
                agent_executor=executor,
            )
        )

    comparison = build_step36_comparison(results)
    return Step36ExperimentResult(
        tasks=results,
        baseline_signature=(
            graph_signature(
                next(
                    result.graph_edges
                    for result in results
                    if result.task_name == BASELINE_TASK_NAME
                )
            )
            if results
            else ""
        ),
        comparison=comparison,
    )


run_multi_task_experiment = run_step36_experiment


def _planner_outputs_for_tasks() -> Dict[str, PlannerOutput]:
    """Return deterministic PlannerOutput fixtures for the experiment tasks."""
    return {
        "conversation": PlannerOutput(
            task_understanding="Greet the assistant with a simple hello.",
            required_capabilities=[],
            steps=["Reply with a friendly greeting."],
            requires_research=False,
            requires_coding=False,
            requires_verification=False,
            estimated_complexity="trivial",
        ),
        "research": PlannerOutput(
            task_understanding=(
                "Research retrieval-augmented generation and summarize the "
                "main approaches."
            ),
            required_capabilities=["research", "information_synthesis"],
            steps=[
                "Survey RAG literature.",
                "Identify the main approaches.",
                "Summarize the findings.",
            ],
            requires_research=True,
            requires_coding=False,
            requires_verification=False,
            estimated_complexity="medium",
        ),
        "coding": PlannerOutput(
            task_understanding=(
                "Write a Java program that finds the second largest element "
                "in an array."
            ),
            required_capabilities=["coding"],
            steps=[
                "Design the single-pass algorithm.",
                "Implement the Java solution.",
                "Explain the approach.",
            ],
            requires_research=False,
            requires_coding=True,
            requires_verification=False,
            estimated_complexity="medium",
        ),
        "coding_verification": PlannerOutput(
            task_understanding=(
                "Write a Java program that finds the second largest element "
                "in an array and verify the edge cases."
            ),
            required_capabilities=["coding", "verification"],
            steps=[
                "Design the single-pass algorithm.",
                "Implement the Java solution.",
                "Verify the edge cases.",
            ],
            requires_research=False,
            requires_coding=True,
            requires_verification=True,
            estimated_complexity="medium",
        ),
    }


def _architecture_decisions_for_tasks() -> Dict[str, Dict[str, Any]]:
    """Return deterministic architecture decisions for the experiment tasks."""
    return {
        "conversation": {
            "decision": "no_change",
            "reasoning": "Simple conversation needs no specialist agents.",
            "actions": [],
        },
        "research": {
            "decision": "apply_actions",
            "reasoning": "Research requires the researcher agent.",
            "actions": [
                {"action_type": "activate_agent", "agent_id": "researcher"}
            ],
        },
        "coding": {
            "decision": "apply_actions",
            "reasoning": "Coding requires the coder agent.",
            "actions": [{"action_type": "activate_agent", "agent_id": "coder"}],
        },
        "coding_verification": {
            "decision": "apply_actions",
            "reasoning": "Coding with verification needs coder and critic.",
            "actions": [
                {"action_type": "activate_agent", "agent_id": "coder"},
                {"action_type": "activate_agent", "agent_id": "critic"},
            ],
        },
    }


def _run_one_deterministic_task(
    task_name: str,
    task: str,
    planner_output: PlannerOutput,
    decision: Dict[str, Any],
) -> Step36TaskResult:
    """Run one task with deterministic fake Planner and LLM clients."""
    client = FakeDecisionLLM(decision)
    runner = FakePlanner(planner_output, client)
    return run_step36_task(
        task_name,
        task,
        planner_runner=runner,
        decision_client=client,
        agent_executor=fake_agent_executor(),
    )


def run_step36_deterministic_experiment(
    tasks: Optional[Sequence[Tuple[str, str]]] = None,
) -> Step36ExperimentResult:
    """Run the full experiment with deterministic fake LLM responses.

    The planner outputs and architecture decisions below are INPUTS (what the
    Planner / architecture LLM would produce), not expected graphs. The
    compiled graphs are always derived at runtime by DynamicGraphBuilder.
    """
    selected_tasks = list(tasks if tasks is not None else EXPERIMENT_TASKS)
    planner_outputs_map = _planner_outputs_for_tasks()
    decisions_map = _architecture_decisions_for_tasks()

    results: List[Step36TaskResult] = []
    for task_name, task in selected_tasks:
        if task_name not in planner_outputs_map:
            raise AssertionError(
                f"Missing deterministic inputs for task '{task_name}'."
            )
        if task_name not in decisions_map:
            raise AssertionError(
                f"Missing deterministic decision for task '{task_name}'."
            )
        result = _run_one_deterministic_task(
            task_name,
            task,
            planner_outputs_map[task_name],
            decisions_map[task_name],
        )
        results.append(result)

    comparison = build_step36_comparison(results)
    return Step36ExperimentResult(
        tasks=results,
        baseline_signature=(
            graph_signature(
                next(
                    result.graph_edges
                    for result in results
                    if result.task_name == BASELINE_TASK_NAME
                )
            )
            if results
            else ""
        ),
        comparison=comparison,
    )


# ---------------------------------------------------------------------------
# Comparison report builder
# ---------------------------------------------------------------------------


def build_step36_comparison(
    results: List[Step36TaskResult],
) -> Step36ComparisonReport:
    """Build the cross-task comparison report and architecture metrics."""

    signatures = {
        result.task_name: graph_signature(result.graph_edges) for result in results
    }
    baseline_signature = signatures.get(BASELINE_TASK_NAME, "")

    differences: List[ArchitectureDifferenceMetric] = []
    for result in results:
        signature = signatures[result.task_name]
        differs_from_others = sorted(
            other.task_name
            for other in results
            if other.task_name != result.task_name
            and signatures[other.task_name] != signature
        )
        differences.append(
            ArchitectureDifferenceMetric(
                task_name=result.task_name,
                active_agent_count=len(result.final_active_agents),
                graph_node_count=len(result.graph_nodes),
                graph_edge_count=len(result.graph_edges),
                graph_signature=signature,
                differs_from_baseline=(
                    bool(baseline_signature) and signature != baseline_signature
                )
                or (
                    result.task_name == BASELINE_TASK_NAME
                    and bool(differs_from_others)
                ),
                baseline_graph_signature=baseline_signature,
                differs_from_other_tasks=differs_from_others,
            )
        )

    task_names = [result.task_name for result in results]
    all_differ_from_baseline = all(
        item.graph_signature != baseline_signature
        for item in differences
        if item.task_name != BASELINE_TASK_NAME
    )
    pairwise_comparisons = [
        signatures[first] != signatures[second]
        for index, first in enumerate(task_names)
        for second in task_names[index + 1 :]
    ]
    all_differ_from_each_other = (
        bool(pairwise_comparisons) and all(pairwise_comparisons)
    )

    baseline_result = next(
        (item for item in results if item.task_name == BASELINE_TASK_NAME), None
    )
    specialists_ran = any(
        len(result.execution_path) > 2 for result in results
    )
    baseline_is_minimal = (
        baseline_result is not None and len(baseline_result.execution_path) <= 2
    )
    if (
        all_differ_from_baseline
        and all_differ_from_each_other
        and baseline_is_minimal
        and specialists_ran
    ):
        conclusion = (
            "Different task requirements produce different compiled agent "
            "architectures and execution paths from the same initial MAS "
            "architecture."
        )
    else:
        conclusion = (
            "The experiment did NOT conclusively demonstrate that different "
            "task requirements produce different compiled architectures."
        )

    return Step36ComparisonReport(
        tasks=task_names,
        required_capabilities={
            result.task_name: result.required_capabilities for result in results
        },
        actions={
            result.task_name: [
                "{action_type}{detail}".format(
                    action_type=action.get("action_type", ""),
                    detail=(
                        f":{action.get('agent_id') or action.get('target') or ''}"
                        if action.get("agent_id") or action.get("target")
                        else ""
                    ),
                )
                for action in result.accepted_actions
            ]
            or (
                ["no_change"]
                if result.architecture_decision.get("decision") == "no_change"
                else []
            )
            for result in results
        },
        final_agents={
            result.task_name: result.final_active_agents for result in results
        },
        compiled_edges={
            result.task_name: [
                f"{edge['source']}->{edge['target']}"
                for edge in sorted(
                    result.graph_edges,
                    key=lambda edge: (edge["source"], edge["target"]),
                )
            ]
            for result in results
        },
        execution_paths={
            result.task_name: result.execution_path for result in results
        },
        graph_signatures=signatures,
        differences=differences,
        all_graphs_differ_from_baseline=all_differ_from_baseline,
        all_graphs_differ_from_each_other=all_differ_from_each_other,
        conclusion=conclusion,
    )


# ---------------------------------------------------------------------------
# Optional PNG export (never required; failure-tolerant)
# ---------------------------------------------------------------------------


def render_step36_graph_images(
    result: Step36ExperimentResult,
    output_dir: str = "step36_graphs",
) -> List[str]:
    """Best-effort export of each task's graph.

    Always writes the Mermaid source (``.mmd``). When the optional ``graphviz``
    package AND the Graphviz ``dot`` executable are available, also renders a
    PNG from the compiled topology. PNG rendering is OPTIONAL: this function
    never raises for missing rendering support and simply returns whatever it
    managed to write.
    """

    written: List[str] = []
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError:
        return written

    dot_available = False
    try:  # pragma: no cover - depends on optional rendering stack
        import graphviz  # type: ignore[import-not-found]

        graphviz.Digraph().pipe(format="png")
        dot_available = True
    except Exception:  # noqa: BLE001 - any failure means "not available"
        dot_available = False

    for task_result in result.tasks:
        if not task_result.graph_representation:
            continue
        mermaid_path = os.path.join(
            output_dir, f"step36_{task_result.task_name}.mmd"
        )
        try:
            with open(mermaid_path, "w", encoding="utf-8") as handle:
                handle.write(task_result.graph_representation)
            written.append(mermaid_path)
        except OSError:  # pragma: no cover - best effort only
            continue

        if not dot_available:
            continue
        png_path = os.path.join(output_dir, f"step36_{task_result.task_name}.png")
        try:  # pragma: no cover - depends on optional rendering stack
            import graphviz  # type: ignore[import-not-found]

            digraph = graphviz.Digraph(
                name=f"step36_{task_result.task_name}", format="png"
            )
            for node in task_result.graph_nodes:
                digraph.node(node)
            for edge in task_result.graph_edges:
                digraph.edge(edge["source"], edge["target"])
            digraph.attr("graph", rankdir="LR")
            digraph.render(png_path, cleanup=True)
            written.append(png_path)
        except Exception:  # noqa: BLE001 - best effort only
            continue
    return written


# ---------------------------------------------------------------------------
# Bounded live experiment entry point
# ---------------------------------------------------------------------------


class LiveExperimentUnavailable(BaseModel):
    """Serializable outcome when the live experiment cannot run."""

    status: str = "unavailable"
    reason: str

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


TIMEOUT_ERROR_PATTERNS = (
    "timeout",
    "timed out",
    "connection error",
    "connection refused",
    "connection reset",
    "api connection",
    "read error",
    "rate limit",
    "authentication",
    "401",
    "403",
    "429",
    "500",
    "502",
    "503",
    "504",
    "service unavailable",
    "bad gateway",
    "internal server error",
    "overloaded",
)


def is_provider_unavailable_error(exc: BaseException) -> bool:
    """Return True when *exc* indicates the live provider is unavailable."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(pattern in text for pattern in TIMEOUT_ERROR_PATTERNS)


def run_step36_live(
    tasks: Optional[Sequence[Tuple[str, str]]] = None,
) -> Any:
    """Run the bounded live experiment using the existing NVIDIA configuration.

    Uses the existing settings-backed Planner and the existing
    ``LLMArchitectureAdapter.from_settings()`` client; agents run through the
    existing ``ExistingLLMAgentExecutor`` default factories. Every task still
    starts from a freshly reset default baseline architecture. Provider
    timeout/failure is reported as a ``LiveExperimentUnavailable`` outcome
    rather than raising, so provider outage is never a test failure.
    """

    selected_tasks = list(tasks if tasks is not None else EXPERIMENT_TASKS)

    if not settings.api_key:
        return LiveExperimentUnavailable(
            reason="LLM API key is not configured (NVIDIA_API_KEY)."
        )
    if not settings.model:
        return LiveExperimentUnavailable(
            reason="LLM model is not configured (NVIDIA_MODEL)."
        )

    try:
        planner = Planner.from_settings()
        settings_adapter = LLMArchitectureAdapter.from_settings()
    except Exception as exc:  # noqa: BLE001 - config problems are unavailability
        return LiveExperimentUnavailable(reason=f"configuration unavailable: {exc}")

    results: List[Step36TaskResult] = []
    try:
        for task_name, task in selected_tasks:
            orchestrator, _, _ = _orchestrator_for_task(
                planner, settings_adapter.client, ExistingLLMAgentExecutor()
            )
            dynamic = orchestrator.run_dynamic(task)
            results.append(
                _task_result_from_dynamic(
                    task_name, task, dynamic, dynamic.architecture_decision
                )
            )
    except Exception as exc:
        if is_provider_unavailable_error(exc):
            return LiveExperimentUnavailable(
                reason=f"live provider unavailable: {type(exc).__name__}: {exc}"
            )
        raise

    comparison = build_step36_comparison(results)
    return Step36ExperimentResult(
        tasks=results,
        baseline_signature=(
            graph_signature(
                next(
                    result.graph_edges
                    for result in results
                    if result.task_name == BASELINE_TASK_NAME
                )
            )
            if results
            else ""
        ),
        comparison=comparison,
    )


def run_step36_deterministic_cli_demo(
    tasks: Optional[Sequence[Tuple[str, str]]] = None,
) -> None:
    """Print each task's section with its Mermaid graph separately.

    Uses deterministic fake planner/LLM responses so the demo is repeatable
    without network access or credentials.
    """
    selected_tasks = list(tasks if tasks is not None else EXPERIMENT_TASKS)
    planner_outputs = _planner_outputs_for_tasks()
    decisions = _architecture_decisions_for_tasks()

    print("STEP 36 DETERMINISTIC DYNAMIC-GRAPH EVALUATION DEMO")
    print("=" * 60)
    print()

    for task_name, task in selected_tasks:
        planner_output = planner_outputs[task_name]
        decision = decisions[task_name]

        result = _run_one_deterministic_task(
            task_name, task, planner_output, decision
        )

        print(f"========== TASK: {task_name.upper()} ==========")
        print()
        print("TASK:")
        print(result.task)
        print()
        print("PLANNER:")
        print(
            f"  task_understanding: {result.planner_output.get('task_understanding')}"
        )
        print(
            f"  required_capabilities: {', '.join(result.required_capabilities) or '(none)'}"
        )
        print(f"  requires_research: {planner_output.requires_research}")
        print(f"  requires_coding: {planner_output.requires_coding}")
        print(f"  requires_verification: {planner_output.requires_verification}")
        print(f"  estimated_complexity: {planner_output.estimated_complexity}")
        print()
        print("ARCHITECTURE DECISION:")
        print(f"  decision: {result.architecture_decision.get('decision')}")
        print(f"  reasoning: {result.architecture_decision.get('reasoning')}")
        print(f"  proposed_actions: {len(result.proposed_actions)}")
        print(f"  accepted_actions: {len(result.accepted_actions)}")
        print(f"  rejected_actions: {len(result.rejected_actions)}")
        print()
        print("INITIAL ACTIVE AGENTS:")
        print(f"  {', '.join(result.initial_active_agents)}")
        print()
        print("FINAL ACTIVE AGENTS:")
        print(f"  {', '.join(result.final_active_agents)}")
        print()
        print(f"ARCHITECTURE CHANGED: {result.architecture_changed}")
        print(f"ARCHITECTURE VERSION: {result.architecture_version}")
        print()
        print("COMPILED GRAPH NODES:")
        print(f"  {', '.join(result.graph_nodes)}")
        print()
        print("COMPILED GRAPH EDGES:")
        for edge in result.graph_edges:
            print(f"  {edge['source']} -> {edge['target']}")
        print()
        print("COMPILED GRAPH REPRESENTATION (Mermaid):")
        for line in result.graph_representation.splitlines():
            print(f"    {line}")
        print()
        print("EXECUTION PATH:")
        print(f"  {' -> '.join(result.execution_path)}")
        print()
        print("EXECUTION TRACE:")
        for event in result.execution_trace:
            node_str = f" node={event.get('node')!r}" if event.get('node') else ""
            print(
                f"  seq={event.get('sequence')} event={event.get('event')}{node_str}"
                f" version={event.get('architecture_version')} stage={event.get('architecture_stage')} status={event.get('status')}"
            )
        print()
        print("FINAL RESPONSE:")
        print(f"  {result.final_response}")
        print()
        print("-" * 60)
        print()


# ---------------------------------------------------------------------------
# Responsibility-based aliases for backward compatibility and clean architecture
# ---------------------------------------------------------------------------

MultiTaskTaskResult = Step36TaskResult
MultiTaskComparisonReport = Step36ComparisonReport
MultiTaskExperimentResult = Step36ExperimentResult
build_multi_task_comparison = build_step36_comparison
run_multi_task_evaluation = run_step36_deterministic_experiment
run_multi_task_cli_demo = run_step36_deterministic_cli_demo
run_multi_task_live = run_step36_live
run_multi_task_task = run_step36_task


if __name__ == "__main__":  # pragma: no cover - manual smoke only
    run_step36_deterministic_cli_demo()
    print()
    experiment = run_step36_deterministic_experiment()
    if experiment.comparison:
        print(experiment.comparison.render_table())
        print()
        print("CONCLUSION:", experiment.comparison.conclusion)
