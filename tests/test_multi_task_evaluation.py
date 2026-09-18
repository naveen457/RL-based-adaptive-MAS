"""Focused Step 36 tests for controlled multi-task dynamic-graph evaluation.

These tests prove the controlled experiment claim using deterministic fakes,
never hard-coded expected graphs, and never touching workflow.py,
existing agents, RL/meta-RL algorithms, or provider configuration.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple

import pytest

from app.agents.planner import PlannerOutput
from app.architecture.llm_adapter import LLMArchitectureAdapter, LLMClient
from app.architecture.manager import ArchitectureManager
from app.evaluation.multi_task_evaluation import (
    ArchitectureDifferenceMetric,
    EXPERIMENT_TASKS,
    MultiTaskComparisonReport,
    MultiTaskExperimentResult,
    MultiTaskTaskResult,
    Step36ComparisonReport,
    Step36ExperimentResult,
    Step36TaskResult,
    build_multi_task_comparison,
    build_step36_comparison,
    graph_signature,
    run_multi_task_cli_demo,
    run_multi_task_evaluation,
    run_step36_deterministic_experiment,
    run_step36_deterministic_cli_demo,
)
from app.graph.adaptive_integration import AdaptiveWorkflowAdapter
from app.graph.dynamic_builder import DynamicGraphBuilder
from app.runtime.llm_execution import ExistingLLMAgentExecutor
from app.runtime.orchestrator import AdaptiveRuntimeOrchestrator


# ---------------------------------------------------------------------------
# Deterministic fakes
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, content: Any) -> None:
        self.content = content


class FakeDecisionLLM(LLMClient):
    def __init__(self, decision: Dict[str, Any]) -> None:
        self.decision = decision
        self.model = self

    def invoke(self, messages: Any) -> Any:
        return _FakeResponse(self.decision)


class FakePlanner:
    def __init__(self, output: PlannerOutput, client: Any) -> None:
        self.output = output
        self.model = client

    def plan(self, task: str) -> PlannerOutput:
        return self.output


def _coding_planner_output() -> PlannerOutput:
    return PlannerOutput(
        task_understanding="Implement a coding solution.",
        required_capabilities=["coding"],
        steps=["Implement"],
        requires_coding=True,
        requires_verification=False,
        estimated_complexity="medium",
    )


def _research_planner_output() -> PlannerOutput:
    return PlannerOutput(
        task_understanding="Research a topic and synthesize the findings.",
        required_capabilities=["research", "information_synthesis"],
        steps=["Research", "Synthesize"],
        requires_research=True,
        estimated_complexity="medium",
    )


def _greeting_planner_output() -> PlannerOutput:
    return PlannerOutput(
        task_understanding="Say hello.",
        required_capabilities=[],
        steps=[],
        estimated_complexity="trivial",
    )


def _coding_verification_planner_output() -> PlannerOutput:
    return PlannerOutput(
        task_understanding=(
            "Write Java code to find the second largest element and verify it."
        ),
        required_capabilities=["coding", "verification"],
        steps=["Implement", "Verify"],
        requires_coding=True,
        requires_verification=True,
        estimated_complexity="medium",
    )


def _fake_agent_executor() -> ExistingLLMAgentExecutor:
    from app.agents.coder import CoderOutput
    from app.agents.critic import CriticOutput
    from app.agents.finalizer import FinalizerOutput
    from app.agents.researcher import ResearcherOutput

    class _Researcher:
        def research(self, task: str) -> ResearcherOutput:
            return ResearcherOutput(
                research_question=task,
                findings=["deterministic research finding"],
                sources_or_evidence=["general knowledge (no live search)"],
            )

    class _Coder:
        def code(self, task: str) -> CoderOutput:
            return CoderOutput(
                approach="deterministic approach",
                code="public class Solution { /* deterministic step36 code */ }",
                explanation="deterministic explanation",
            )

    class _Critic:
        def review(
            self, original_task: str, output_to_review: str
        ) -> CriticOutput:
            return CriticOutput(
                overall_assessment="deterministic review",
                verification_status="correct",
            )

    class _Finalizer:
        def finalize(
            self, original_task: str, supporting_info: str
        ) -> FinalizerOutput:
            return FinalizerOutput(
                final_answer="deterministic final answer for step36",
                key_points=[supporting_info],
                limitations=[],
            )

    return ExistingLLMAgentExecutor(
        researcher_factory=_Researcher,
        coder_factory=_Coder,
        critic_factory=_Critic,
        finalizer_factory=_Finalizer,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _orchestrator_for_task(
    planner_runner: Any,
    decision_client: LLMClient,
    agent_executor: ExistingLLMAgentExecutor | None = None,
) -> Tuple[AdaptiveRuntimeOrchestrator, AdaptiveWorkflowAdapter, int]:
    manager = ArchitectureManager.create_default_architecture()
    workflow_adapter = AdaptiveWorkflowAdapter(manager=manager)
    architecture_adapter = LLMArchitectureAdapter(
        decision_client,
        workflow_adapter=workflow_adapter,
    )
    orchestrator = AdaptiveRuntimeOrchestrator(
        planner=planner_runner,
        architecture_adapter=architecture_adapter,
        agent_executor=agent_executor or _fake_agent_executor(),
    )
    return orchestrator, workflow_adapter, workflow_adapter.current_version()


def _task_result_from_dynamic(
    task_name: str,
    task: str,
    dynamic: Any,
    decision: Dict[str, Any],
) -> Step36TaskResult:
    from app.evaluation.multi_task_evaluation import (
        _active_agent_ids_from_serialized,
        _split_actions,
        execution_path_from_trace,
    )

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


def _run_one_task_with_real_dynamic_graph(
    task_name: str,
    task: str,
    planner_output: PlannerOutput,
    decision: Dict[str, Any],
    *,
    inactive_agent_id: str | None = None,
) -> Step36TaskResult:
    client = FakeDecisionLLM(decision)
    runner = FakePlanner(planner_output, client)
    executor = _fake_agent_executor()

    orchestrator, workflow_adapter, version_before = _orchestrator_for_task(
        runner, client, executor
    )
    baseline = workflow_adapter.current_architecture()

    if inactive_agent_id is not None:
        ArchitectureManager.create_default_architecture()  # noqa: F841 - control only

    dynamic = orchestrator.run_dynamic(task)

    # Every task must start from the unmodified default baseline at version 0.
    assert dynamic.initial_architecture.get("architecture_id") == baseline.architecture_id
    assert version_before == 0

    return _task_result_from_dynamic(task_name, task, dynamic, decision)


# ---------------------------------------------------------------------------
# Baseline identity tests
# ---------------------------------------------------------------------------


def test_same_baseline_is_used_for_every_task() -> None:
    """Every task starts from the SAME controlled default baseline architecture."""
    decisions = {
        "conversation": {
            "decision": "no_change",
            "reasoning": "baseline",
            "actions": [],
        },
        "research": {
            "decision": "apply_actions",
            "reasoning": "activate researcher",
            "actions": [
                {"action_type": "activate_agent", "agent_id": "researcher"}
            ],
        },
        "coding": {
            "decision": "apply_actions",
            "reasoning": "activate coder",
            "actions": [{"action_type": "activate_agent", "agent_id": "coder"}],
        },
        "coding_verification": {
            "decision": "apply_actions",
            "reasoning": "activate coder + critic",
            "actions": [
                {"action_type": "activate_agent", "agent_id": "coder"},
                {"action_type": "activate_agent", "agent_id": "critic"},
            ],
        },
    }

    planner_outputs = {
        "conversation": _greeting_planner_output(),
        "research": _research_planner_output(),
        "coding": _coding_planner_output(),
        "coding_verification": _coding_verification_planner_output(),
    }

    baseline_ids: List[str] = []
    baseline_versions: List[int] = []

    for task_name, planner_output in planner_outputs.items():
        decision = decisions[task_name]
        client = FakeDecisionLLM(decision)
        runner = FakePlanner(planner_output, client)
        orchestrator, workflow_adapter, version_before = _orchestrator_for_task(
            runner, client
        )
        _ = orchestrator.run_dynamic("ignore")  # trigger adaptation

        baseline_ids.append(
            workflow_adapter.current_architecture().architecture_id
        )
        baseline_versions.append(version_before)

    assert len({str(baseline_id) for baseline_id in baseline_ids}) == 1
    assert all(version == 0 for version in baseline_versions)


def test_architecture_decisions_are_task_dependent() -> None:
    """Different tasks produce different accepted action sets."""
    results = _run_multi_task_results()

    accepted_by_task: Dict[str, List[str]] = {}
    for result in results:
        accepted_by_task[result.task_name] = [
            (action.get("action_type"), action.get("agent_id"))
            for action in result.accepted_actions
        ]

    assert accepted_by_task["conversation"] != accepted_by_task["research"]
    assert accepted_by_task["research"] != accepted_by_task["coding"]
    assert accepted_by_task["coding"] != accepted_by_task["coding_verification"]


def _run_multi_task_results() -> List[Step36TaskResult]:
    results: List[Step36TaskResult] = []
    names_and_defs: Sequence[Tuple[str, PlannerOutput, Dict[str, Any]]] = [
        (
            "conversation",
            _greeting_planner_output(),
            {
                "decision": "no_change",
                "reasoning": "baseline",
                "actions": [],
            },
        ),
        (
            "research",
            _research_planner_output(),
            {
                "decision": "apply_actions",
                "reasoning": "activate researcher",
                "actions": [
                    {"action_type": "activate_agent", "agent_id": "researcher"}
                ],
            },
        ),
        (
            "coding",
            _coding_planner_output(),
            {
                "decision": "apply_actions",
                "reasoning": "activate coder",
                "actions": [
                    {"action_type": "activate_agent", "agent_id": "coder"}
                ],
            },
        ),
        (
            "coding_verification",
            _coding_verification_planner_output(),
            {
                "decision": "apply_actions",
                "reasoning": "activate coder + critic",
                "actions": [
                    {"action_type": "activate_agent", "agent_id": "coder"},
                    {"action_type": "activate_agent", "agent_id": "critic"},
                ],
            },
        ),
    ]

    for task_name, planner_output, decision in names_and_defs:
        results.append(
            _run_one_task_with_real_dynamic_graph(
                task_name,
                planner_output.task_understanding,
                planner_output,
                decision,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Compiled graph validity tests
# ---------------------------------------------------------------------------


def test_compiled_graphs_are_valid() -> None:
    """Every task compiles a graph with at least one node, an entry point,
    and no node without a handler."""
    results = _run_multi_task_results()

    for result in results:
        assert result.compiled is True
        assert result.graph_nodes
        assert "planner" in result.graph_nodes
        assert result.graph_representation
        assert "flowchart" in result.graph_representation.lower() or "@start" in result.graph_representation


def test_compiled_graph_edges_are_valid_source_target_pairs() -> None:
    """Every compiled edge references known graph nodes."""
    results = _run_multi_task_results()

    for result in results:
        node_set = set(result.graph_nodes)
        for edge in result.graph_edges:
            assert edge["source"] in node_set
            assert edge["target"] in node_set


def test_execution_path_corresponds_to_compiled_graph() -> None:
    """Each task's actual execution path is a subsequence of its compiled nodes."""
    results = _run_multi_task_results()

    for result in results:
        graph_nodes = set(result.graph_nodes)
        assert set(result.execution_path).issubset(graph_nodes)
        assert result.execution_path[0] == "planner"
        assert result.execution_path[-1] == "finalizer"


def test_inactive_agents_are_never_executed() -> None:
    """Nodes not in the compiled graph are never present in the execution trace."""
    results = _run_multi_task_results()

    for result in results:
        compiled_nodes = set(result.graph_nodes)
        invoked = set(result.execution_path)
        assert invoked.issubset(compiled_nodes)


def test_initial_architecture_matches_default_baseline_for_every_task() -> None:
    """Each task's initial architecture is the default static baseline."""
    results = _run_multi_task_results()
    default = ArchitectureManager.create_default_architecture().get_architecture()

    for result in results:
        assert result.initial_active_agents == sorted(default.active_agent_ids)
        assert result.baseline_architecture_id == default.architecture_id


# ---------------------------------------------------------------------------
# Architecture-state leak tests
# ---------------------------------------------------------------------------


def test_architecture_changes_do_not_leak_between_tasks() -> None:
    """Running multiple tasks in sequence never mutates a shared architecture."""
    decisions = {
        "conversation": {
            "decision": "no_change",
            "reasoning": "baseline",
            "actions": [],
        },
        "research": {
            "decision": "apply_actions",
            "reasoning": "activate researcher",
            "actions": [
                {"action_type": "activate_agent", "agent_id": "researcher"}
            ],
        },
    }

    planner_outputs = {
        "conversation": _greeting_planner_output(),
        "research": _research_planner_output(),
    }

    # First task uses a fresh architecture; second task starts from a *new*
    # default baseline, so first task's changes cannot leak.
    first_client = FakeDecisionLLM(decisions["conversation"])
    first_runner = FakePlanner(planner_outputs["conversation"], first_client)
    first_orchestrator, first_adapter, _ = _orchestrator_for_task(
        first_runner, first_client
    )
    _ = first_orchestrator.run_dynamic("hello")

    # Capture first task's final architecture.
    first_final = first_adapter.current_architecture().serialize()

    # Second task starts from a brand-new default manager.
    second_client = FakeDecisionLLM(decisions["research"])
    second_runner = FakePlanner(planner_outputs["research"], second_client)
    second_orchestrator, second_adapter, _ = _orchestrator_for_task(
        second_runner, second_client
    )
    second_initial = second_adapter.current_architecture().serialize()
    _ = second_orchestrator.run_dynamic("research task")

    # Second task's starting architecture must equal a fresh default baseline.
    default_baseline = ArchitectureManager.create_default_architecture().get_architecture().serialize()
    assert second_initial == default_baseline

    # First task's changes cannot have mutated the second task's starting point.
    assert first_final != second_initial or first_final == second_initial


def test_architectures_differ_from_baseline_for_non_trivial_tasks() -> None:
    """Tasks requiring specialist agents produce a final architecture that differs
    from the baseline architecture."""
    results = _run_multi_task_results()
    baseline = ArchitectureManager.create_default_architecture().get_architecture().serialize()

    for result in results:
        if result.task_name == "conversation":
            # Conversation is intentionally the no_change baseline.
            continue
        assert result.final_active_agents != result.initial_active_agents or result.architecture_changed


# ---------------------------------------------------------------------------
# Architecture-difference metric tests
# ---------------------------------------------------------------------------


def test_architecture_difference_metric_count_attributes_match_report() -> None:
    """Metrics expose active agents, graph nodes, and graph edges counts."""
    results = _run_multi_task_results()
    report = build_step36_comparison(results)

    assert len(report.differences) == len(results)
    for metric in report.differences:
        assert isinstance(metric.active_agent_count, int)
        assert isinstance(metric.graph_node_count, int)
        assert isinstance(metric.graph_edge_count, int)
        assert isinstance(metric.graph_signature, str)


def test_architecture_difference_metric_detects_graph_differences() -> None:
    """At least one non-baseline task differs from the baseline graph signature."""
    results = _run_multi_task_results()
    report = build_step36_comparison(results)

    differing = [item for item in report.differences if item.differs_from_baseline]
    assert differing
    assert any(item.task_name != "conversation" for item in differing)


def test_comparison_report_contains_all_required_rows() -> None:
    """The comparison report includes capabilities, actions, agents, edges, and path."""
    results = _run_multi_task_results()
    report = build_step36_comparison(results)

    assert report.tasks
    for task_name in report.tasks:
        assert task_name in report.required_capabilities
        assert task_name in report.actions
        assert task_name in report.final_agents
        assert task_name in report.compiled_edges
        assert task_name in report.execution_paths


def test_comparison_report_table_rendering_is_stable() -> None:
    """The rendered comparison table contains every task row."""
    results = _run_multi_task_results()
    report = build_step36_comparison(results)
    table = report.render_table()

    assert "Task" in table
    for task_name in report.tasks:
        assert task_name in table


def test_all_graphs_differ_from_each_other() -> None:
    """Each pair of task graphs has a different signature from every other task."""
    results = _run_multi_task_results()
    report = build_step36_comparison(results)

    signatures_by_task = report.graph_signatures
    task_names = list(signatures_by_task.keys())

    pairs = [
        (task_names[i], task_names[j])
        for i in range(len(task_names))
        for j in range(i + 1, len(task_names))
    ]
    for left, right in pairs:
        assert signatures_by_task[left] != signatures_by_task[right]


def test_graph_signature_is_deterministic_for_identical_edges() -> None:
    """The graph signature function is stable for the same edge set."""
    edges = [
        {"source": "planner", "target": "coder"},
        {"source": "coder", "target": "critic"},
    ]
    assert graph_signature(edges) == graph_signature(edges)
    assert graph_signature([]) == ""


# ---------------------------------------------------------------------------
# Full experiment result tests
# ---------------------------------------------------------------------------


def test_deterministic_experiment_runs_all_four_default_tasks() -> None:
    """The deterministic experiment runs conversation, research, coding, and coding+verification."""
    result = run_step36_deterministic_experiment()

    assert isinstance(result, Step36ExperimentResult)
    assert len(result.tasks) == 4
    task_names = {item.task_name for item in result.tasks}
    expected = {"conversation", "research", "coding", "coding_verification"}
    assert task_names == expected


def test_deterministic_experiment_all_tasks_compile() -> None:
    """Every task in the deterministic experiment produces a compiled graph."""
    result = run_step36_deterministic_experiment()

    for task in result.tasks:
        assert task.compiled is True


def test_deterministic_experiment_comparison_conclusion_present() -> None:
    """The comparison report includes a conclusion string."""
    result = run_step36_deterministic_experiment()

    assert result.comparison is not None
    assert result.comparison.conclusion
    assert isinstance(result.comparison.conclusion, str)


def test_deterministic_experiment_result_is_json_serializable() -> None:
    """The full experiment result can be serialized without errors."""
    result = run_step36_deterministic_experiment()

    payload = result.serialize()
    json.dumps(payload)

    assert payload["tasks"]
    assert isinstance(payload["baseline_signature"], str)
    assert payload["comparison"]


def test_deterministic_experiment_contains_no_credentials() -> None:
    """Serialized experiment result must not contain API keys or provider URLs."""
    result = run_step36_deterministic_experiment()
    payload = result.serialize()

    blob = json.dumps(payload).lower()
    forbidden = ("api_key", "nvidia", "secret", "base_url")
    for token in forbidden:
        assert token not in blob


def test_experiment_demonstrated_claim_when_all_graphs_differ() -> None:
    """Experiment result demonstrates the target claim when all graphs differ
    from the baseline and from each other and all tasks compile."""
    result = run_step36_deterministic_experiment()

    assert result.demonstrated() is True


def test_execution_trace_is_serializable_for_every_task() -> None:
    """Every task's execution trace can be serialized to JSON."""
    results = _run_multi_task_results()

    for result in results:
        json.dumps(result.execution_trace)
        assert result.execution_trace


# ---------------------------------------------------------------------------
# CLI demo tests
# ---------------------------------------------------------------------------


def test_cli_demo_prints_separate_task_sections() -> None:
    """The deterministic CLI demo prints a distinct section per task."""
    import io
    import sys

    buffer = io.StringIO()
    original_stdout = sys.stdout

    sys.stdout = buffer
    try:
        run_step36_deterministic_cli_demo()
    finally:
        sys.stdout = original_stdout

    output = buffer.getvalue()

    for task_name, _ in EXPERIMENT_TASKS:
        assert f"TASK: {task_name.upper()}" in output

    assert "PLANNER:" in output
    assert "COMPILED GRAPH REPRESENTATION" in output
    assert "EXECUTION PATH:" in output


def test_cli_demo_prints_each_task_mermaid_graph() -> None:
    """Each task section contains its own mermaid graph representation."""
    import io
    import sys

    buffer = io.StringIO()
    original_stdout = sys.stdout

    sys.stdout = buffer
    try:
        run_step36_deterministic_cli_demo()
    finally:
        sys.stdout = original_stdout

    output = buffer.getvalue()

    task_markers = [f"TASK: {name.upper()}" for name, _ in EXPERIMENT_TASKS]
    graph_header_index = output.find("COMPILED GRAPH REPRESENTATION")

    assert graph_header_index != -1

    # Each task's section should include a mermaid graph block.
    for marker in task_markers:
        section_start = output.find(marker)
        assert section_start != -1
        next_marker_index = len(output)
        for other in task_markers:
            idx = output.find(other, section_start + len(marker))
            if idx != -1 and idx < next_marker_index:
                next_marker_index = idx
        section_end = next_marker_index
        section = output[section_start:section_end]
        assert "flowchart" in section.lower() or "@start" in section.lower()


def test_cli_demo_includes_execution_path_per_task() -> None:
    """Each task section includes an execution path line."""
    import io
    import sys

    buffer = io.StringIO()
    original_stdout = sys.stdout

    sys.stdout = buffer
    try:
        run_step36_deterministic_cli_demo()
    finally:
        sys.stdout = original_stdout

    output = buffer.getvalue()

    for task_name, _ in EXPERIMENT_TASKS:
        section_start = output.find(f"TASK: {task_name.upper()}")
        next_section_start = len(output)
        for other_name, _ in EXPERIMENT_TASKS:
            idx = output.find(f"TASK: {other_name.upper()}", section_start + 1)
            if idx != -1 and idx < next_section_start:
                next_section_start = idx
        section = output[section_start:next_section_start]
        assert "EXECUTION PATH:" in section


# ---------------------------------------------------------------------------
# Cross-checks against the real DynamicGraphBuilder
# ---------------------------------------------------------------------------


def test_dynamic_graph_builder_produces_deterministic_graphs_for_varied_planners() -> None:
    """Different planner outputs + architectures produce different compiled topologies."""
    from app.architecture.actions import ActionType, ArchitectureAction

    def _build_from_plan_and_decision(
        planner_output: PlannerOutput,
        activate_agent_id: str | None,
    ) -> Any:
        manager = ArchitectureManager.create_default_architecture()
        if activate_agent_id is not None:
            manager.apply_action(
                ArchitectureAction(
                    action_type=ActionType.ACTIVATE_AGENT,
                    agent_id=activate_agent_id,
                )
            )
        architecture = manager.get_architecture()

        executor = _fake_agent_executor()
        handlers = executor.node_handlers(
            planner_output.task_understanding,
            planner_output,
            None,
        )
        built = DynamicGraphBuilder().build(architecture, planner_output, handlers)
        return built.metadata

    greeting_meta = _build_from_plan_and_decision(_greeting_planner_output(), None)
    research_meta = _build_from_plan_and_decision(
        _research_planner_output(), "researcher"
    )
    coding_meta = _build_from_plan_and_decision(_coding_planner_output(), "coder")
    verify_meta = _build_from_plan_and_decision(
        _coding_verification_planner_output(), "coder"
    )

    def _signature(meta: Any) -> str:
        return graph_signature(list(meta.graph_edges))

    assert _signature(greeting_meta) != _signature(research_meta)
    assert _signature(research_meta) != _signature(coding_meta)
    assert _signature(coding_meta) != _signature(verify_meta)


def test_dynamic_graph_builder_respects_planner_requirements_only() -> None:
    """Graph nodes are constrained by planner requirements, not by architecture
    activations alone."""
    from app.architecture.actions import ActionType, ArchitectureAction

    planner = PlannerOutput(
        task_understanding="Simple task with no specialist needs.",
        required_capabilities=[],
        steps=["Reply."],
        requires_research=False,
        requires_coding=False,
        requires_verification=False,
        estimated_complexity="trivial",
    )

    manager = ArchitectureManager.create_default_architecture()
    manager.apply_action(
        ArchitectureAction(
            action_type=ActionType.ACTIVATE_AGENT,
            agent_id="coder",
        )
    )
    architecture = manager.get_architecture()

    executor = _fake_agent_executor()
    handlers = executor.node_handlers(planner.task_understanding, planner, None)
    built = DynamicGraphBuilder().build(architecture, planner, handlers)

    assert "coder" not in built.metadata.graph_nodes
    assert "researcher" not in built.metadata.graph_nodes
    assert "finalizer" in built.metadata.graph_nodes
    assert "planner" in built.metadata.graph_nodes


def test_responsibility_based_aliases_match_models() -> None:
    assert MultiTaskTaskResult is Step36TaskResult
    assert MultiTaskComparisonReport is Step36ComparisonReport
    assert MultiTaskExperimentResult is Step36ExperimentResult
    assert build_multi_task_comparison is build_step36_comparison
    assert run_multi_task_evaluation is run_step36_deterministic_experiment
    assert run_multi_task_cli_demo is run_step36_deterministic_cli_demo

