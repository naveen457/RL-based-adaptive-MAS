import pytest
from unittest.mock import MagicMock

from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.manager import ArchitectureManager
from app.architecture.models import MASArchitecture
from app.agents.planner import PlannerOutput
from app.evaluation.theoretical_evaluator import TheoreticalArchitectureEvaluator
from app.runtime.llm_execution import ExistingLLMAgentExecutor


def test_tool_executor_edges_auto_connected_on_activation():
    """Activating tool_executor automatically connects planner->tool_executor and tool_executor->finalizer."""
    mgr = ArchitectureManager.create_default_architecture()
    action = ArchitectureAction(action_type=ActionType.ACTIVATE_AGENT, agent_id="tool_executor")
    updated = mgr.apply_action(action)

    tool_agent = [a for a in updated.agents if a.agent_id == "tool_executor"][0]
    assert tool_agent.active is True

    edge_tuples = [(e.source, e.target) for e in updated.communication_edges]
    assert ("planner", "tool_executor") in edge_tuples
    assert ("tool_executor", "finalizer") in edge_tuples


def test_tool_executor_evaluates_as_optimal_not_malformed():
    """Topology with planner -> tool_executor -> finalizer must be classified as OPTIMAL, not MALFORMED."""
    mgr = ArchitectureManager.create_default_architecture()
    mgr.apply_action(ArchitectureAction(action_type=ActionType.ACTIVATE_AGENT, agent_id="tool_executor"))
    mgr.apply_action(ArchitectureAction(action_type=ActionType.DEACTIVATE_AGENT, agent_id="researcher"))
    mgr.apply_action(ArchitectureAction(action_type=ActionType.DEACTIVATE_AGENT, agent_id="coder"))
    updated = mgr.apply_action(ArchitectureAction(action_type=ActionType.DEACTIVATE_AGENT, agent_id="critic"))

    evaluator = TheoreticalArchitectureEvaluator()
    eval_res = evaluator.evaluate(updated, required_capabilities=["web_search"])

    assert eval_res.pareto.classification == "optimal"
    assert len(eval_res.topology.orphan_nodes) == 0
    assert len(eval_res.topology.dead_end_nodes) == 0
    assert eval_res.pareto.net_utility > 0.7


def test_tool_disambiguation_prunes_date_on_trends_query():
    """Queries for 'current trends' must execute web_search and not call get_current_date."""
    mock_tool_executor = MagicMock()
    mock_result = MagicMock()
    mock_result.serialize.return_value = {"results": "mocked search results"}
    mock_tool_executor.execute.return_value = mock_result
    mock_tool_executor.registered_tools = {"web_search", "get_current_date", "calculator"}

    executor = ExistingLLMAgentExecutor(
        finalizer_factory=lambda: MagicMock(finalize=lambda original_task, supporting_info: "final"),
        tool_executor_factory=lambda: mock_tool_executor,
    )

    task = "what are the current treands in vit ap amaravati"
    planner_out = PlannerOutput(
        task_understanding="find current trends in vit ap amaravati",
        required_capabilities=["web_search", "tool_use"],
        requires_tools=True,
        tools_needed=["web_search", "get_current_date"],  # simulating both being returned
        selected_agents=["planner", "tool_executor", "finalizer"],
    )
    arch = ArchitectureManager.create_default_architecture().get_architecture()

    res = executor.execute(task, planner_out, arch)
    tool_trace = [rec for rec in res.execution_trace if rec.agent_id == "tool_executor"]
    assert len(tool_trace) == 1
    assert "web_search" in tool_trace[0].output
    assert "get_current_date" not in tool_trace[0].output
