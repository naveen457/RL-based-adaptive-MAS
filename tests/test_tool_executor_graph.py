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


def test_multi_tool_execution_with_context_chaining():
    """Verify that multiple tools execute together and pass context from date tool to search tool."""
    mock_tool_executor = MagicMock()
    executed_tools = []
    executed_queries = []

    def mock_execute(tool_name, kwargs):
        executed_tools.append(tool_name)
        res = MagicMock()
        if tool_name == "get_current_date":
            res.status = "success"
            res.result = {"current_date": "2026-09-19", "year": 2026}
            res.serialize.return_value = {"tool_name": "get_current_date", "result": res.result}
        else:
            res.status = "success"
            executed_queries.append(kwargs.get("query"))
            res.result = [{"title": "BRICS 2026 Outcome", "content": "18th summit concluded"}]
            res.serialize.return_value = {"tool_name": "web_search", "result": res.result}
        return res

    mock_tool_executor.execute = mock_execute
    mock_tool_executor.registered_tools = {"web_search", "get_current_date", "calculator"}

    executor = ExistingLLMAgentExecutor(
        finalizer_factory=lambda: MagicMock(finalize=lambda original_task, supporting_info: "final"),
        tool_executor_factory=lambda: mock_tool_executor,
    )

    task = "explain what happened at the 18th BRICS summit today"
    planner_out = PlannerOutput(
        task_understanding="find live outcome of 18th BRICS summit",
        required_capabilities=["web_search", "tool_use"],
        requires_tools=True,
        tools_needed=["web_search", "get_current_date"],
        selected_agents=["planner", "tool_executor", "finalizer"],
    )
    arch = ArchitectureManager.create_default_architecture().get_architecture()

    res = executor.execute(task, planner_out, arch)
    tool_trace = [rec for rec in res.execution_trace if rec.agent_id == "tool_executor"]
    assert len(tool_trace) == 1

    # Both tools must execute
    assert "get_current_date" in executed_tools
    assert "web_search" in executed_tools

    # Priority order: get_current_date must run before web_search
    assert executed_tools.index("get_current_date") < executed_tools.index("web_search")

    # Tool-to-tool communication: web_search must receive the 2026 year context
    assert len(executed_queries) > 0
    assert "2026" in executed_queries[0]


def test_multi_tool_automatic_temporal_expansion():
    """Even if planner only specifies ['web_search'], temporal queries automatically include get_current_date."""
    mock_tool_executor = MagicMock()
    executed_tools = []
    executed_queries = []

    def mock_execute(tool_name, kwargs):
        executed_tools.append(tool_name)
        res = MagicMock()
        if tool_name == "get_current_date":
            res.status = "success"
            res.result = {"current_date": "2026-09-19", "year": 2026}
            res.serialize.return_value = {"tool_name": "get_current_date", "result": res.result}
        else:
            res.status = "success"
            executed_queries.append(kwargs.get("query"))
            res.result = [{"title": "Live Summit News", "content": "Ongoing discussions"}]
            res.serialize.return_value = {"tool_name": "web_search", "result": res.result}
        return res

    mock_tool_executor.execute = mock_execute
    mock_tool_executor.registered_tools = {"web_search", "get_current_date", "calculator"}

    executor = ExistingLLMAgentExecutor(
        finalizer_factory=lambda: MagicMock(finalize=lambda original_task, supporting_info: "final"),
        tool_executor_factory=lambda: mock_tool_executor,
    )

    # Prompt with temporal signal 'latest', but planner only emitted 'web_search'
    task = "find the latest summit developments today"
    planner_out = PlannerOutput(
        task_understanding="find latest summit developments",
        required_capabilities=["web_search"],
        requires_tools=True,
        tools_needed=["web_search"],  # Only 1 tool specified by planner
        selected_agents=["planner", "tool_executor", "finalizer"],
    )
    arch = ArchitectureManager.create_default_architecture().get_architecture()

    res = executor.execute(task, planner_out, arch)

    # Both tools must execute due to automatic temporal grounding
    assert "get_current_date" in executed_tools
    assert "web_search" in executed_tools
    assert executed_tools.index("get_current_date") < executed_tools.index("web_search")
    assert any("2026" in q for q in executed_queries)


def test_multi_tool_loop_with_calculation():
    """Verify 3-tool sequential execution with calculator when math intent is present."""
    mock_tool_executor = MagicMock()
    executed_tools = []

    def mock_execute(tool_name, kwargs):
        executed_tools.append(tool_name)
        res = MagicMock()
        res.status = "success"
        if tool_name == "get_current_date":
            res.result = {"current_date": "2026-09-19", "year": 2026}
        elif tool_name == "calculator":
            res.result = 42
        else:
            res.result = "search results"
        res.serialize.return_value = {"tool_name": tool_name, "result": res.result}
        return res

    mock_tool_executor.execute = mock_execute
    mock_tool_executor.registered_tools = {"web_search", "get_current_date", "calculator"}

    executor = ExistingLLMAgentExecutor(
        finalizer_factory=lambda: MagicMock(finalize=lambda original_task, supporting_info: "final"),
        tool_executor_factory=lambda: mock_tool_executor,
    )

    task = "find latest oil price today and calculate the 10% tax"
    planner_out = PlannerOutput(
        task_understanding="find price and calculate tax",
        required_capabilities=["web_search", "tool_use"],
        requires_tools=True,
        tools_needed=["web_search"],
        selected_agents=["planner", "tool_executor", "finalizer"],
    )
    arch = ArchitectureManager.create_default_architecture().get_architecture()

    res = executor.execute(task, planner_out, arch)

    # All 3 synergistic tools must execute in topological order
    assert executed_tools == ["get_current_date", "web_search", "calculator"]


def test_arxiv_search_tool_registration():
    from app.agents.tool_executor import ToolExecutor
    executor = ToolExecutor()
    assert "arxiv_search" in executor.registered_tools


def test_research_papers_query_routes_to_arxiv_search():
    """Queries asking for research papers or literature must invoke arxiv_search."""
    mock_tool_executor = MagicMock()
    mock_result = MagicMock()
    mock_result.serialize.return_value = {"tool_name": "arxiv_search", "result": {"papers": [{"title": "Test Paper"}]}}
    mock_result.status = "success"
    mock_result.result = {"papers": [{"title": "Test Paper"}]}
    mock_tool_executor.execute.return_value = mock_result
    mock_tool_executor.registered_tools = {"web_search", "get_current_date", "calculator", "arxiv_search"}

    executor = ExistingLLMAgentExecutor(
        finalizer_factory=lambda: MagicMock(finalize=lambda original_task, supporting_info: "final"),
        tool_executor_factory=lambda: mock_tool_executor,
    )

    task = "find recent arxiv research papers on reinforcement learning in multi-agent systems"
    planner_out = PlannerOutput(
        task_understanding="find research papers on MARL",
        required_capabilities=["research"],
        requires_tools=True,
        selected_agents=["planner", "tool_executor", "finalizer"],
    )
    arch = ArchitectureManager.create_default_architecture().get_architecture()

    res = executor.execute(task, planner_out, arch)
    mock_tool_executor.execute.assert_called_with("arxiv_search", {"query": task, "max_results": 5})
    assert res is not None



