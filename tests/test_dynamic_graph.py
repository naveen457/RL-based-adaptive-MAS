import json

from app.agents.coder import CoderOutput
from app.agents.critic import CriticOutput
from app.agents.finalizer import FinalizerOutput
from app.agents.planner import PlannerOutput
from app.agents.researcher import ResearcherOutput
from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.llm_adapter import LLMArchitectureAdapter
from app.architecture.manager import ArchitectureManager
from app.graph.adaptive_integration import AdaptiveWorkflowAdapter
from app.runtime.llm_execution import ExistingLLMAgentExecutor
from app.runtime.orchestrator import AdaptiveRuntimeOrchestrator


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeDecisionLLM:
    def __init__(self, response):
        self.response = response

    def invoke(self, messages):
        return FakeResponse(self.response)


class FakePlanner:
    def __init__(self, output, client):
        self.output = output
        self.model = client

    def plan(self, task):
        return self.output


class Researcher:
    def research(self, task):
        return ResearcherOutput(research_question=task, findings=["research result"])


class Coder:
    def code(self, task):
        return CoderOutput(
            approach="test", code="class Solution {}", explanation="code result"
        )


class Critic:
    def review(self, original_task, output_to_review):
        return CriticOutput(
            overall_assessment="verified", verification_status="correct"
        )


class Finalizer:
    def finalize(self, original_task, supporting_info):
        return FinalizerOutput(
            final_answer="final response from dynamic graph",
            key_points=[supporting_info],
            limitations=[],
        )


def _executor():
    return ExistingLLMAgentExecutor(
        researcher_factory=Researcher,
        coder_factory=Coder,
        critic_factory=Critic,
        finalizer_factory=Finalizer,
    )


def _runtime(plan, decision, *, inactive_agent=None):
    manager = ArchitectureManager.create_default_architecture()
    if inactive_agent:
        manager.apply_action(
            ArchitectureAction(
                action_type=ActionType.DEACTIVATE_AGENT,
                agent_id=inactive_agent,
            )
        )
    client = FakeDecisionLLM(decision)
    adapter = LLMArchitectureAdapter(
        client,
        workflow_adapter=AdaptiveWorkflowAdapter(manager=manager),
    )
    return AdaptiveRuntimeOrchestrator(
        planner=FakePlanner(plan, client),
        architecture_adapter=adapter,
        agent_executor=_executor(),
    )


def _coding_plan():
    return PlannerOutput(
        task_understanding="Implement and verify a coding solution.",
        required_capabilities=["coding", "verification"],
        steps=["Implement", "Verify"],
        requires_coding=True,
        requires_verification=True,
        estimated_complexity="medium",
    )


def _research_plan():
    return PlannerOutput(
        task_understanding="Research a topic and synthesize the findings.",
        required_capabilities=["research", "information_synthesis"],
        steps=["Research", "Synthesize"],
        requires_research=True,
        estimated_complexity="medium",
    )


def _greeting_plan():
    return PlannerOutput(
        task_understanding="Say hello.",
        required_capabilities=[],
        steps=[],
        estimated_complexity="trivial",
    )


def test_coding_activation_builds_and_executes_dynamic_graph() -> None:
    runtime = _runtime(
        _coding_plan(),
        {
            "decision": "apply_actions",
            "reasoning": "activate coder",
            "actions": [{"action_type": "activate_agent", "agent_id": "coder"}],
        },
        inactive_agent="coder",
    )

    result = runtime.run_dynamic("Implement and verify a coding solution.")

    assert "coder" in result.graph_nodes
    assert result.compiled is True
    assert "coder" in result.agents_invoked
    assert ("planner", "coder") in {
        (edge["source"], edge["target"]) for edge in result.graph_edges
    }
    assert result.final_response["final_answer"] == "final response from dynamic graph"


def test_greeting_graph_contains_only_required_nodes() -> None:
    runtime = _runtime(
        _greeting_plan(),
        {"decision": "no_change", "reasoning": "baseline", "actions": []},
    )

    result = runtime.run_dynamic("Say hello.")

    assert result.graph_nodes == ["finalizer", "planner"]
    assert result.agents_invoked == ["planner", "finalizer"]
    assert result.compiled is True
    assert "coder" not in result.graph_nodes
    assert "researcher" not in result.graph_nodes


def test_research_activation_enters_adapted_graph() -> None:
    runtime = _runtime(
        _research_plan(),
        {
            "decision": "apply_actions",
            "reasoning": "activate researcher",
            "actions": [{"action_type": "activate_agent", "agent_id": "researcher"}],
        },
        inactive_agent="researcher",
    )

    result = runtime.run_dynamic("Research a topic and synthesize the findings.")

    assert "researcher" in result.graph_nodes
    assert "researcher" in result.agents_invoked
    assert result.compiled is True


def test_invalid_adaptation_builds_graph_from_unchanged_architecture() -> None:
    runtime = _runtime(
        _coding_plan(),
        {
            "decision": "apply_actions",
            "reasoning": "invalid",
            "actions": [
                {"action_type": "add_edge", "source": "planner", "target": "planner"}
            ],
        },
    )
    before = runtime.workflow_adapter.current_architecture().serialize()

    result = runtime.run_dynamic("Implement and verify a coding solution.")

    assert result.initial_architecture == before
    assert result.final_architecture == before
    assert result.compiled is True
    assert "coder" in result.graph_nodes


def test_trace_and_graph_consistency() -> None:
    runtime = _runtime(
        _coding_plan(),
        {
            "decision": "apply_actions",
            "reasoning": "activate coder",
            "actions": [{"action_type": "activate_agent", "agent_id": "coder"}],
        },
        inactive_agent="coder",
    )

    result = runtime.run_dynamic("Implement and verify a coding solution.")
    events = [event.event for event in result.execution_trace]
    graph_edges = {(edge["source"], edge["target"]) for edge in result.graph_edges}

    assert events.index("planner.completed") < events.index("graph.compile.completed")
    assert events.index("graph.compile.completed") < events.index("agent.coder.started")
    assert events[-1] == "workflow.completed"
    assert set(result.agents_invoked).issubset(set(result.graph_nodes))
    assert ("coder", "critic") in graph_edges
    json.dumps(result.serialize())
