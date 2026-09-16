import json
import pytest

from app.agents.planner import PlannerOutput
from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.llm_adapter import (
    ArchitectureRecommendation,
    LLMArchitectureAdapter,
)
from app.architecture.manager import ArchitectureManager
from app.graph.adaptive_integration import AdaptiveWorkflowAdapter

TASK = "Research a technical topic and synthesize the findings."


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return FakeResponse(self.response)


def _adapter(response):
    return LLMArchitectureAdapter(
        FakeLLM(response),
        workflow_adapter=AdaptiveWorkflowAdapter(
            manager=ArchitectureManager.create_default_architecture()
        ),
    )


def test_prompt_contains_task_and_serialized_architecture() -> None:
    adapter = _adapter('{"reasoning":"none","actions":[]}')

    messages = adapter.build_prompt(TASK)

    assert TASK in messages[1]["content"]
    assert "static-mas-v1" in messages[1]["content"]
    assert "researcher" in messages[1]["content"]


def test_structured_response_parsing_from_json_and_fence() -> None:
    adapter = _adapter('```json\n{"reasoning":"use research","actions":[]}\n```')

    recommendation = adapter.recommend(TASK)

    assert recommendation.reasoning == "use research"
    assert recommendation.actions == []


def test_valid_architecture_action_conversion() -> None:
    adapter = _adapter(
        json.dumps(
            {
                "reasoning": "Add a synthesis path",
                "actions": [
                    {
                        "action_type": "add_edge",
                        "source": "researcher",
                        "target": "finalizer",
                    }
                ],
            }
        )
    )

    recommendation = adapter.recommend(TASK)
    action = adapter.parse_response(recommendation).actions[0]
    result = adapter.apply_recommendation(TASK, recommendation)

    assert ActionType(action["action_type"]) is ActionType.ADD_EDGE
    assert result.valid_actions == [action]
    assert {
        "source": "researcher",
        "target": "finalizer",
    } in result.workflow_plan.expected_edges


def test_invalid_action_is_rejected_without_stopping_other_actions() -> None:
    adapter = _adapter(
        {
            "reasoning": "try two changes",
            "actions": [
                {"action_type": "add_edge", "source": "planner", "target": "planner"},
                {
                    "action_type": "add_edge",
                    "source": "researcher",
                    "target": "finalizer",
                },
                {"action_type": "unknown_action", "agent_id": "researcher"},
            ],
        }
    )

    result = adapter.adapt(TASK)

    assert len(result.valid_actions) == 1
    assert len(result.rejected_actions) == 2
    assert (
        result.final_architecture["communication_edges"][-1]["source"] == "researcher"
    )


def test_multiple_valid_actions_are_applied_sequentially() -> None:
    adapter = _adapter(
        {
            "reasoning": "add two supported edges",
            "actions": [
                {
                    "action_type": "add_edge",
                    "source": "researcher",
                    "target": "finalizer",
                },
                {"action_type": "add_edge", "source": "coder", "target": "finalizer"},
            ],
        }
    )

    result = adapter.adapt(TASK)

    assert len(result.valid_actions) == 2
    assert result.workflow_plan.compatibility.compatible is True
    assert result.workflow_plan.execution_mode == "adapted"


def test_result_serialization_is_json_friendly() -> None:
    result = _adapter('{"reasoning":"none","actions":[]}').adapt(TASK)

    serialized = result.serialize()

    json.dumps(serialized)
    assert serialized["langgraph_compatible"] is True
    assert serialized["final_architecture"]["architecture_id"] == "static-mas-v1"


def test_architecture_validation_and_application_use_existing_integration() -> None:
    adapter = _adapter(
        {
            "reasoning": "change role",
            "actions": [
                {
                    "action_type": "change_role",
                    "agent_id": "researcher",
                    "new_role": "analysis",
                }
            ],
        }
    )

    result = adapter.adapt(TASK)

    researcher = next(
        agent
        for agent in result.final_architecture["agents"]
        if agent["agent_id"] == "researcher"
    )
    assert researcher["role"] == "analysis"
    assert result.valid_actions[0]["action_type"] == "change_role"


def test_langgraph_compatibility_conversion() -> None:
    result = _adapter('{"reasoning":"none","actions":[]}').adapt(TASK)

    assert result.workflow_plan.compatibility.compatible is True
    assert result.langgraph_compatible is True
    assert result.workflow_plan.active_agents == [
        "coder",
        "critic",
        "finalizer",
        "planner",
        "researcher",
    ]


def test_deterministic_mocked_end_to_end_flow() -> None:
    response = {
        "reasoning": "connect research to synthesis",
        "actions": [
            {"action_type": "add_edge", "source": "researcher", "target": "finalizer"}
        ],
    }

    first = _adapter(response).adapt(TASK).serialize()
    second = _adapter(response).adapt(TASK).serialize()

    assert first == second


def test_basic_flow_delegates_to_existing_workflow(monkeypatch) -> None:
    adapter = _adapter('{"reasoning":"none","actions":[]}')
    marker = {"final_answer": "done"}
    monkeypatch.setattr(
        adapter.workflow_adapter,
        "run_baseline_workflow",
        lambda task: marker,
    )

    result, state = adapter.run_basic_flow(TASK)

    assert result.langgraph_compatible is True
    assert state is marker


# ---------------------------------------------------------------------------
# Planner Output Adaptation Tests
# ---------------------------------------------------------------------------


def _planner(**overrides):
    values = {
        "task_understanding": "Complete the requested task.",
        "required_capabilities": [],
        "steps": [],
        "requires_research": False,
        "requires_coding": False,
        "requires_verification": False,
        "estimated_complexity": "low",
    }
    values.update(overrides)
    return PlannerOutput(**values)


def _manager_with_deactivated(agent_id):
    manager = ArchitectureManager.create_default_architecture()
    manager.apply_action(
        ArchitectureAction(
            action_type=ActionType.DEACTIVATE_AGENT,
            agent_id=agent_id,
        )
    )
    return manager


def test_research_planner_causes_research_capable_proposal() -> None:
    adapter = _adapter(
        {
            "decision": "apply_actions",
            "reasoning": "Research is required, so activate the researcher.",
            "actions": [{"action_type": "activate_agent", "agent_id": "researcher"}],
        }
    )
    adapter.workflow_adapter = AdaptiveWorkflowAdapter(
        manager=_manager_with_deactivated("researcher")
    )
    planner = _planner(
        task_understanding="Research the topic and synthesize findings.",
        required_capabilities=["research", "synthesis"],
        requires_research=True,
        estimated_complexity="high",
    )

    result = adapter.adapt_from_planner_output(planner)

    assert result.valid_actions[0]["action_type"] == ActionType.ACTIVATE_AGENT.value
    assert "researcher" in result.workflow_plan.active_agents
    assert json.dumps(adapter.client.messages[0])
    assert "required_capabilities" in adapter.client.messages[0][1]["content"]


def test_coding_planner_causes_implementation_capable_proposal() -> None:
    adapter = _adapter(
        {
            "decision": "apply_actions",
            "reasoning": "Coding is required, so activate the coder.",
            "actions": [{"action_type": "activate_agent", "agent_id": "coder"}],
        }
    )
    adapter.workflow_adapter = AdaptiveWorkflowAdapter(
        manager=_manager_with_deactivated("coder")
    )
    planner = _planner(
        task_understanding="Implement and verify a coding solution.",
        required_capabilities=["coding", "verification"],
        requires_coding=True,
        requires_verification=True,
    )

    result = adapter.adapt_from_planner_output(planner)

    assert result.valid_actions == [
        {"action_type": "activate_agent", "agent_id": "coder"}
    ]
    coder = next(
        agent
        for agent in result.final_architecture["agents"]
        if agent["agent_id"] == "coder"
    )
    assert coder["active"] is True


def test_simple_conversation_can_produce_no_change() -> None:
    adapter = _adapter(
        {
            "decision": "no_change",
            "reasoning": "The baseline is sufficient.",
            "actions": [],
        }
    )
    planner = _planner(
        task_understanding="Have a simple conversation.",
        estimated_complexity="trivial",
    )
    before = adapter.workflow_adapter.current_architecture().serialize()

    result = adapter.adapt_from_planner_output(planner)

    assert result.architecture_decision["decision"] == "no_change"
    assert result.valid_actions == []
    assert result.final_architecture == before


def test_no_change_action_alias_is_supported() -> None:
    adapter = _adapter(
        {
            "reasoning": "No adaptation needed.",
            "actions": [{"action_type": "no_change"}],
        }
    )

    result = adapter.adapt_from_planner_output(_planner())

    assert result.architecture_decision["decision"] == "no_change"
    assert result.valid_actions == []


def test_malformed_planner_decision_is_rejected_safely() -> None:
    adapter = _adapter("not valid JSON")

    with pytest.raises(ValueError):
        adapter.adapt_from_planner_output(_planner())


def test_invalid_proposal_does_not_mutate_architecture() -> None:
    adapter = _adapter(
        {
            "decision": "apply_actions",
            "reasoning": "Invalid self-loop.",
            "actions": [
                {"action_type": "add_edge", "source": "planner", "target": "planner"}
            ],
        }
    )
    before = adapter.workflow_adapter.current_architecture().serialize()

    result = adapter.adapt_from_planner_output(_planner())

    assert result.valid_actions == []
    assert result.rejected_actions
    assert result.final_architecture == before
    assert adapter.workflow_adapter.current_version() == 0


def test_mixed_invalid_proposal_is_atomic() -> None:
    adapter = _adapter(
        {
            "decision": "apply_actions",
            "reasoning": "One valid and one invalid action.",
            "actions": [
                {
                    "action_type": "add_edge",
                    "source": "researcher",
                    "target": "finalizer",
                },
                {"action_type": "add_edge", "source": "planner", "target": "planner"},
            ],
        }
    )
    before = adapter.workflow_adapter.current_architecture().serialize()

    result = adapter.adapt_from_planner_output(_planner())

    assert result.valid_actions == []
    assert result.final_architecture == before


def test_result_contains_typed_planner_and_architecture_state() -> None:
    adapter = _adapter(
        {
            "decision": "apply_actions",
            "reasoning": "Connect research to synthesis.",
            "actions": [
                {
                    "action_type": "add_edge",
                    "source": "researcher",
                    "target": "finalizer",
                }
            ],
        }
    )

    result = adapter.adapt_from_planner_output(
        _planner(required_capabilities=["research"])
    )
    serialized = result.serialize()

    assert serialized["planner_output"]["required_capabilities"] == ["research"]
    assert serialized["architecture_decision"]["decision"] == "apply_actions"
    assert serialized["workflow_plan"]["compatibility"]["compatible"] is True

