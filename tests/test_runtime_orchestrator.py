"""Unit and behavioral tests for AdaptiveRuntimeOrchestrator.

Tests baseline architecture preservation, dynamic action application,
execution trace observation from state, and serialization.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from app.agents.coder import CoderOutput
from app.agents.critic import CriticOutput
from app.agents.finalizer import FinalizerOutput
from app.agents.planner import PlannerOutput
from app.agents.researcher import ResearcherOutput
from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.llm_adapter import LLMArchitectureAdapter
from app.architecture.manager import ArchitectureManager
from app.graph.adaptive_integration import AdaptiveWorkflowAdapter
from app.runtime.orchestrator import AdaptiveRuntimeOrchestrator


class FakeResponse:
    def __init__(self, content: Any) -> None:
        self.content = content


class FakeLLM:
    def __init__(self, response: Any) -> None:
        self.response = response

    def invoke(self, messages: Any) -> Any:
        return FakeResponse(self.response)


class FakePlanner:
    def __init__(self, output: PlannerOutput, client: Any = None) -> None:
        self.output = output
        self.model = client

    def plan(self, task: str) -> PlannerOutput:
        return self.output


def _planner(**overrides: Any) -> PlannerOutput:
    values: Dict[str, Any] = {
        "task_understanding": "Have a simple conversation.",
        "required_capabilities": [],
        "steps": [],
        "requires_research": False,
        "requires_coding": False,
        "requires_verification": False,
        "estimated_complexity": "trivial",
    }
    values.update(overrides)
    return PlannerOutput(**values)


def _coding_plan() -> PlannerOutput:
    return PlannerOutput(
        task_understanding="Implement and verify a coding solution.",
        required_capabilities=["coding", "verification"],
        steps=["Implement", "Verify"],
        requires_coding=True,
        requires_verification=True,
        estimated_complexity="medium",
    )


def _research_plan() -> PlannerOutput:
    return PlannerOutput(
        task_understanding="Research a topic and synthesize the findings.",
        required_capabilities=["research", "information_synthesis"],
        steps=["Research", "Synthesize"],
        requires_research=True,
        estimated_complexity="medium",
    )


def _final_answer(text: str) -> FinalizerOutput:
    return FinalizerOutput(final_answer=text, key_points=[text], limitations=[])


def _critic_output() -> CriticOutput:
    return CriticOutput(
        overall_assessment="The output is ready.",
        verification_status="correct",
    )


def _runtime(
    planner_output: PlannerOutput,
    decision: Any,
    workflow_state: Any = None,
    *,
    inactive_agent: str | None = None,
) -> AdaptiveRuntimeOrchestrator:
    manager = ArchitectureManager.create_default_architecture()
    if inactive_agent:
        manager.apply_action(
            ArchitectureAction(
                action_type=ActionType.DEACTIVATE_AGENT,
                agent_id=inactive_agent,
            )
        )
    workflow_adapter = AdaptiveWorkflowAdapter(manager=manager)
    llm = FakeLLM(decision)
    architecture_adapter = LLMArchitectureAdapter(
        llm,
        workflow_adapter=workflow_adapter,
    )
    runner = (
        (lambda task: workflow_state)
        if workflow_state is not None
        else (lambda task: {"final_answer": {"answer": task}})
    )
    return AdaptiveRuntimeOrchestrator(
        planner=FakePlanner(planner_output, llm),
        architecture_adapter=architecture_adapter,
        workflow_runner=runner,
    )


# ---------------------------------------------------------------------------
# Baseline Preservation and Action Application Tests
# ---------------------------------------------------------------------------


def test_simple_task_preserves_baseline_architecture_and_executes() -> None:
    runtime = _runtime(
        _planner(),
        {
            "decision": "no_change",
            "reasoning": "Baseline is sufficient.",
            "actions": [],
        },
    )

    result = runtime.run("Have a simple conversation.")

    assert result.accepted_actions == []
    assert result.architecture_changed is False
    assert result.initial_active_agents == result.final_active_agents
    assert result.final_response == {"answer": "Have a simple conversation."}
    assert result.langgraph_compatible is True


def test_coding_task_activates_coder_before_execution() -> None:
    runtime = _runtime(
        _planner(
            task_understanding="Implement and verify a coding solution.",
            required_capabilities=["coding", "verification"],
            requires_coding=True,
            requires_verification=True,
            estimated_complexity="medium",
        ),
        {
            "decision": "apply_actions",
            "reasoning": "Coding requires the coder.",
            "actions": [{"action_type": "activate_agent", "agent_id": "coder"}],
        },
        inactive_agent="coder",
    )

    result = runtime.run("Implement and verify a coding solution.")

    assert "coder" not in result.initial_active_agents
    assert "coder" in result.final_active_agents
    assert result.accepted_actions == [
        {"action_type": "activate_agent", "agent_id": "coder"}
    ]
    assert result.architecture_changed is True
    assert result.architecture_version == 1
    assert result.langgraph_compatible is True


def test_research_task_activates_researcher_before_execution() -> None:
    runtime = _runtime(
        _planner(
            task_understanding="Research a topic and synthesize the findings.",
            required_capabilities=["research", "information_synthesis"],
            requires_research=True,
            estimated_complexity="medium",
        ),
        {
            "decision": "apply_actions",
            "reasoning": "Research requires the researcher.",
            "actions": [{"action_type": "activate_agent", "agent_id": "researcher"}],
        },
        inactive_agent="researcher",
    )

    result = runtime.run("Research a topic and synthesize the findings.")

    assert "researcher" not in result.initial_active_agents
    assert "researcher" in result.final_active_agents
    assert result.architecture_changed is True
    assert result.langgraph_compatible is True


def test_invalid_proposal_preserves_architecture_and_continues_execution() -> None:
    runtime = _runtime(
        _planner(
            task_understanding="Implement and verify a coding solution.",
            required_capabilities=["coding", "verification"],
            requires_coding=True,
            requires_verification=True,
        ),
        {
            "decision": "apply_actions",
            "reasoning": "Invalid self-loop.",
            "actions": [
                {"action_type": "add_edge", "source": "planner", "target": "planner"}
            ],
        },
        inactive_agent="coder",
    )

    result = runtime.run("Implement and verify a coding solution.")

    assert result.accepted_actions == []
    assert result.rejected_actions
    assert result.architecture_changed is False
    assert result.initial_active_agents == result.final_active_agents
    assert result.final_response is not None
    assert result.architecture_version == 0


def test_runtime_result_is_serializable_and_preserves_planner_output() -> None:
    runtime = _runtime(
        _planner(required_capabilities=["research"], requires_research=True),
        {"decision": "no_change", "reasoning": "No change.", "actions": []},
    )

    result = runtime.run("Research a topic.")
    serialized = result.serialize()

    json.dumps(serialized)
    assert serialized["planner_output"]["requires_research"] is True
    assert serialized["architecture_decision"]["decision"] == "no_change"


# ---------------------------------------------------------------------------
# Execution Trace Observation Tests
# ---------------------------------------------------------------------------


def test_coding_activation_is_reflected_in_actual_execution_trace() -> None:
    task = "Implement and verify a coding solution."
    state = {
        "planner_output": _coding_plan(),
        "coder_output": CoderOutput(
            approach="Use a direct implementation.",
            code="def solve(): pass",
            explanation="The implementation solves the task.",
        ),
        "critic_output": _critic_output(),
        "final_answer": _final_answer("Coder output was synthesized."),
    }
    runtime = _runtime(
        _coding_plan(),
        {
            "decision": "apply_actions",
            "reasoning": "Activate coder.",
            "actions": [{"action_type": "activate_agent", "agent_id": "coder"}],
        },
        state,
        inactive_agent="coder",
    )

    result = runtime.run(task)

    assert "coder" in result.agents_actually_invoked
    coder_record = next(
        record for record in result.execution_trace if record.agent_id == "coder"
    )
    assert coder_record.status == "completed"
    assert coder_record.output["code"] == "def solve(): pass"
    assert result.final_response["final_answer"] == "Coder output was synthesized."
    assert result.execution_consistency_errors == []


def test_research_activation_is_reflected_in_actual_execution_trace() -> None:
    task = "Research a topic and synthesize the findings."
    state = {
        "planner_output": _research_plan(),
        "research_output": ResearcherOutput(
            research_question=task,
            findings=["Finding from the researcher."],
            sources_or_evidence=["Provided context"],
        ),
        "final_answer": _final_answer("Research output was synthesized."),
    }
    runtime = _runtime(
        _research_plan(),
        {
            "decision": "apply_actions",
            "reasoning": "Activate researcher.",
            "actions": [{"action_type": "activate_agent", "agent_id": "researcher"}],
        },
        state,
        inactive_agent="researcher",
    )

    result = runtime.run(task)

    assert "researcher" in result.agents_actually_invoked
    researcher_record = next(
        record for record in result.execution_trace if record.agent_id == "researcher"
    )
    assert researcher_record.status == "completed"
    assert researcher_record.output["findings"] == ["Finding from the researcher."]
    assert result.execution_consistency_errors == []


def test_simple_task_does_not_invoke_specialized_agents() -> None:
    plan = PlannerOutput(
        task_understanding="Have a simple conversation.",
        required_capabilities=[],
        steps=[],
        estimated_complexity="trivial",
    )
    state = {
        "planner_output": plan,
        "final_answer": _final_answer("Conversation response."),
    }
    runtime = _runtime(
        plan,
        {
            "decision": "no_change",
            "reasoning": "No specialization needed.",
            "actions": [],
        },
        state,
    )

    result = runtime.run("Have a simple conversation.")

    assert result.agents_actually_invoked == ["planner", "finalizer"]
    assert result.execution_trace[1].status == "not_invoked"
    assert result.execution_trace[2].status == "not_invoked"
    assert result.final_response["final_answer"] == "Conversation response."


def test_inactive_final_agent_is_not_allowed_to_execute() -> None:
    plan = _coding_plan()
    state = {
        "planner_output": plan,
        "critic_output": _critic_output(),
        "final_answer": _final_answer("Execution continued safely."),
    }
    runtime = _runtime(
        plan,
        {
            "decision": "no_change",
            "reasoning": "No architecture action.",
            "actions": [],
        },
        state,
        inactive_agent="coder",
    )

    result = runtime.run("Implement and verify a coding solution.")

    assert "coder" not in result.agents_actually_invoked
    assert result.execution_consistency_errors == []


def test_output_from_inactive_agent_is_reported_as_inconsistent() -> None:
    plan = _coding_plan()
    state = {
        "planner_output": plan,
        "coder_output": CoderOutput(
            approach="invalid execution",
            code="pass",
            explanation="Should not have run.",
        ),
        "final_answer": _final_answer("Observed response."),
    }
    runtime = _runtime(
        plan,
        {
            "decision": "no_change",
            "reasoning": "No architecture action.",
            "actions": [],
        },
        state,
        inactive_agent="coder",
    )

    result = runtime.run("Implement and verify a coding solution.")

    assert "coder" in result.agents_actually_invoked
    assert (
        "inactive agent 'coder' produced workflow output"
        in result.execution_consistency_errors
    )


def test_final_response_is_taken_from_workflow_output_not_architecture_metadata() -> None:
    plan = _research_plan()
    state = {
        "planner_output": plan,
        "research_output": ResearcherOutput(
            research_question="Research",
            findings=["Actual finding"],
            sources_or_evidence=[],
        ),
        "final_answer": _final_answer("Final answer from workflow."),
    }
    runtime = _runtime(
        plan,
        {
            "decision": "apply_actions",
            "reasoning": "Activate researcher.",
            "actions": [{"action_type": "activate_agent", "agent_id": "researcher"}],
        },
        state,
        inactive_agent="researcher",
    )

    result = runtime.run("Research")

    assert result.final_response["final_answer"] == "Final answer from workflow."
    assert "Actual finding" in result.execution_trace[1].output["findings"]
    assert result.final_response != result.final_architecture
