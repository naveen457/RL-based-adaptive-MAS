"""Unit tests for Multi-Turn Architecture State & Continuous Evolution (Step 1).

Tests that:
1. Multi-agent topology persists and evolves across consecutive turns in continuous mode.
2. Architecture versions advance monotonically (v0 -> v1 -> v2...).
3. Reset to baseline restores the clean v0 architecture.
4. Active agents from previous turns are preserved until adapted.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.agents.coder import CoderOutput
from app.agents.finalizer import FinalizerOutput
from app.agents.planner import PlannerOutput
from app.agents.researcher import ResearcherOutput
from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.llm_adapter import LLMArchitectureAdapter
from app.architecture.manager import ArchitectureManager
from app.graph.adaptive_integration import AdaptiveWorkflowAdapter
from app.runtime.llm_execution import ExistingLLMAgentExecutor
from app.runtime.orchestrator import AdaptiveRuntimeOrchestrator


class MockDecisionClient:
    """Mock client returning sequential decisions."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.idx = 0

    def invoke(self, messages: Any) -> Any:
        class ContentWrapper:
            def __init__(self, content: str) -> None:
                self.content = content

        if self.idx < len(self.responses):
            resp = self.responses[self.idx]
            self.idx += 1
        else:
            resp = {"decision": "no_change", "reasoning": "no change", "actions": []}
        return ContentWrapper(json.dumps(resp))


class MockPlanner:
    def __init__(self, plans: list[PlannerOutput], client: Any) -> None:
        self.plans = list(plans)
        self.idx = 0
        self.model = client

    def plan(self, task: str) -> PlannerOutput:
        if self.idx < len(self.plans):
            p = self.plans[self.idx]
            self.idx += 1
            return p
        return self.plans[-1]


def _make_mock_executor() -> ExistingLLMAgentExecutor:
    class Res:
        def research(self, task: str = "", **kwargs):
            return ResearcherOutput(research_question=task, findings=["findings"])

    class Cod:
        def code(self, task: str = "", **kwargs):
            return CoderOutput(approach="app", code="code", explanation="exp")

    class Fin:
        def finalize(self, original_task: str = "", supporting_info: str = "", **kwargs):
            return FinalizerOutput(final_answer="done", key_points=[], limitations=[])

    return ExistingLLMAgentExecutor(
        researcher_factory=Res,
        coder_factory=Cod,
        finalizer_factory=Fin,
    )


def test_continuous_evolution_preserves_and_increments_version() -> None:
    """Verify that architecture evolves across consecutive turns without resetting."""
    plan1 = PlannerOutput(
        task_understanding="Task 1: Research only",
        required_capabilities=["research"],
        requires_research=True,
        requires_coding=False,
    )
    decision1 = {
        "decision": "apply_actions",
        "reasoning": "deactivate coder",
        "actions": [{"action_type": "deactivate_agent", "agent_id": "coder"}],
    }

    plan2 = PlannerOutput(
        task_understanding="Task 2: Coding needed",
        required_capabilities=["coding"],
        requires_research=False,
        requires_coding=True,
    )
    decision2 = {
        "decision": "apply_actions",
        "reasoning": "reactivate coder and deactivate researcher",
        "actions": [
            {"action_type": "activate_agent", "agent_id": "coder"},
            {"action_type": "deactivate_agent", "agent_id": "researcher"},
        ],
    }

    client = MockDecisionClient([decision1, decision2])
    planner = MockPlanner([plan1, plan2], client)
    manager = ArchitectureManager.create_default_architecture()
    wf_adapter = AdaptiveWorkflowAdapter(manager=manager)
    adapter = LLMArchitectureAdapter(client, workflow_adapter=wf_adapter)

    orch = AdaptiveRuntimeOrchestrator(
        planner=planner,
        architecture_adapter=adapter,
        agent_executor=_make_mock_executor(),
    )

    assert orch.current_version == 0
    assert "coder" in orch.current_architecture.active_agent_ids

    # Turn 1: Research task (coder deactivated)
    res1 = orch.run_dynamic("Research AI")
    assert res1.architecture_version == 1
    assert "coder" not in orch.current_architecture.active_agent_ids
    assert "researcher" in orch.current_architecture.active_agent_ids

    # Turn 2: Continuous execution (preserves state from Turn 1, then adapts)
    res2 = orch.run_dynamic("Code Python")
    assert res2.architecture_version >= 2
    assert "coder" in orch.current_architecture.active_agent_ids
    assert "researcher" not in orch.current_architecture.active_agent_ids


def test_reset_to_baseline_restores_v0() -> None:
    """Verify reset_to_baseline restores full 5-agent baseline topology."""
    plan = PlannerOutput(
        task_understanding="Research task",
        required_capabilities=["research"],
        requires_research=True,
    )
    decision = {
        "decision": "apply_actions",
        "reasoning": "prune coder",
        "actions": [{"action_type": "deactivate_agent", "agent_id": "coder"}],
    }

    client = MockDecisionClient([decision])
    planner = MockPlanner([plan], client)
    manager = ArchitectureManager.create_default_architecture()
    wf_adapter = AdaptiveWorkflowAdapter(manager=manager)
    adapter = LLMArchitectureAdapter(client, workflow_adapter=wf_adapter)

    orch = AdaptiveRuntimeOrchestrator(
        planner=planner,
        architecture_adapter=adapter,
        agent_executor=_make_mock_executor(),
    )

    orch.run_dynamic("Task 1")
    assert "coder" not in orch.current_architecture.active_agent_ids

    # Reset
    baseline_arch = orch.reset_to_baseline()
    assert orch.current_version == 0
    assert "coder" in baseline_arch.active_agent_ids
    assert "researcher" in baseline_arch.active_agent_ids
    assert len(baseline_arch.active_agent_ids) == 5
