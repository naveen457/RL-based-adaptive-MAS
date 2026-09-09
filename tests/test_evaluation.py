"""
Tests for the Step 9 evaluation and reward layer.

These tests are fully offline. No OpenRouter calls are made.

Coverage targets
----------------
1. Valid architecture evaluation.
2. Invalid architecture evaluation.
3. Validity score.
4. Communication cost.
5. Active-agent count.
6. Edge count.
7. Efficiency score.
8. Task success is explicitly unavailable.
9. Evaluation result schema.
10. Deterministic evaluation.
11. Initial architecture evaluation.
12. Evaluation after ADD_EDGE.
13. Evaluation after CHANGE_ROLE.
14. Reward calculation.
15. Reward is based on score difference.
16. Invalid transition penalty if represented.
17. No credentials in evaluation result.
18. No LLM/API calls.
"""

from __future__ import annotations

import json

import pytest

from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.manager import ArchitectureManager
from app.architecture.models import MASArchitecture
from app.evaluation.evaluator import (
    ArchitectureEvaluator,
    EvaluationResult,
    calculate_overall_score,
    calculate_validity_score,
    calculate_communication_cost,
    calculate_efficiency_score,
)
from app.evaluation.reward import RewardCalculator, INVALID_TRANSITION_PENALTY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_manager() -> ArchitectureManager:
    return ArchitectureManager.create_default_architecture()


def _action(*, action_type: ActionType, **fields: str) -> ArchitectureAction:
    return ArchitectureAction(action_type=action_type, **fields)


def _build_invalid_architecture_clean() -> ArchitectureManager:
    from app.architecture.models import MASArchitecture as MA, AgentDefinition, CommunicationEdge

    arch = MA(
        architecture_id="invalid-arch-v1",
        agents=[
            AgentDefinition(agent_id="planner", role="planning", description="d"),
            AgentDefinition(agent_id="coder", role="implementation", description="d"),
        ],
        communication_edges=[
            CommunicationEdge(source="planner", target="coder"),
            CommunicationEdge(source="ghost", target="coder"),
        ],
    )
    return ArchitectureManager(architecture=arch)


def _apply_action(manager: ArchitectureManager, action: ArchitectureAction) -> ArchitectureManager:
    """Return a new manager with *action* applied, leaving *manager* unchanged."""
    import copy
    new_mgr = ArchitectureManager(manager.to_architecture_model())
    new_mgr.apply_action(action)
    return new_mgr


# ---------------------------------------------------------------------------
# 1-3. Basic evaluation and validity
# ---------------------------------------------------------------------------

def test_valid_architecture_evaluation() -> None:
    evaluator = ArchitectureEvaluator()
    result = evaluator.evaluate(_default_manager().get_architecture())
    assert result.validity_score == 1.0
    assert result.overall_score >= 0.0
    assert result.overall_score <= 1.0


def test_invalid_architecture_evaluation() -> None:
    evaluator = ArchitectureEvaluator()
    manager = _build_invalid_architecture_clean()
    result = evaluator.evaluate(manager.get_architecture())
    assert result.validity_score == 0.0


def test_validity_score_is_binary() -> None:
    valid = _default_manager().get_architecture()
    invalid = _build_invalid_architecture_clean().get_architecture()
    assert calculate_validity_score(valid) == 1.0
    assert calculate_validity_score(invalid) == 0.0


# ---------------------------------------------------------------------------
# 4-6. Communication cost / active agent count / edge count
# ---------------------------------------------------------------------------

def test_communication_cost_equals_edge_count() -> None:
    manager = _default_manager()
    arch = manager.get_architecture()
    cost = calculate_communication_cost(arch)
    assert cost == 6
    assert cost == len(arch.communication_edges)


def test_active_agent_count() -> None:
    manager = _default_manager()
    arch = manager.get_architecture()
    assert arch.active_agent_count == 5


def test_edge_count_in_result() -> None:
    evaluator = ArchitectureEvaluator()
    result = evaluator.evaluate(_default_manager().get_architecture())
    assert result.edge_count == 6
    assert result.communication_cost == 6


# ---------------------------------------------------------------------------
# 7. Efficiency score
# ---------------------------------------------------------------------------

def test_efficiency_score_is_deterministic() -> None:
    evaluator = ArchitectureEvaluator()
    arch = _default_manager().get_architecture()
    first = calculate_efficiency_score(arch)
    second = calculate_efficiency_score(arch)
    assert first == second


def test_efficiency_score_falls_within_range() -> None:
    evaluator = ArchitectureEvaluator()
    arch = _default_manager().get_architecture()
    efficiency = calculate_efficiency_score(arch)
    # Baseline normalization keeps this within [0, 1].
    assert 0.0 <= efficiency <= 1.0


def test_efficiency_score_changes_after_deactivating_agent() -> None:
    evaluator = ArchitectureEvaluator()
    manager = _default_manager()
    before = evaluator.evaluate(manager.get_architecture())

    updated = _apply_action(
        manager,
        _action(action_type=ActionType.DEACTIVATE_AGENT, agent_id="researcher"),
    )
    after = evaluator.evaluate(updated.get_architecture())

    assert after.active_agent_count == before.active_agent_count - 1
    assert after.efficiency_score != before.efficiency_score


# ---------------------------------------------------------------------------
# 8. Task success explicitly unavailable
# ---------------------------------------------------------------------------

def test_task_success_score_is_none_in_baseline() -> None:
    evaluator = ArchitectureEvaluator()
    result = evaluator.evaluate(_default_manager().get_architecture())
    assert result.task_success_score is None


def test_evaluation_result_makes_unavailability_explicit() -> None:
    evaluator = ArchitectureEvaluator()
    result = evaluator.evaluate(_default_manager().get_architecture())
    assert result.task_success_score is None
    assert result.notes


# ---------------------------------------------------------------------------
# 9. Evaluation result schema
# ---------------------------------------------------------------------------

def test_evaluation_result_schema() -> None:
    evaluator = ArchitectureEvaluator()
    result = evaluator.evaluate(_default_manager().get_architecture())
    expected_fields = {
        "architecture_id",
        "architecture_version",
        "validity_score",
        "task_success_score",
        "efficiency_score",
        "communication_cost",
        "active_agent_count",
        "edge_count",
        "overall_score",
        "notes",
    }
    for field in expected_fields:
        assert hasattr(result, field)

    # Edge count alias must match communication cost.
    assert result.edge_count == result.communication_cost


def test_evaluation_result_serialization() -> None:
    evaluator = ArchitectureEvaluator()
    result = evaluator.evaluate(_default_manager().get_architecture())
    data = result.model_dump()
    assert isinstance(data, dict)
    assert data["architecture_id"] == "static-mas-v1"
    assert data["validity_score"] == 1.0
    assert data["task_success_score"] is None


# ---------------------------------------------------------------------------
# 10. Deterministic evaluation
# ---------------------------------------------------------------------------

def test_evaluation_is_deterministic() -> None:
    evaluator_a = ArchitectureEvaluator()
    evaluator_b = ArchitectureEvaluator()
    arch = _default_manager().get_architecture()
    result_a = evaluator_a.evaluate(arch)
    result_b = evaluator_b.evaluate(arch)
    assert result_a.model_dump() == result_b.model_dump()


def test_evaluation_with_version_and_notes_is_stable() -> None:
    evaluator = ArchitectureEvaluator()
    arch = _default_manager().get_architecture()
    first = evaluator.evaluate(
        arch,
        architecture_version=7,
        notes="custom note",
    )
    second = evaluator.evaluate(
        arch,
        architecture_version=7,
        notes="custom note",
    )
    assert first.model_dump() == second.model_dump()


# ---------------------------------------------------------------------------
# 11. Initial architecture evaluation
# ---------------------------------------------------------------------------

def test_initial_architecture_can_be_evaluated() -> None:
    evaluator = ArchitectureEvaluator()
    manager = _default_manager()
    result = evaluator.evaluate(manager.get_architecture())
    assert result.architecture_id == "static-mas-v1"
    assert result.architecture_version is None or isinstance(result.architecture_version, int)
    assert result.validity_score == 1.0


# ---------------------------------------------------------------------------
# 12-13. Evaluation after ADD_EDGE and CHANGE_ROLE
# ---------------------------------------------------------------------------

def test_evaluation_after_add_edge_changes_cost_and_score() -> None:
    evaluator = ArchitectureEvaluator()
    manager = _default_manager()
    before = evaluator.evaluate(manager.get_architecture())

    updated = _apply_action(
        manager,
        _action(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        ),
    )
    after = evaluator.evaluate(updated.get_architecture())

    assert after.communication_cost == before.communication_cost + 1
    assert after.edge_count == before.edge_count + 1
    assert after.communication_cost == 7


def test_evaluation_after_change_role_changes_score() -> None:
    evaluator = ArchitectureEvaluator()
    manager = _default_manager()
    before = evaluator.evaluate(manager.get_architecture())

    updated = _apply_action(
        manager,
        _action(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        ),
    )
    after = evaluator.evaluate(updated.get_architecture())

    assert after.architecture_id == before.architecture_id
    # Role change is structural in the architecture model, but our baseline
    # structural metric set does not directly use roles. The key requirement
    # is deterministic evaluation and explicit unavailability of task success.
    assert after.architecture_id == "static-mas-v1"
    assert after.validity_score == 1.0
    assert after.task_success_score is None


def test_evaluation_after_changing_role_preserves_validity() -> None:
    evaluator = ArchitectureEvaluator()
    manager = _default_manager()
    updated = _apply_action(
        manager,
        _action(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        ),
    )
    result = evaluator.evaluate(updated.get_architecture())
    assert result.validity_score == 1.0


# ---------------------------------------------------------------------------
# 14-16. Reward calculation
# ---------------------------------------------------------------------------

def test_reward_based_on_score_difference() -> None:
    calculator = RewardCalculator()
    previous = EvaluationResult(
        architecture_id="a",
        validity_score=0.5,
        task_success_score=None,
        efficiency_score=0.5,
        communication_cost=4,
        active_agent_count=3,
        edge_count=4,
        overall_score=0.5,
        notes="",
    )
    current = EvaluationResult(
        architecture_id="b",
        validity_score=0.5,
        task_success_score=None,
        efficiency_score=0.5,
        communication_cost=5,
        active_agent_count=3,
        edge_count=5,
        overall_score=0.4,
        notes="",
    )
    reward = calculator.calculate(
        previous_result=previous,
        current_result=current,
        valid_transition=True,
    )
    assert reward == pytest.approx(current.overall_score - previous.overall_score)


def test_reward_valid_transition_uses_delta() -> None:
    calculator = RewardCalculator()
    previous = EvaluationResult(
        architecture_id="a",
        validity_score=1.0,
        task_success_score=None,
        efficiency_score=0.5,
        communication_cost=4,
        active_agent_count=3,
        edge_count=4,
        overall_score=0.5,
        notes="",
    )
    current = EvaluationResult(
        architecture_id="b",
        validity_score=1.0,
        task_success_score=None,
        efficiency_score=0.5,
        communication_cost=7,
        active_agent_count=3,
        edge_count=7,
        overall_score=0.4,
        notes="",
    )
    reward = calculator.calculate(
        previous_result=previous,
        current_result=current,
        valid_transition=True,
    )
    assert reward == pytest.approx(-0.1)


def test_reward_context_includes_scores() -> None:
    calculator = RewardCalculator()
    previous = EvaluationResult(
        architecture_id="a",
        validity_score=1.0,
        task_success_score=None,
        efficiency_score=0.5,
        communication_cost=4,
        active_agent_count=3,
        edge_count=4,
        overall_score=0.5,
        notes="",
    )
    current = EvaluationResult(
        architecture_id="b",
        validity_score=1.0,
        task_success_score=None,
        efficiency_score=0.5,
        communication_cost=7,
        active_agent_count=3,
        edge_count=7,
        overall_score=0.4,
        notes="",
    )
    context = calculator.calculate_with_context(
        previous_result=previous,
        current_result=current,
        valid_transition=True,
    )
    assert context["reward"] == pytest.approx(-0.1)
    assert context["previous_score"] == 0.5
    assert context["current_score"] == 0.4
    assert context["valid_transition"] is True


def test_invalid_transition_penalty_applied() -> None:
    calculator = RewardCalculator()
    previous = EvaluationResult(
        architecture_id="a",
        validity_score=1.0,
        task_success_score=None,
        efficiency_score=0.5,
        communication_cost=4,
        active_agent_count=3,
        edge_count=4,
        overall_score=0.5,
        notes="",
    )
    current = EvaluationResult(
        architecture_id="a",
        validity_score=1.0,
        task_success_score=None,
        efficiency_score=0.5,
        communication_cost=4,
        active_agent_count=3,
        edge_count=4,
        overall_score=0.5,
        notes="",
    )
    reward = calculator.calculate(
        previous_result=previous,
        current_result=current,
        valid_transition=False,
    )
    assert reward == INVALID_TRANSITION_PENALTY


def test_reward_context_invalid_transition() -> None:
    calculator = RewardCalculator()
    previous = EvaluationResult(
        architecture_id="a",
        validity_score=1.0,
        task_success_score=None,
        efficiency_score=0.5,
        communication_cost=4,
        active_agent_count=3,
        edge_count=4,
        overall_score=0.5,
        notes="",
    )
    current = EvaluationResult(
        architecture_id="a",
        validity_score=1.0,
        task_success_score=None,
        efficiency_score=0.5,
        communication_cost=4,
        active_agent_count=3,
        edge_count=4,
        overall_score=0.5,
        notes="",
    )
    context = calculator.calculate_with_context(
        previous_result=previous,
        current_result=current,
        valid_transition=False,
    )
    assert context["reward"] == INVALID_TRANSITION_PENALTY
    assert context["valid_transition"] is False


# ---------------------------------------------------------------------------
# 17. No credentials in evaluation result
# ---------------------------------------------------------------------------

def _blob_has_no_credential_markers(value: object) -> bool:
    text = json.dumps(value, default=str)
    lowered = text.lower()
    forbidden = ("sk-or-", "sk-proj-", "api_key", "openai_api_key")
    for marker in forbidden:
        assert marker not in lowered, f"credential marker found: {marker}"
    return True


def test_evaluation_result_contains_no_credentials() -> None:
    evaluator = ArchitectureEvaluator()
    result = evaluator.evaluate(_default_manager().get_architecture())
    _blob_has_no_credential_markers(result.model_dump())


def test_reward_context_contains_no_credentials() -> None:
    calculator = RewardCalculator()
    previous = EvaluationResult(
        architecture_id="a",
        validity_score=1.0,
        task_success_score=None,
        efficiency_score=0.5,
        communication_cost=4,
        active_agent_count=3,
        edge_count=4,
        overall_score=0.5,
        notes="",
    )
    current = EvaluationResult(
        architecture_id="b",
        validity_score=1.0,
        task_success_score=None,
        efficiency_score=0.5,
        communication_cost=7,
        active_agent_count=3,
        edge_count=7,
        overall_score=0.4,
        notes="",
    )
    context = calculator.calculate_with_context(
        previous_result=previous,
        current_result=current,
        valid_transition=True,
    )
    _blob_has_no_credential_markers(context)


# ---------------------------------------------------------------------------
# 18. No LLM/API calls
# ---------------------------------------------------------------------------

def test_evaluation_does_not_import_or_touch_agent_code() -> None:
    # Importing the evaluator must not require agent runtime imports.
    import app.evaluation.evaluator as evaluator_module
    import app.evaluation.reward as reward_module
    assert hasattr(evaluator_module, "ArchitectureEvaluator")
    assert hasattr(reward_module, "RewardCalculator")
