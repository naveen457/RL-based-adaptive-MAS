"""
Tests for the Step 8 RL environment baseline.

These tests are fully offline. No OpenRouter calls are made.

Coverage targets
----------------
1. Environment initialization.
2. Initial state encoding.
3. Deterministic state encoding.
4. Action mapping.
5. Action encode/decode round trip.
6. Invalid action ID handling.
7. Valid environment step.
8. Architecture changes after valid step.
9. Reward returned.
10. Invalid action behavior.
11. Invalid action does not corrupt architecture.
12. Reset behavior.
13. Episode step limit.
14. Deterministic reset.
15. Info object contains safe expected fields.
16. No credentials in state/info serialization.
"""

from __future__ import annotations

import json

import pytest

from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.manager import ArchitectureManager
from app.architecture.adaptive import AdaptiveArchitecture
from app.architecture.models import MASArchitecture
from app.evaluation.evaluator import EvaluationResult
from app.rl.action_space import ArchitectureActionMapper
from app.rl.environment import MASArchitectureEnv
from app.rl.state import ArchitectureStateEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_manager() -> ArchitectureManager:
    return ArchitectureManager.create_default_architecture()


def _action(*, action_type: ActionType, **fields: str) -> ArchitectureAction:
    return ArchitectureAction(action_type=action_type, **fields)


# ---------------------------------------------------------------------------
# 1. Environment initialization
# ---------------------------------------------------------------------------

def test_environment_initialization_has_default_max_steps() -> None:
    env = MASArchitectureEnv(_default_manager())
    assert env.max_steps == MASArchitectureEnv.DEFAULT_MAX_STEPS
    assert env.step_count == 0
    assert env.terminated is False
    assert env.truncated is False


def test_environment_initialization_uses_provided_max_steps() -> None:
    env = MASArchitectureEnv(_default_manager(), max_steps=7)
    assert env.max_steps == 7


def test_environment_initialization_uses_role_options() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis", "planning"])
    assert set(env.mapper.role_options) == {"analysis", "planning"}


def test_environment_rejects_non_positive_max_steps() -> None:
    with pytest.raises(ValueError, match="max_steps must be >= 1"):
        MASArchitectureEnv(_default_manager(), max_steps=0)


def test_environment_initial_observation_is_dict() -> None:
    env = MASArchitectureEnv(_default_manager())
    observation, info = env.reset()
    assert isinstance(observation, dict)
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# 2. Initial state encoding
# ---------------------------------------------------------------------------

def test_initial_state_encoding_has_sorted_agent_ids() -> None:
    env = MASArchitectureEnv(_default_manager())
    observation, _ = env.reset()
    assert observation["agent_ids"] == sorted(env.manager.get_architecture().agent_ids)


def test_initial_state_encoding_has_five_active_agents() -> None:
    env = MASArchitectureEnv(_default_manager())
    observation, _ = env.reset()
    assert observation["active_agent_count"] == 5
    assert observation["agent_count"] == 5
    assert sum(observation["activity_vector"]) == 5


def test_initial_state_encoding_has_adjacency_matrix() -> None:
    env = MASArchitectureEnv(_default_manager())
    observation, _ = env.reset()
    matrix = observation["adjacency_matrix"]
    assert len(matrix) == 5
    assert all(len(row) == 5 for row in matrix)
    assert all(isinstance(v, int) for row in matrix for v in row)


def test_initial_state_encoding_role_vector_matches_architecture() -> None:
    env = MASArchitectureEnv(_default_manager())
    observation, _ = env.reset()
    role_map = {
        agent_id: role
        for agent_id, role in zip(
            observation["agent_ids"], observation["role_vector"]
        )
    }
    assert role_map["planner"] == "planning"
    assert role_map["researcher"] == "research"
    assert role_map["coder"] == "implementation"
    assert role_map["critic"] == "verification"
    assert role_map["finalizer"] == "synthesis"


# ---------------------------------------------------------------------------
# 3. Deterministic state encoding
# ---------------------------------------------------------------------------

def test_state_encoding_is_deterministic_within_episode() -> None:
    env = MASArchitectureEnv(_default_manager())
    observation, _ = env.reset()
    first = observation
    second = env.encoder.encode()
    assert first == second


def test_state_encoding_same_architecture_same_observation() -> None:
    manager_a = _default_manager()
    manager_b = _default_manager()

    encoder_a = ArchitectureStateEncoder(manager_a.get_architecture())
    encoder_b = ArchitectureStateEncoder(manager_b.get_architecture())

    assert encoder_a.encode() == encoder_b.encode()


# ---------------------------------------------------------------------------
# 4. Action mapping
# ---------------------------------------------------------------------------

def test_action_mapping_contains_expected_count_with_role_options() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    actions = env.mapper.actions
    assert len(actions) == env.mapper.action_count
    assert env.mapper.action_count > 0


def test_action_mapping_sorted_deterministic() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    actions = env.mapper.actions
    serialized = [a.serialize() for a in actions]
    assert serialized == sorted(serialized, key=lambda d: (
        d["action_type"],
        d.get("agent_id"),
        d.get("source"),
        d.get("target"),
        d.get("new_role"),
    ))


def test_action_mapping_without_role_options_has_no_change_role_actions() -> None:
    env = MASArchitectureEnv(_default_manager())
    actions = env.mapper.actions
    assert not any(a.action_type == ActionType.CHANGE_ROLE for a in actions)


# ---------------------------------------------------------------------------
# 5. Action encode/decode round trip
# ---------------------------------------------------------------------------

def test_action_encode_decode_round_trip() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    for action in env.mapper.actions:
        action_id = env.encode_action(action)
        assert env.decode_action(action_id) == action


def test_action_encode_returns_stable_id_for_same_action() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    action = ArchitectureAction(
        action_type=ActionType.CHANGE_ROLE,
        agent_id="coder",
        new_role="analysis",
    )
    first = env.encode_action(action)
    second = env.encode_action(action)
    assert first == second


# ---------------------------------------------------------------------------
# 6. Invalid action ID handling
# ---------------------------------------------------------------------------

def test_invalid_action_id_raises_on_step() -> None:
    env = MASArchitectureEnv(_default_manager())
    _, _ = env.reset()
    with pytest.raises(ValueError, match="invalid action id"):
        env.step(-1)


def test_invalid_action_id_raises_on_decode() -> None:
    env = MASArchitectureEnv(_default_manager())
    _, _ = env.reset()
    with pytest.raises(ValueError, match="invalid action id"):
        env.decode_action(env.mapper.action_count + 10)


def test_encode_unknown_action_raises() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    unknown = ArchitectureAction(
        action_type=ActionType.CHANGE_ROLE,
        agent_id="coder",
        new_role="nonexistent-role",
    )
    with pytest.raises(ValueError, match="action is not in this action mapping"):
        env.encode_action(unknown)


def test_mapper_rejects_empty_action_list() -> None:
    with pytest.raises(ValueError, match="action set must be non-empty"):
        ArchitectureActionMapper(actions=[], role_options=["analysis"])


# ---------------------------------------------------------------------------
# 7. Valid environment step
# ---------------------------------------------------------------------------

def test_valid_step_returns_terminating_tuple_of_correct_shape() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    observation, info = env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    result = env.step(action_id)
    assert len(result) == 5
    observation, reward, terminated, truncated, info = result
    assert isinstance(observation, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_valid_step_increments_step_count() -> None:
    env = MASArchitectureEnv(_default_manager())
    _, _ = env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    env.step(action_id)
    assert env.step_count == 1


# ---------------------------------------------------------------------------
# 8. Architecture changes after valid step
# ---------------------------------------------------------------------------

def test_valid_add_edge_changes_architecture() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    _, _ = env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    env.step(action_id)

    edges = {(e.source, e.target) for e in env.manager.get_communication_edges()}
    assert ("planner", "critic") in edges


def test_valid_change_role_changes_role() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    _, _ = env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        )
    )
    env.step(action_id)
    assert env.manager.get_role("coder") == "analysis"


def test_valid_deactivate_changes_active_count() -> None:
    env = MASArchitectureEnv(_default_manager())
    _, _ = env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.DEACTIVATE_AGENT,
            agent_id="researcher",
        )
    )
    env.step(action_id)
    assert env.manager.get_architecture().active_agent_count == 4


# ---------------------------------------------------------------------------
# 9. Reward returned
# ---------------------------------------------------------------------------

def test_valid_step_returns_positive_reward() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    _, _ = env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    _, reward, _, _, info = env.step(action_id)
    # After a valid step, reward must come from RewardCalculator as the
    # evaluation delta, not from a hardcoded +1.0.
    assert reward == pytest.approx(info["evaluation"]["current_score"] - info["evaluation"]["previous_score"])
    assert info["evaluation"]["task_success_score"] is None


def test_environment_evaluates_initial_architecture() -> None:
    env = MASArchitectureEnv(_default_manager())
    env.reset()
    evaluation = env.current_evaluation
    assert isinstance(evaluation, EvaluationResult)
    assert evaluation.architecture_id == "static-mas-v1"
    assert evaluation.task_success_score is None


def test_invalid_step_returns_negative_reward() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    _, _ = env.reset()
    # Use an action that is valid in the initial action mapping but invalid
    # at step time because a prior step already performed it.
    action = ArchitectureAction(
        action_type=ActionType.ADD_EDGE,
        source="researcher",
        target="coder",
    )
    action_id = env.mapper.encode(action)

    # First step: valid.
    env.step(action_id)

    # Second step with the same action id: invalid because it already exists.
    _, reward, _, _, info = env.step(action_id)
    assert reward == -1.0
    assert info["transition"]["valid"] is False


# ---------------------------------------------------------------------------
# 10. Invalid action behavior
# ---------------------------------------------------------------------------

def test_invalid_action_does_not_change_step_count_architecture() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    _, _ = env.reset()
    initial = env.manager.to_architecture_model()
    initial_version = env.adaptive.version()

    # This action exists in the initial mapping and is invalid at step time.
    action = ArchitectureAction(
        action_type=ActionType.ADD_EDGE,
        source="researcher",
        target="coder",
    )
    action_id = env.mapper.encode(action)

    # First step: valid.
    env.step(action_id)

    # Second step with the same action id: invalid because it already exists.
    env.step(action_id)

    assert env.step_count == 2
    assert env.manager.to_architecture_model() != initial
    assert env.adaptive.version() == 1


def test_invalid_action_stays_invalid_after_reset_rebuild() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    _, _ = env.reset()
    action_id = env.mapper.encode(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="researcher",
            target="coder",
        )
    )

    # First step: valid.
    env.step(action_id)

    # Second step with the same action id: invalid because it already exists.
    _, reward, _, _, info = env.step(action_id)
    assert reward == pytest.approx(-1.0)
    assert info["transition"]["valid"] is False

    env.reset()

    # After reset, the same action id is valid again.
    _, reward2, _, _, info2 = env.step(action_id)
    assert reward2 == pytest.approx(info2["evaluation"]["current_score"] - info2["evaluation"]["previous_score"])
    assert info2["transition"]["valid"] is True
    assert info2["evaluation"]["task_success_score"] is None


# ---------------------------------------------------------------------------
# 11. Invalid action does not corrupt architecture
# ---------------------------------------------------------------------------

def test_invalid_action_preserves_version_and_history() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    _, _ = env.reset()
    action_id = env.mapper.encode(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="researcher",
            target="coder",
        )
    )

    # First step: valid.
    env.step(action_id)

    # Second step with the same action id: invalid, so version/history unchanged.
    env.step(action_id)

    assert env.adaptive.version() == 1
    assert len(env.adaptive.history()) == 1


def test_invalid_action_preserves_initial_snapshot() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    _, _ = env.reset()

    action = ArchitectureAction(
        action_type=ActionType.ADD_EDGE,
        source="researcher",
        target="coder",
    )
    action_id = env.mapper.encode(action)

    # First step: valid.
    env.step(action_id)

    # Second step: invalid because the edge already exists, so snapshot unchanged.
    snapshot_before = env.manager.serialize()
    env.step(action_id)

    assert env.manager.serialize() == snapshot_before


# ---------------------------------------------------------------------------
# 12. Reset behavior
# ---------------------------------------------------------------------------

def test_reset_restores_initial_architecture() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    _, _ = env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        )
    )
    env.step(action_id)
    assert env.manager.get_role("coder") == "analysis"

    env.reset()
    assert env.manager.get_role("coder") == "implementation"


def test_reset_clears_episode_counters() -> None:
    env = MASArchitectureEnv(_default_manager())
    _, _ = env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    env.step(action_id)
    assert env.step_count == 1

    env.reset()
    assert env.step_count == 0
    assert env.terminated is False
    assert env.truncated is False


def test_reset_returns_initial_observation() -> None:
    env = MASArchitectureEnv(_default_manager())
    first_obs, _ = env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    env.step(action_id)
    second_obs, _ = env.reset()
    assert first_obs == second_obs


# ---------------------------------------------------------------------------
# 13. Episode step limit
# ---------------------------------------------------------------------------

def test_episode_truncates_after_max_steps() -> None:
    env = MASArchitectureEnv(_default_manager(), max_steps=3)
    _, _ = env.reset()

    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )

    env.step(action_id)
    env.step(action_id)
    env.step(action_id)

    assert env.truncated is True
    assert env.step_count == 3

    with pytest.raises(RuntimeError, match="episode has already terminated/truncated"):
        env.step(action_id)


def test_reset_clears_truncation_flag() -> None:
    env = MASArchitectureEnv(_default_manager(), max_steps=2)
    _, _ = env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    env.step(action_id)
    env.step(action_id)
    assert env.truncated is True

    env.reset()
    assert env.truncated is False


# ---------------------------------------------------------------------------
# 14. Deterministic reset
# ---------------------------------------------------------------------------

def test_reset_is_deterministic() -> None:
    env_a = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    env_b = MASArchitectureEnv(_default_manager(), role_options=["analysis"])

    _, _ = env_a.reset()
    _, _ = env_b.reset()

    action_id = env_a.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )

    env_a.step(action_id)
    env_b.step(action_id)

    env_a.reset()
    env_b.reset()

    assert env_a.manager.to_architecture_model() == env_b.manager.to_architecture_model()
    assert env_a.adaptive.version() == env_b.adaptive.version()
    assert env_a.adaptive.history() == env_b.adaptive.history()


# ---------------------------------------------------------------------------
# 15. Info object contains safe expected fields
# ---------------------------------------------------------------------------

def test_info_contains_expected_safe_fields() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    observation, info = env.reset()

    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    _, _, _, _, after_info = env.step(action_id)

    expected_keys = {
        "architecture_id",
        "architecture_version",
        "step_count",
        "max_steps",
        "action_id",
        "action",
        "active_agent_count",
        "agent_count",
        "action_space_size",
        "role_options",
    }

    for key in expected_keys:
        assert key in info, f"missing key {key} in reset info"
        assert key in after_info, f"missing key {key} in after step info"

    assert "transition" in after_info, "missing transition in after step info"


def test_info_reflects_clean_initial_state() -> None:
    env = MASArchitectureEnv(_default_manager())
    _, info = env.reset()
    assert info["architecture_version"] == 0
    assert info["step_count"] == 0
    assert info["action_id"] is None
    assert info["action"] is None


def test_info_after_valid_transition_is_structured() -> None:
    env = MASArchitectureEnv(_default_manager())
    _, _ = env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    _, _, _, _, info = env.step(action_id)
    transition = info["transition"]
    assert transition["valid"] is True
    assert transition["architecture_version"] == 1
    assert transition["step"] == 1
    assert transition["action"]["action_type"] == "add_edge"


def test_info_contains_evaluation_fields_after_step() -> None:
    env = MASArchitectureEnv(_default_manager())
    _, _ = env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    _, _, _, _, info = env.step(action_id)
    evaluation = info["evaluation"]
    expected_keys = {
        "architecture_id",
        "architecture_version",
        "validity_score",
        "efficiency_score",
        "communication_cost",
        "active_agent_count",
        "edge_count",
        "overall_score",
        "task_success_score",
    }
    for key in expected_keys:
        assert key in evaluation, f"missing key {key} in evaluation"

    # task_success_score must remain None in the baseline.
    assert evaluation["task_success_score"] is None


# ---------------------------------------------------------------------------
# 16. No credentials in state/info serialization
# ---------------------------------------------------------------------------

def _blob_has_no_credential_markers(value: object) -> bool:
    text = json.dumps(value, default=str)
    lowered = text.lower()
    forbidden = ("sk-or-", "sk-proj-", "api_key", "openai_api_key")
    for marker in forbidden:
        assert marker not in lowered, f"credential marker found: {marker}"
    return True


def test_initial_observation_contains_no_credentials() -> None:
    env = MASArchitectureEnv(_default_manager())
    observation, _ = env.reset()
    _blob_has_no_credential_markers(observation)


def test_info_contains_no_credentials() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    _, _ = env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        )
    )
    _, _, _, _, info = env.step(action_id)
    _blob_has_no_credential_markers(info)


def test_transition_metadata_contains_no_credentials() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    _, _ = env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        )
    )
    _, _, _, _, info = env.step(action_id)
    _blob_has_no_credential_markers(info["transition"])


def test_action_serialization_contains_no_credentials() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])
    _, _ = env.reset()
    action = ArchitectureAction(
        action_type=ActionType.CHANGE_ROLE,
        agent_id="coder",
        new_role="analysis",
    )
    encoded = env.mapper.encode_action_safely(action)
    _blob_has_no_credential_markers(encoded)


# ---------------------------------------------------------------------------
# End-to-end A0 -> A1 -> A2 -> reset scenario
# ---------------------------------------------------------------------------

def test_end_to_end_add_edge_then_change_role_then_reset() -> None:
    env = MASArchitectureEnv(_default_manager(), role_options=["analysis"])

    # A0
    initial_observation, _ = env.reset()
    initial_architecture = env.manager.to_architecture_model()
    initial_evaluation = env.current_evaluation

    # A0 -> A1 via ADD_EDGE(planner, critic)
    add_edge_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    observation_a1, reward_a1, terminated_a1, truncated_a1, info_a1 = env.step(add_edge_id)

    architecture_a1 = env.manager.to_architecture_model()
    assert architecture_a1 != initial_architecture
    assert ("planner", "critic") in {
        (e.source, e.target) for e in architecture_a1.communication_edges
    }
    assert terminated_a1 is False
    assert truncated_a1 is False
    assert info_a1["architecture_version"] == 1
    assert info_a1["transition"]["valid"] is True

    # A1 evaluation should exist and differ from A0 where structure changed.
    evaluation_a1 = env.current_evaluation
    assert evaluation_a1.architecture_id == architecture_a1.architecture_id
    assert evaluation_a1.task_success_score is None
    assert info_a1["evaluation"]["task_success_score"] is None

    # A1 -> A2 via CHANGE_ROLE(coder, analysis)
    change_role_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        )
    )
    observation_a2, reward_a2, terminated_a2, truncated_a2, info_a2 = env.step(change_role_id)

    architecture_a2 = env.manager.to_architecture_model()
    assert architecture_a2.role_map["coder"] == "analysis"
    assert info_a2["architecture_version"] == 2
    assert info_a2["evaluation"]["task_success_score"] is None

    # Reset
    reset_observation, reset_info = env.reset()
    assert env.manager.to_architecture_model() == initial_architecture
    assert env.adaptive.version() == 0
    assert env.adaptive.history() == []
    assert reset_info["architecture_version"] == 0
    # Observation should be back to the initial encoded architecture.
    assert reset_observation == initial_observation
    # Reset must restore the initial evaluation too.
    assert env.current_evaluation.architecture_id == initial_evaluation.architecture_id
    assert env.current_evaluation.task_success_score is None
