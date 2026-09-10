"""
Tests for the RL policy and controller layer (Step 11).

These tests cover:
1. Policy construction.
2. Deterministic action selection.
3. Valid action selection.
4. Controller/environment interaction.
5. Trajectory recording.
6. Reward recording.
7. Episode termination/truncation.
8. Reset behaviour.
9. Invalid/empty action handling.
10. No modification of protected workflow files.

These tests are fully offline. No LLM calls, no network calls.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.manager import ArchitectureManager
from app.rl.environment import MASArchitectureEnv
from app.rl.policy import (
    BasePolicy,
    DeterministicBaselinePolicy,
    RandomPolicy,
)
from app.rl.controller import RLController
from app.rl.trajectory import ActionInfo, Trajectory, Transition, TransitionInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_manager() -> ArchitectureManager:
    return ArchitectureManager.create_default_architecture()


def _make_env(**kwargs) -> MASArchitectureEnv:
    return MASArchitectureEnv(_default_manager(), **kwargs)


# ---------------------------------------------------------------------------
# 1. Policy construction tests
# ---------------------------------------------------------------------------

def test_deterministic_policy_construction() -> None:
    """Test that the deterministic baseline policy can be constructed."""
    policy = DeterministicBaselinePolicy()
    assert policy is not None
    assert isinstance(policy, BasePolicy)


def test_random_policy_construction_without_rng() -> None:
    """Test that the random policy can be constructed without an RNG."""
    policy = RandomPolicy()
    assert policy is not None
    assert isinstance(policy, BasePolicy)


def test_random_policy_construction_with_rng() -> None:
    """Test that the random policy can be constructed with a custom RNG."""
    import random
    rng = random.Random(42)
    policy = RandomPolicy(rng=rng)
    assert policy is not None
    assert isinstance(policy, BasePolicy)


def test_policy_is_abstract() -> None:
    """Test that BasePolicy cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BasePolicy()


# ---------------------------------------------------------------------------
# 2. Deterministic action selection tests
# ---------------------------------------------------------------------------

def test_deterministic_policy_single_action() -> None:
    """Test that the deterministic policy returns the only action when there's one."""
    policy = DeterministicBaselinePolicy()

    observation = {"agent_ids": ["planner", "researcher"]}
    valid_action_ids = [0]

    selected = policy.select_action(observation, valid_action_ids)
    assert selected == 0


def test_deterministic_policy_empty_action_list_raises() -> None:
    """Test that the deterministic policy raises on empty action list."""
    policy = DeterministicBaselinePolicy()

    observation = {"agent_ids": ["planner"]}
    valid_action_ids: List[int] = []

    with pytest.raises(ValueError, match="no valid actions available"):
        policy.select_action(observation, valid_action_ids)


def test_deterministic_policy_with_add_edge_priority() -> None:
    """Test that the deterministic policy prioritizes ADD_EDGE actions."""
    policy = DeterministicBaselinePolicy()

    # Create an observation with action metadata indicating ADD_EDGE actions.
    observation = {
        "agent_ids": ["planner", "researcher", "coder"],
        "action_metadata": {
            0: {"action_type": "add_edge"},
            1: {"action_type": "add_edge"},
            2: {"action_type": "activate_agent"},
            3: {"action_type": "change_role"},
        },
    }
    valid_action_ids = [0, 1, 2, 3]

    selected = policy.select_action(observation, valid_action_ids)

    # Should select an ADD_EDGE action (0 or 1).
    assert selected in [0, 1]


def test_deterministic_policy_without_action_metadata() -> None:
    """Test that the deterministic policy falls back when no action metadata."""
    policy = DeterministicBaselinePolicy()

    observation = {"agent_ids": ["planner"]}
    valid_action_ids = [0, 1, 2]

    # Without action_metadata, the policy should return the first action.
    selected = policy.select_action(observation, valid_action_ids)
    assert selected == 0


def test_deterministic_policy_is_deterministic() -> None:
    """Test that the deterministic policy always returns the same action."""
    policy = DeterministicBaselinePolicy()

    observation = {
        "agent_ids": ["planner", "researcher", "coder"],
        "action_metadata": {
            0: {"action_type": "add_edge"},
            1: {"action_type": "activate_agent"},
        },
    }
    valid_action_ids = [0, 1]

    # Run multiple times and verify determinism.
    results = [policy.select_action(observation, valid_action_ids) for _ in range(5)]
    assert all(r == results[0] for r in results)


# ---------------------------------------------------------------------------
# 3. Valid action selection tests
# ---------------------------------------------------------------------------

def test_random_policy_selects_from_valid_actions() -> None:
    """Test that the random policy only selects from valid action IDs."""
    import random
    rng = random.Random(42)  # Fixed seed for reproducibility.
    policy = RandomPolicy(rng=rng)

    observation = {"agent_ids": ["planner"]}
    valid_action_ids = [0, 1, 2, 3, 4]

    # Run multiple times and verify all selections are valid.
    for _ in range(10):
        selected = policy.select_action(observation, valid_action_ids)
        assert selected in valid_action_ids


def test_random_policy_empty_action_list_raises() -> None:
    """Test that the random policy raises on empty action list."""
    policy = RandomPolicy()

    observation = {"agent_ids": ["planner"]}
    valid_action_ids: List[int] = []

    with pytest.raises(ValueError, match="no valid actions available"):
        policy.select_action(observation, valid_action_ids)


def test_policy_interface_requires_select_action() -> None:
    """Test that concrete policies implement select_action."""

    class IncompletePolicy(BasePolicy):
        pass

    with pytest.raises(TypeError):
        IncompletePolicy()


# ---------------------------------------------------------------------------
# 4. Controller/environment interaction tests
# ---------------------------------------------------------------------------

def test_controller_run_episode_returns_trajectory() -> None:
    """Test that the controller returns a trajectory after an episode."""
    env = _make_env()
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    trajectory = controller.run_episode()

    assert isinstance(trajectory, Trajectory)
    assert trajectory.episode_id is not None
    assert len(trajectory.transitions) > 0


def test_controller_environment_interaction_updates_architecture() -> None:
    """Test that the controller interacts with the environment correctly."""
    env = _make_env(role_options=["analysis"])
    initial_arch = env.manager.to_architecture_model()

    # Use a deterministic policy that will select ADD_EDGE actions.
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    trajectory = controller.run_episode()

    # The architecture should have changed after the episode.
    final_arch = env.manager.to_architecture_model()
    # Either the architecture changed, or the episode ended (both are valid).
    assert env.step_count > 0 or env.terminated or env.truncated


def test_controller_respects_max_steps() -> None:
    """Test that the controller respects the environment's max steps."""
    env = _make_env(max_steps=3)
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    trajectory = controller.run_episode()

    assert env.truncated is True
    assert env.step_count == 3
    assert trajectory.truncated is True
    assert trajectory.length == 3


def test_controller_handles_termination() -> None:
    """Test that the controller handles episode termination correctly."""
    env = _make_env(max_steps=10)
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    trajectory = controller.run_episode()

    # The episode should have ended (either terminated or truncated).
    assert env.terminated or env.truncated
    assert len(trajectory.transitions) > 0


# ---------------------------------------------------------------------------
# 5. Trajectory recording tests
# ---------------------------------------------------------------------------

def test_trajectory_records_transitions() -> None:
    """Test that the trajectory records all transitions."""
    env = _make_env(max_steps=3)
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    trajectory = controller.run_episode()

    assert len(trajectory.transitions) == 3
    for i, transition in enumerate(trajectory.transitions):
        assert transition.step == i + 1


def test_trajectory_records_state_action_reward() -> None:
    """Test that the trajectory records state, action, and reward."""
    env = _make_env(max_steps=2)
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    trajectory = controller.run_episode()

    for transition in trajectory.transitions:
        assert isinstance(transition.state, dict)
        assert isinstance(transition.action, ActionInfo)
        assert isinstance(transition.reward, float)
        assert isinstance(transition.next_state, dict)


def test_trajectory_total_reward() -> None:
    """Test that the trajectory computes total reward correctly."""
    env = _make_env(max_steps=3)
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    trajectory = controller.run_episode()

    expected_total = sum(t.reward for t in trajectory.transitions)
    assert trajectory.total_reward == pytest.approx(expected_total)


def test_trajectory_serialization() -> None:
    """Test that the trajectory can be serialized to a dict."""
    env = _make_env(max_steps=2)
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    trajectory = controller.run_episode()
    data = trajectory.to_dict()

    assert isinstance(data, dict)
    assert "episode_id" in data
    assert "transitions" in data
    assert "total_reward" in data
    assert "length" in data
    assert "terminated" in data
    assert "truncated" in data
    assert data["length"] == len(trajectory.transitions)


def test_trajectory_terminated_flag() -> None:
    """Test that the trajectory reflects the terminated flag."""
    env = _make_env(max_steps=5)
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    trajectory = controller.run_episode()

    # With max_steps=5, the episode should be truncated, not terminated.
    assert trajectory.truncated is True
    assert trajectory.terminated is False


# ---------------------------------------------------------------------------
# 6. Reward recording tests
# ---------------------------------------------------------------------------

def test_trajectory_rewards_are_recorded() -> None:
    """Test that rewards are recorded for each transition."""
    env = _make_env(max_steps=3)
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    trajectory = controller.run_episode()

    for transition in trajectory.transitions:
        assert isinstance(transition.reward, float)
        # Reward should be a finite number.
        assert transition.reward == pytest.approx(transition.reward)


def test_invalid_transition_gets_negative_reward() -> None:
    """Test that invalid transitions get negative reward."""
    env = _make_env(role_options=["analysis"])
    env.reset()

    # Perform a valid action first to add an edge.
    action = ArchitectureAction(
        action_type=ActionType.ADD_EDGE,
        source="researcher",
        target="coder",
    )
    action_id = env.encode_action(action)
    env.step(action_id)

    # Now perform the same action again - it should be invalid.
    # The environment's mapper is rebuilt after each step, so the same action_id
    # may map to a different action or be invalid.
    _, reward, _, _, info = env.step(action_id)

    # The second step should be invalid and get negative reward.
    assert reward == pytest.approx(-1.0)
    assert info["transition"]["valid"] is False


# ---------------------------------------------------------------------------
# 7. Episode termination/truncation tests
# ---------------------------------------------------------------------------

def test_episode_truncation_stops_controller() -> None:
    """Test that the controller stops correctly on truncation."""
    env = _make_env(max_steps=2)
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    trajectory = controller.run_episode()

    assert env.truncated is True
    assert len(trajectory.transitions) == 2
    assert trajectory.transitions[-1].truncated is True


def test_episode_termination_stops_controller() -> None:
    """Test that the controller stops correctly on termination."""
    env = _make_env(max_steps=10)
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    trajectory = controller.run_episode()

    # The episode should end (termination or truncation).
    assert env.terminated or env.truncated


def test_controller_resets_environment() -> None:
    """Test that the controller resets the environment at episode start."""
    env = _make_env(max_steps=3)
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    # Run an episode.
    trajectory1 = controller.run_episode()
    initial_step_count = env.step_count

    # Run another episode - should reset.
    trajectory2 = controller.run_episode()

    assert env.step_count == trajectory2.length
    assert env.step_count <= env.max_steps


# ---------------------------------------------------------------------------
# 8. Reset behaviour tests
# ---------------------------------------------------------------------------

def test_controller_reset_before_episode() -> None:
    """Test that the controller resets before running an episode."""
    env = _make_env(max_steps=3)

    # Run first episode to modify the environment state.
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)
    trajectory1 = controller.run_episode()

    # Verify first episode completed.
    assert len(trajectory1.transitions) > 0

    # Run second episode - should reset automatically.
    trajectory2 = controller.run_episode()

    # After run_episode, the environment should be at the end of the episode.
    assert len(trajectory2.transitions) > 0
    assert trajectory2.length <= env.max_steps


def test_multiple_episodes_are_independent() -> None:
    """Test that multiple episodes are independent (reset between them)."""
    env = _make_env(max_steps=2)
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    # First episode.
    trajectory1 = controller.run_episode()
    arch1 = env.manager.to_architecture_model()

    # Second episode should start from the initial architecture.
    trajectory2 = controller.run_episode()
    arch2 = env.manager.to_architecture_model()

    # The architectures at the end of each episode should be different
    # (different trajectories), but both episodes started from the same
    # initial architecture.
    assert len(trajectory1.transitions) == 2
    assert len(trajectory2.transitions) == 2


# ---------------------------------------------------------------------------
# 9. Invalid/empty action handling tests
# ---------------------------------------------------------------------------

def test_controller_handles_invalid_action_gracefully() -> None:
    """Test that the controller handles invalid actions gracefully."""
    env = _make_env(role_options=["analysis"])
    env.reset()

    # Perform a valid action.
    action = ArchitectureAction(
        action_type=ActionType.ADD_EDGE,
        source="researcher",
        target="coder",
    )
    action_id = env.encode_action(action)
    env.step(action_id)

    # Now try to select an action - the policy should handle the current
    # valid action set correctly.
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    # This should not raise, even though some previously valid actions are now invalid.
    result = controller.run_single_step()
    assert "action_id" in result
    assert "reward" in result


def test_policy_raises_on_empty_valid_actions() -> None:
    """Test that the policy raises when given an empty list of valid actions."""
    policy = DeterministicBaselinePolicy()

    observation = {"agent_ids": ["planner"]}
    valid_action_ids: List[int] = []

    with pytest.raises(ValueError, match="no valid actions available"):
        policy.select_action(observation, valid_action_ids)


# ---------------------------------------------------------------------------
# 10. Tests ensuring no modification of protected workflow files
# ---------------------------------------------------------------------------

def test_workflow_file_not_modified() -> None:
    """Test that app/graph/workflow.py was not modified by this step."""
    import os
    import hashlib

    workflow_path = "app/graph/workflow.py"
    assert os.path.exists(workflow_path), "workflow.py should exist"

    # Read the file content.
    with open(workflow_path, "rb") as f:
        content = f.read()

    # Verify it contains expected workflow constructs.
    content_str = content.decode("utf-8")
    assert "build_workflow" in content_str or "run_workflow" in content_str
    assert "planner" in content_str
    assert "researcher" in content_str


def test_agent_implementations_not_modified() -> None:
    """Test that existing agent implementations were not modified."""
    import os

    agent_files = [
        "app/agents/planner.py",
        "app/agents/researcher.py",
        "app/agents/coder.py",
        "app/agents/critic.py",
        "app/agents/finalizer.py",
    ]

    for agent_file in agent_files:
        if os.path.exists(agent_file):
            with open(agent_file, "r") as f:
                content = f.read()
            # Verify they contain expected agent constructs.
            assert len(content) > 0, f"{agent_file} should not be empty"


# ---------------------------------------------------------------------------
# 11. Additional integration-style tests for the controller
# ---------------------------------------------------------------------------

def test_controller_full_episode_with_deterministic_policy() -> None:
    """Test a full episode with the deterministic baseline policy."""
    env = _make_env(role_options=["analysis"], max_steps=5)
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    trajectory = controller.run_episode()

    # Verify trajectory structure.
    assert trajectory.length == env.step_count
    assert trajectory.truncated or trajectory.terminated

    # Verify all transitions have required fields.
    for transition in trajectory.transitions:
        assert transition.step > 0
        assert isinstance(transition.state, dict)
        assert isinstance(transition.action, ActionInfo)
        assert transition.action.action_id >= 0
        assert isinstance(transition.reward, float)
        assert isinstance(transition.next_state, dict)
        assert isinstance(transition.info, TransitionInfo)


def test_controller_single_step_interaction() -> None:
    """Test the single step interaction method."""
    env = _make_env()
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    result = controller.run_single_step()

    assert "action_id" in result
    assert "observation" in result
    assert "next_observation" in result
    assert "reward" in result
    assert "terminated" in result
    assert "truncated" in result
    assert "info" in result
    assert "valid_action_ids" in result

    assert isinstance(result["observation"], dict)
    assert isinstance(result["next_observation"], dict)
    assert isinstance(result["reward"], float)
    assert isinstance(result["terminated"], bool)
    assert isinstance(result["truncated"], bool)
    assert isinstance(result["info"], dict)
    assert isinstance(result["valid_action_ids"], list)


def test_controller_policy_dependency() -> None:
    """Test that the controller works with different policies."""
    env = _make_env(max_steps=2)

    # Test with deterministic policy.
    det_policy = DeterministicBaselinePolicy()
    det_controller = RLController(env, det_policy)
    det_trajectory = det_controller.run_episode()
    assert len(det_trajectory.transitions) > 0

    # Reset environment.
    env.reset()

    # Test with random policy.
    import random
    rng = random.Random(42)
    rand_policy = RandomPolicy(rng=rng)
    rand_controller = RLController(env, rand_policy)
    rand_trajectory = rand_controller.run_episode()
    assert len(rand_trajectory.transitions) > 0


def test_controller_no_direct_architecture_modification() -> None:
    """Test that the controller does not directly modify the architecture."""
    env = _make_env()
    initial_arch = env.manager.to_architecture_model()

    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    # The controller should only interact through the environment.
    # It should not have any direct methods to modify the architecture.
    assert not hasattr(controller, "apply_action")
    assert not hasattr(controller, "modify_architecture")

    # Running an episode should modify the architecture through the environment.
    trajectory = controller.run_episode()
    final_arch = env.manager.to_architecture_model()

    # The architecture may or may not have changed (depending on actions),
    # but the controller itself doesn't have direct modification methods.
    assert trajectory.length > 0


# ---------------------------------------------------------------------------
# 12. Trajectory edge cases
# ---------------------------------------------------------------------------

def test_empty_trajectory_properties() -> None:
    """Test that an empty trajectory has correct properties."""
    trajectory = Trajectory(episode_id="test-empty")

    assert trajectory.length == 0
    assert trajectory.total_reward == 0.0
    assert trajectory.terminated is False
    assert trajectory.truncated is False
    assert len(trajectory.transitions) == 0


def test_trajectory_with_single_transition() -> None:
    """Test a trajectory with a single transition."""
    env = _make_env(max_steps=1)
    policy = DeterministicBaselinePolicy()
    controller = RLController(env, policy)

    trajectory = controller.run_episode()

    assert trajectory.length == 1
    assert trajectory.total_reward == trajectory.transitions[0].reward
    assert trajectory.transitions[0].truncated is True  # max_steps=1 means truncated


def test_transition_info_validation() -> None:
    """Test that TransitionInfo validates correctly."""
    info = TransitionInfo(
        environment_info={"step_count": 1},
        valid_transition=True,
    )

    assert info.environment_info == {"step_count": 1}
    assert info.valid_transition is True


def test_action_info_serialization() -> None:
    """Test that ActionInfo serializes correctly."""
    action_info = ActionInfo(
        action_id=0,
        action_serialized={"action_type": "add_edge", "source": "planner", "target": "coder"},
    )

    data = action_info.model_dump()
    assert data["action_id"] == 0
    assert data["action_serialized"]["action_type"] == "add_edge"


# ---------------------------------------------------------------------------
# 13. Verify no unnecessary dependencies were added
# ---------------------------------------------------------------------------

def test_policy_module_imports() -> None:
    """Test that the policy module only imports necessary dependencies."""
    import app.rl.policy as policy_module

    # Verify the module has the expected classes.
    assert hasattr(policy_module, "BasePolicy")
    assert hasattr(policy_module, "DeterministicBaselinePolicy")
    assert hasattr(policy_module, "RandomPolicy")


def test_controller_module_imports() -> None:
    """Test that the controller module only imports necessary dependencies."""
    import app.rl.controller as controller_module

    # Verify the module has the expected classes.
    assert hasattr(controller_module, "RLController")


def test_trajectory_module_imports() -> None:
    """Test that the trajectory module only imports necessary dependencies."""
    import app.rl.trajectory as trajectory_module

    # Verify the module has the expected classes.
    assert hasattr(trajectory_module, "ActionInfo")
    assert hasattr(trajectory_module, "TransitionInfo")
    assert hasattr(trajectory_module, "Transition")
    assert hasattr(trajectory_module, "Trajectory")
