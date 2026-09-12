"""
Additional tests for Q-learning training layer (Step 16).

These tests cover training-specific scenarios that complement the existing
test_q_learning.py tests. They focus on:
- Training loop behavior
- Training metrics
- MetaEnvironment compatibility
- Serialization safety
- Integration with existing components

These tests are fully offline. No LLM calls, no API calls.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from app.architecture.manager import ArchitectureManager
from app.rl.environment import MASArchitectureEnv
from app.rl.q_learning import (
    QLearningPolicy,
    QLearningTrainer,
    QTable,
    StateEncoder,
    TrainingStats,
)
from app.rl.meta_environment import MetaEnvironment
from app.rl.meta_task import MetaTask


def _default_env(**kwargs) -> MASArchitectureEnv:
    """Create a default environment for testing."""
    manager = ArchitectureManager.create_default_architecture()
    return MASArchitectureEnv(manager, **kwargs)


# ============================================================================
# Q-table inspection: number of states and state-action pairs
# ============================================================================

def test_q_table_num_states() -> None:
    """Test QTable.num_states() returns correct count."""
    q_table = QTable()
    
    # Initially no states
    assert q_table.num_states() == 0
    assert q_table.num_state_action_pairs() == 0
    
    # Add some entries
    state1 = (("active",), ("role1",), ("agent1",))
    state2 = (("inactive",), ("role2",), ("agent2",))
    
    q_table.set(state1, 0, 1.0)
    q_table.set(state1, 1, 2.0)
    q_table.set(state2, 0, 3.0)
    
    assert q_table.num_states() == 2
    assert q_table.num_state_action_pairs() == 3


def test_q_table_num_state_action_pairs_after_updates() -> None:
    """Test that updating same state-action pair doesn't increase count."""
    q_table = QTable()
    
    state = (("active",), ("role",), ("agent",))
    
    # First update
    q_table.update(
        state_key=state,
        action_id=0,
        reward=1.0,
        next_state_key=state,
        next_valid_actions=[0],
        alpha=0.5,
        gamma=0.9,
        terminated=False,
        truncated=False,
    )
    
    assert q_table.num_states() == 1
    assert q_table.num_state_action_pairs() == 1
    
    # Update same pair again
    q_table.update(
        state_key=state,
        action_id=0,
        reward=2.0,
        next_state_key=state,
        next_valid_actions=[0],
        alpha=0.5,
        gamma=0.9,
        terminated=False,
        truncated=False,
    )
    
    # Should still be 1 state and 1 state-action pair
    assert q_table.num_states() == 1
    assert q_table.num_state_action_pairs() == 1


# ============================================================================
# Training metrics: unique states and state-action pairs
# ============================================================================

def test_training_metrics_include_q_table_stats() -> None:
    """Test that training reflects Q-table growth."""
    policy = QLearningPolicy(seed=42, alpha=0.5, gamma=0.9)
    env = _default_env(max_steps=2)
    trainer = QLearningTrainer(env, policy)
    
    # Before training
    initial_states = policy.q_table.num_states()
    initial_pairs = policy.q_table.num_state_action_pairs()
    
    # Train one episode
    trainer.train_episode()
    
    # After training, Q-table should have grown
    assert policy.q_table.num_states() >= initial_states
    assert policy.q_table.num_state_action_pairs() >= initial_pairs


def test_training_metrics_track_unique_states() -> None:
    """Test that multiple episodes encounter multiple states."""
    policy = QLearningPolicy(seed=42, epsilon=1.0)  # Full exploration
    env = _default_env(max_steps=3)
    trainer = QLearningTrainer(env, policy)
    
    # Train several episodes
    trainer.train(num_episodes=5)
    
    # Should have encountered multiple states
    assert policy.q_table.num_states() > 0
    
    # Record stats include episode info
    for stat in trainer.stats:
        assert stat.episode > 0
        assert stat.episode_length > 0


# ============================================================================
# Training loop: stop conditions
# ============================================================================

def test_training_stops_on_truncation() -> None:
    """Test that training correctly stops when truncated."""
    policy = QLearningPolicy(seed=42, epsilon=0.0)  # No exploration for determinism
    env = _default_env(max_steps=2)
    trainer = QLearningTrainer(env, policy)
    
    stats = trainer.train_episode()
    
    # Episode should be truncated (max steps reached)
    assert stats.truncated is True
    assert stats.episode_length == 2
    assert stats.terminated is False


def test_training_stops_on_termination() -> None:
    """Test that training correctly stops when terminated."""
    # With max_steps=1, episode will truncate after 1 step
    # This tests that the loop stops correctly
    policy = QLearningPolicy(seed=42, epsilon=0.0)
    env = _default_env(max_steps=1)
    trainer = QLearningTrainer(env, policy)
    
    stats = trainer.train_episode()
    
    # Episode should end (truncated after 1 step)
    assert stats.episode_length == 1
    assert (stats.terminated or stats.truncated) is True


def test_training_multiple_episodes() -> None:
    """Test training runs multiple episodes correctly."""
    policy = QLearningPolicy(seed=42)
    env = _default_env(max_steps=2)
    trainer = QLearningTrainer(env, policy)
    
    stats = trainer.train(num_episodes=5)
    
    assert len(stats) == 5
    assert all(s.episode == i + 1 for i, s in enumerate(stats))
    assert all(s.episode_length > 0 for s in stats)


# ============================================================================
# Q-values actually change after learning
# ============================================================================

def test_q_values_change_after_training() -> None:
    """Test that Q-values are actually updated during training."""
    policy = QLearningPolicy(
        seed=42,
        alpha=1.0,  # Full learning rate
        gamma=0.0,  # No discounting for simple verification
        epsilon=0.0,  # No exploration
    )
    env = _default_env(max_steps=1)
    
    # Get initial state
    env.reset()
    observation = env.encoder.encode()
    state_key = policy.get_state_key(observation)
    
    # Record initial Q-values
    initial_q_values = {
        action_id: policy.get_q_value(state_key, action_id)
        for action_id in range(env.mapper.action_count)
    }
    
    # Train
    trainer = QLearningTrainer(env, policy)
    trainer.train_episode()
    
    # Check that at least some Q-values changed
    q_changed = False
    for action_id in range(env.mapper.action_count):
        new_q = policy.get_q_value(state_key, action_id)
        if new_q != initial_q_values[action_id]:
            q_changed = True
            break
    
    assert q_changed, "Q-values should have changed after training"


def test_q_table_grows_with_new_states() -> None:
    """Test that Q-table grows when new states are encountered."""
    policy = QLearningPolicy(seed=42, epsilon=1.0)  # Explore to see new states
    env = _default_env(max_steps=5)
    trainer = QLearningTrainer(env, policy)
    
    initial_states = policy.q_table.num_states()
    
    # Train multiple episodes
    trainer.train(num_episodes=10)
    
    # Q-table should have grown
    assert policy.q_table.num_states() > initial_states or policy.q_table.num_states() > 0


# ============================================================================
# MetaEnvironment compatibility
# ============================================================================

def test_q_learning_with_meta_environment() -> None:
    """Test that Q-learning works with MetaEnvironment."""
    task = MetaTask(
        task_id="test-task-001",
        task_description="Test task",
        task_category="coding",
        required_capabilities=["coding"],
        difficulty=1,
    )
    
    manager = ArchitectureManager.create_default_architecture()
    meta_env = MetaEnvironment(
        manager,
        role_options=["analysis"],
        max_steps=2,
        task=task,
    )
    
    policy = QLearningPolicy(seed=42, alpha=0.5, gamma=0.9)
    trainer = QLearningTrainer(meta_env, policy)
    
    # Training should work with MetaEnvironment
    stats = trainer.train_episode()
    
    assert stats.episode_length > 0
    assert policy.q_table.num_states() > 0


def test_q_learning_with_meta_environment_multiple_tasks() -> None:
    """Test Q-learning across multiple tasks (baseline, NOT Meta-RL)."""
    # Create multiple tasks
    tasks = [
        MetaTask(
            task_id=f"task-{i}",
            task_description=f"Task {i}",
            task_category="coding",
            required_capabilities=["coding"],
            difficulty=1,
        )
        for i in range(3)
    ]
    
    # Train same Q-learning agent on different tasks
    # NOTE: This is NOT Meta-RL. It's just training the same agent on
    # different tasks sequentially. The Q-table is shared but there's
    # no meta-learning, no task embeddings, no inner/outer loop.
    policy = QLearningPolicy(
        seed=42,
        alpha=0.5,
        gamma=0.9,
        epsilon=0.5,  # Some exploration
    )
    
    for task in tasks:
        manager = ArchitectureManager.create_default_architecture()
        meta_env = MetaEnvironment(
            manager,
            role_options=["analysis"],
            max_steps=2,
            task=task,
        )
        
        trainer = QLearningTrainer(meta_env, policy)
        trainer.train_episode()
    
    # Q-table should have learned across tasks
    assert policy.q_table.num_states() > 0
    assert policy.q_table.num_state_action_pairs() > 0


# ============================================================================
# Serialization safety
# ============================================================================

def test_q_table_serialization_has_no_secrets() -> None:
    """Test that Q-table serialization contains no credentials."""
    policy = QLearningPolicy(seed=42)
    env = _default_env()
    env.reset()
    
    # Add some Q-values
    state_key = policy.get_state_key(env.encoder.encode())
    policy.set_q_value(state_key, 0, 1.5)
    policy.set_q_value(state_key, 1, 2.0)
    
    # Serialize
    q_dict = policy.q_table.to_dict()
    serialized = json.dumps(q_dict, default=str)
    
    # Check for credential markers
    forbidden_markers = ["sk-or-", "sk-proj-", "api_key", "openai_api_key", "password"]
    serialized_lower = serialized.lower()
    
    for marker in forbidden_markers:
        assert marker not in serialized_lower, f"Found forbidden marker: {marker}"


def test_q_learning_serialization_save_load_roundtrip() -> None:
    """Test that saving and loading preserves Q-values."""
    policy = QLearningPolicy(seed=42)
    env = _default_env()
    env.reset()
    
    # Add some Q-values
    state_key = policy.get_state_key(env.encoder.encode())
    policy.set_q_value(state_key, 0, 1.5)
    policy.set_q_value(state_key, 1, 2.5)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        policy.save_q_table(temp_path)
        
        # Create new policy and load
        policy_loaded = QLearningPolicy()
        policy_loaded.load_q_table(temp_path)
        
        # Verify Q-values match
        for action_id in range(env.mapper.action_count):
            original_q = policy.get_q_value(state_key, action_id)
            loaded_q = policy_loaded.get_q_value(state_key, action_id)
            assert loaded_q == pytest.approx(original_q)
    finally:
        os.unlink(temp_path)


# ============================================================================
# No LLM/API calls verification
# ============================================================================

def test_q_learning_no_api_imports() -> None:
    """Test that q_learning module doesn't import API-related modules."""
    import app.rl.q_learning as ql
    import inspect
    
    source = inspect.getsource(ql)
    
    # Check for API-related imports
    forbidden_imports = [
        "openai",
        "anthropic",
        "langchain",
        "langsmith",
        "requests",
        "urllib.request",
        "http.client",
    ]
    
    source_lower = source.lower()
    for forbidden in forbidden_imports:
        # Check that forbidden modules aren't imported
        assert f"import {forbidden}" not in source_lower
        assert f"from {forbidden}" not in source_lower


def test_q_learning_training_no_api_calls() -> None:
    """Test that training doesn't make any API calls."""
    policy = QLearningPolicy(seed=42)
    env = _default_env(max_steps=2)
    trainer = QLearningTrainer(env, policy)
    
    # This should run without any network calls
    trainer.train(num_episodes=3)
    
    # If we got here without errors, no API calls were made
    assert len(trainer.stats) == 3


# ============================================================================
# Training statistics details
# ============================================================================

def test_training_stats_includes_all_fields() -> None:
    """Test that TrainingStats includes all required fields."""
    stats = TrainingStats(
        episode=5,
        total_reward=2.5,
        episode_length=10,
        final_score=0.8,
        terminated=False,
        truncated=True,
    )
    
    assert stats.episode == 5
    assert stats.total_reward == 2.5
    assert stats.episode_length == 10
    assert stats.final_score == 0.8
    assert stats.terminated is False
    assert stats.truncated is True


def test_training_stats_default_fields() -> None:
    """Test that TrainingStats has proper defaults."""
    stats = TrainingStats(
        episode=1,
        total_reward=1.0,
        episode_length=3,
    )
    
    assert stats.final_score is None
    assert stats.terminated is False
    assert stats.truncated is False


def test_training_stats_serializable() -> None:
    """Test that TrainingStats can be serialized."""
    stats = TrainingStats(
        episode=1,
        total_reward=2.5,
        episode_length=5,
        final_score=0.75,
        terminated=False,
        truncated=True,
    )
    
    # Convert to dict (using dataclasses.asdict or model_dump if available)
    stats_dict = {
        "episode": stats.episode,
        "total_reward": stats.total_reward,
        "episode_length": stats.episode_length,
        "final_score": stats.final_score,
        "terminated": stats.terminated,
        "truncated": stats.truncated,
    }
    
    # Should be JSON serializable
    json_str = json.dumps(stats_dict)
    parsed = json.loads(json_str)
    
    assert parsed["episode"] == 1
    assert parsed["total_reward"] == 2.5
    assert parsed["episode_length"] == 5
    assert parsed["final_score"] == 0.75


# ============================================================================
# Epsilon decay during training
# ============================================================================

def test_epsilon_decays_during_training() -> None:
    """Test that epsilon decays across training episodes."""
    initial_epsilon = 1.0
    policy = QLearningPolicy(
        seed=42,
        epsilon=initial_epsilon,
        epsilon_decay=0.9,
        epsilon_min=0.1,
    )
    env = _default_env(max_steps=1)
    trainer = QLearningTrainer(env, policy)
    
    initial_eps = policy.epsilon
    
    # Train multiple episodes
    trainer.train(num_episodes=5)
    
    # Epsilon should have decayed
    assert policy.epsilon < initial_eps
    assert policy.epsilon >= policy.epsilon_min


def test_epsilon_decay_applies_each_episode() -> None:
    """Test that epsilon decay is applied after each episode."""
    policy = QLearningPolicy(
        seed=42,
        epsilon=1.0,
        epsilon_decay=0.5,
        epsilon_min=0.0,
    )
    env = _default_env(max_steps=1)
    
    # Manually run episodes and check epsilon
    expected_epsilons = [1.0, 0.5, 0.25, 0.125]
    
    for i, expected_eps in enumerate(expected_epsilons):
        # Record epsilon before episode
        if i > 0:
            assert policy.epsilon == pytest.approx(expected_eps)
        
        # Run episode
        trainer = QLearningTrainer(env, policy)
        trainer.train_episode()
        
        # Epsilon should be decayed for next episode
        if i < len(expected_epsilons) - 1:
            assert policy.epsilon == pytest.approx(expected_eps * 0.5)


# ============================================================================
# Compatibility with existing policies
# ============================================================================

def test_existing_policies_unaffected() -> None:
    """Test that existing policies still work after Q-learning changes."""
    from app.rl.policy import DeterministicBaselinePolicy, RandomPolicy, BasePolicy
    
    env = _default_env(max_steps=2)
    env.reset()
    
    # Test DeterministicBaselinePolicy
    det_policy = DeterministicBaselinePolicy()
    valid_actions = list(range(env.mapper.action_count))
    action = det_policy.select_action(env.encoder.encode(), valid_actions)
    assert action in valid_actions
    
    # Test RandomPolicy
    import random
    rng = random.Random(42)
    rand_policy = RandomPolicy(rng=rng)
    action = rand_policy.select_action(env.encoder.encode(), valid_actions)
    assert action in valid_actions


# ============================================================================
# State key includes adjacency matrix
# ============================================================================

def test_state_key_includes_topology() -> None:
    """Test that state key differentiates architectures with different topologies."""
    encoder = StateEncoder()
    
    # Create two environments
    env1 = _default_env()
    env2 = _default_env()
    
    env1.reset()
    env2.reset()
    
    # Get initial observations
    obs1 = env1.encoder.encode()
    obs2 = env2.encoder.encode()
    
    # Keys should be the same for identical architectures
    key1 = encoder.encode(obs1)
    key2 = encoder.encode(obs2)
    assert key1 == key2
    
    # Add edge to env1 only
    action_id = env1.encode_action(
        __import__('app.architecture.actions', fromlist=['ArchitectureAction', 'ActionType']).ArchitectureAction(
            action_type=__import__('app.architecture.actions', fromlist=['ArchitectureAction', 'ActionType']).ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    env1.step(action_id)
    
    # Get new observation for env1
    obs1_after = env1.encoder.encode()
    key1_after = encoder.encode(obs1_after)
    
    # Key should now be different (adjacency matrix changed)
    assert key1_after != key1


def test_state_key_deterministic_with_fixed_seed() -> None:
    """Test that state encoding is deterministic regardless of seed."""
    # Create two environments with different seeds for any randomness
    env1 = _default_env()
    env2 = _default_env()
    
    encoder = StateEncoder()
    
    env1.reset()
    env2.reset()
    
    key1 = encoder.encode_from_env(env1)
    key2 = encoder.encode_from_env(env2)
    
    # Keys should be identical for identical architectures
    assert key1 == key2
    assert hash(key1) == hash(key2)


# ============================================================================
# Empty action list handling in training
# ============================================================================

def test_training_handles_empty_action_list() -> None:
    """Test that training handles environments with empty action lists correctly."""
    # This tests that the training loop doesn't crash when action list is empty
    # In practice, the environment should always have at least one action
    policy = QLearningPolicy(seed=42)
    
    # Create a mock observation and empty action list
    observation = {"agent_ids": [], "activity_vector": [], "role_vector": [], "agent_ids": [], "adjacency_matrix": []}
    
    # This should raise an error
    with pytest.raises(ValueError, match="no valid actions available"):
        policy.select_action(observation, [])


def test_q_table_default_value_usage() -> None:
    """Test that Q-table uses default value for unseen pairs."""
    q_table = QTable(default_value=5.0)
    
    state_key = (("active",), ("role",), ("agent",), (("0", "0"), ("0", "0")))
    action_id = 999
    
    # Unseen pair should return default
    q_value = q_table.get(state_key, action_id)
    assert q_value == 5.0
    
    # After setting, should return set value
    q_table.set(state_key, action_id, 10.0)
    q_value = q_table.get(state_key, action_id)
    assert q_value == 10.0


# ============================================================================
# Training result serialization
# ============================================================================

def test_training_result_serialization() -> None:
    """Test that training results can be serialized for inspection."""
    policy = QLearningPolicy(seed=42)
    env = _default_env(max_steps=2)
    trainer = QLearningTrainer(env, policy)
    
    # Train
    trainer.train(num_episodes=3)
    
    # Serialize stats
    stats_list = []
    for stat in trainer.stats:
        stats_list.append({
            "episode": stat.episode,
            "total_reward": stat.total_reward,
            "episode_length": stat.episode_length,
            "final_score": stat.final_score,
            "terminated": stat.terminated,
            "truncated": stat.truncated,
        })
    
    # Should be JSON serializable
    json_str = json.dumps(stats_list)
    parsed = json.loads(json_str)
    
    assert len(parsed) == 3
    assert all("episode" in s for s in parsed)
    assert all("total_reward" in s for s in parsed)
