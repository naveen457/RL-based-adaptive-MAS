"""
Tests for task-aware RL state integration (Step 17).

These tests verify that the RL state can include both:
1. Current MAS architecture state
2. Current MetaTask/task context

Coverage targets:
1. Task-aware state construction
2. Deterministic encoding
3. Same task produces same key
4. Different tasks produce different keys
5. Architecture changes produce different keys
6. Architecture + task combination
7. Q-learning compatibility
8. MetaEnvironment compatibility
9. Reset behavior
10. Serialization
11. No LLM/API calls
12. No credentials/secrets

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
    TaskAwareStateEncoder,
    TrainingStats,
)
from app.rl.meta_environment import MetaEnvironment
from app.rl.meta_task import MetaTask, MetaTaskContext


def _default_env(**kwargs) -> MASArchitectureEnv:
    """Create a default environment for testing."""
    manager = ArchitectureManager.create_default_architecture()
    return MASArchitectureEnv(manager, **kwargs)


def _create_sample_task(
    task_id: str = "test-task-001",
    task_category: str = "coding",
    difficulty: int = 2,
) -> MetaTask:
    """Create a sample MetaTask for testing."""
    return MetaTask(
        task_id=task_id,
        task_description=f"Sample {task_category} task",
        task_category=task_category,
        required_capabilities=["coding", "testing"],
        difficulty=difficulty,
        context_features={"language": "python", "complexity": "medium"},
    )


# ============================================================================
# Task-aware state construction tests
# ============================================================================

def test_task_aware_state_encoder_construction() -> None:
    """Test that TaskAwareStateEncoder can be constructed."""
    encoder = TaskAwareStateEncoder()
    assert encoder is not None


def test_task_aware_state_encoder_produces_tuple() -> None:
    """Test that TaskAwareStateEncoder produces a hashable tuple."""
    encoder = TaskAwareStateEncoder()
    env = _default_env()
    env.reset()
    observation = env.encoder.encode()
    
    state_key = encoder.encode(observation)
    
    assert isinstance(state_key, tuple)
    # Should be hashable
    hash(state_key)


def test_task_aware_state_encoder_with_task_context() -> None:
    """Test that TaskAwareStateEncoder includes task context."""
    encoder = TaskAwareStateEncoder()
    env = _default_env()
    env.reset()
    observation = env.encoder.encode()
    
    task = _create_sample_task()
    task_context = task.to_context()
    
    state_key_with_task = encoder.encode(observation, task_context)
    state_key_without_task = encoder.encode(observation, None)
    
    # State key with task should have more components
    assert len(state_key_with_task) == 8  # 4 architecture + 4 task components
    assert len(state_key_without_task) == 8  # Still 8, but with default task values
    
    # Task components should be populated
    assert state_key_with_task[4] == "coding"  # task_category
    assert state_key_with_task[5] == ("coding", "testing")  # required_capabilities
    assert state_key_with_task[6] == 2  # difficulty
    assert len(state_key_with_task[7]) > 0  # context_features


# ============================================================================
# Deterministic encoding tests
# ============================================================================

def test_task_aware_encoding_is_deterministic() -> None:
    """Test that task-aware encoding is deterministic for same inputs."""
    encoder = TaskAwareStateEncoder()
    env = _default_env()
    env.reset()
    observation = env.encoder.encode()
    
    task = _create_sample_task()
    task_context = task.to_context()
    
    key_a = encoder.encode(observation, task_context)
    key_b = encoder.encode(observation, task_context)
    
    assert key_a == key_b
    assert hash(key_a) == hash(key_b)


def test_task_aware_encoding_same_architecture_same_task() -> None:
    """Test that same architecture + same task produces same key."""
    encoder = TaskAwareStateEncoder()
    
    env_a = _default_env()
    env_b = _default_env()
    env_a.reset()
    env_b.reset()
    
    observation_a = env_a.encoder.encode()
    observation_b = env_b.encoder.encode()
    
    task = _create_sample_task()
    task_context = task.to_context()
    
    key_a = encoder.encode(observation_a, task_context)
    key_b = encoder.encode(observation_b, task_context)
    
    assert key_a == key_b


# ============================================================================
# Same task produces same key tests
# ============================================================================

def test_same_task_produces_same_key() -> None:
    """Test that same task produces same task context components."""
    encoder = TaskAwareStateEncoder()
    env = _default_env()
    env.reset()
    observation = env.encoder.encode()
    
    task_a = _create_sample_task(task_id="task-001", task_category="coding")
    task_b = _create_sample_task(task_id="task-001", task_category="coding")
    
    key_a = encoder.encode(observation, task_a.to_context())
    key_b = encoder.encode(observation, task_b.to_context())
    
    assert key_a == key_b


def test_different_tasks_produce_different_keys() -> None:
    """Test that different tasks produce different keys."""
    encoder = TaskAwareStateEncoder()
    env = _default_env()
    env.reset()
    observation = env.encoder.encode()
    
    task_a = _create_sample_task(task_id="task-coding", task_category="coding", difficulty=2)
    task_b = _create_sample_task(task_id="task-research", task_category="research", difficulty=3)
    
    key_a = encoder.encode(observation, task_a.to_context())
    key_b = encoder.encode(observation, task_b.to_context())
    
    assert key_a != key_b
    # Task category should differ
    assert key_a[4] != key_b[4]


# ============================================================================
# Architecture changes produce different keys tests
# ============================================================================

def test_architecture_changes_produce_different_keys() -> None:
    """Test that architecture changes produce different keys."""
    encoder = TaskAwareStateEncoder()
    env = _default_env()
    env.reset()
    
    task = _create_sample_task()
    task_context = task.to_context()
    
    # Initial architecture
    observation_before = env.encoder.encode()
    key_before = encoder.encode(observation_before, task_context)
    
    # Add an edge to change architecture
    from app.architecture.actions import ActionType, ArchitectureAction
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    env.step(action_id)
    
    # Changed architecture
    observation_after = env.encoder.encode()
    key_after = encoder.encode(observation_after, task_context)
    
    assert key_before != key_after
    # Adjacency matrix should differ
    assert key_before[3] != key_after[3]


def test_same_task_different_architecture() -> None:
    """Test that same task + different architecture produces different keys."""
    encoder = TaskAwareStateEncoder()
    
    env_a = _default_env()
    env_b = _default_env()
    env_a.reset()
    env_b.reset()
    
    # Change env_b's architecture by deactivating an agent
    from app.architecture.actions import ActionType, ArchitectureAction
    action_id = env_b.encode_action(
        ArchitectureAction(
            action_type=ActionType.DEACTIVATE_AGENT,
            agent_id="researcher",
        )
    )
    env_b.step(action_id)
    
    observation_a = env_a.encoder.encode()
    observation_b = env_b.encoder.encode()
    
    task = _create_sample_task()
    task_context = task.to_context()
    
    key_a = encoder.encode(observation_a, task_context)
    key_b = encoder.encode(observation_b, task_context)
    
    assert key_a != key_b


# ============================================================================
# Architecture + task combination tests
# ============================================================================

def test_architecture_task_combination() -> None:
    """Test that architecture and task components are combined correctly."""
    encoder = TaskAwareStateEncoder()
    env = _default_env()
    env.reset()
    observation = env.encoder.encode()
    
    task = _create_sample_task(
        task_id="test-task",
        task_category="analysis",
        difficulty=4,
    )
    task_context = task.to_context()
    
    state_key = encoder.encode(observation, task_context)
    
    # Verify architecture components
    assert state_key[0] == tuple(observation["activity_vector"])
    assert state_key[1] == tuple(observation["role_vector"])
    assert state_key[2] == tuple(observation["agent_ids"])
    assert state_key[3] == tuple(tuple(row) for row in observation["adjacency_matrix"])
    
    # Verify task components
    assert state_key[4] == "analysis"
    assert state_key[5] == tuple(["coding", "testing"])
    assert state_key[6] == 4
    assert len(state_key[7]) > 0


def test_task_context_with_empty_features() -> None:
    """Test task context with empty context_features."""
    encoder = TaskAwareStateEncoder()
    env = _default_env()
    env.reset()
    observation = env.encoder.encode()
    
    task = MetaTask(
        task_id="test-task",
        task_description="Test task",
        task_category="research",
        required_capabilities=[],
        difficulty=1,
        context_features={},
    )
    task_context = task.to_context()
    
    state_key = encoder.encode(observation, task_context)
    
    # Empty context_features should produce empty tuple
    assert state_key[7] == ()


# ============================================================================
# Q-learning compatibility tests
# ============================================================================

def test_task_aware_q_learning_policy_construction() -> None:
    """Test that task-aware Q-learning policy can be constructed."""
    policy = QLearningPolicy(task_aware=True, seed=42)
    
    assert policy is not None
    assert policy.task_aware is True
    assert isinstance(policy.state_encoder, TaskAwareStateEncoder)


def test_architecture_only_q_learning_policy_still_works() -> None:
    """Test that architecture-only Q-learning policy still works (backward compatibility)."""
    policy = QLearningPolicy(task_aware=False, seed=42)
    
    assert policy is not None
    assert policy.task_aware is False
    assert isinstance(policy.state_encoder, StateEncoder)
    
    # Should work normally
    env = _default_env()
    env.reset()
    observation = env.encoder.encode()
    valid_actions = [0, 1, 2]
    
    action = policy.select_action(observation, valid_actions)
    assert action in valid_actions


def test_task_aware_policy_selects_valid_actions() -> None:
    """Test that task-aware policy selects valid actions."""
    policy = QLearningPolicy(task_aware=True, seed=42, epsilon=0.0)
    
    env = _default_env()
    env.reset()
    observation = env.encoder.encode()
    valid_actions = list(range(env.mapper.action_count))
    
    task = _create_sample_task()
    task_context = task.to_context()
    
    action = policy.select_action(observation, valid_actions, task_context)
    assert action in valid_actions


def test_task_aware_policy_with_different_tasks() -> None:
    """Test that task-aware policy handles different tasks."""
    policy = QLearningPolicy(task_aware=True, seed=42, epsilon=0.0)
    
    env = _default_env()
    env.reset()
    observation = env.encoder.encode()
    valid_actions = [0, 1, 2]
    
    task_a = _create_sample_task(task_category="coding")
    task_b = _create_sample_task(task_category="research")
    
    # Set Q-values for both task states
    state_key_a = policy.get_state_key(observation, task_a.to_context())
    state_key_b = policy.get_state_key(observation, task_b.to_context())
    
    policy.set_q_value(state_key_a, 0, 1.0)
    policy.set_q_value(state_key_b, 1, 2.0)
    
    # Should select different best actions for different tasks
    action_a = policy.select_action(observation, valid_actions, task_a.to_context())
    action_b = policy.select_action(observation, valid_actions, task_b.to_context())
    
    # Verify Q-values were set correctly
    assert policy.get_q_value(state_key_a, 0) == 1.0
    assert policy.get_q_value(state_key_b, 1) == 2.0


# ============================================================================
# MetaEnvironment compatibility tests
# ============================================================================

def test_task_aware_policy_with_meta_environment() -> None:
    """Test task-aware policy works with MetaEnvironment."""
    task = _create_sample_task()
    
    manager = ArchitectureManager.create_default_architecture()
    meta_env = MetaEnvironment(
        manager,
        role_options=["analysis"],
        max_steps=3,
        task=task,
    )
    
    policy = QLearningPolicy(task_aware=True, seed=42, epsilon=0.5)
    trainer = QLearningTrainer(meta_env, policy)
    
    # Training should work
    stats = trainer.train_episode()
    
    assert stats.episode_length > 0
    assert policy.q_table.num_states() > 0


def test_meta_environment_task_context_preserved() -> None:
    """Test that MetaEnvironment preserves task context in observations."""
    task = _create_sample_task(task_category="coding", difficulty=3)
    
    manager = ArchitectureManager.create_default_architecture()
    meta_env = MetaEnvironment(
        manager,
        role_options=["analysis"],
        max_steps=2,
        task=task,
    )
    
    observation, _ = meta_env.reset()
    
    # Observation should include task_context
    assert "task_context" in observation
    assert observation["task_context"]["task_category"] == "coding"
    assert observation["task_context"]["difficulty"] == 3


def test_task_aware_encoder_with_meta_environment() -> None:
    """Test TaskAwareStateEncoder works with MetaEnvironment observations."""
    task = _create_sample_task()
    
    manager = ArchitectureManager.create_default_architecture()
    meta_env = MetaEnvironment(
        manager,
        role_options=["analysis"],
        max_steps=2,
        task=task,
    )
    
    encoder = TaskAwareStateEncoder()
    observation, _ = meta_env.reset()
    
    # Encode using task context from environment
    state_key = encoder.encode_from_env(meta_env)
    
    assert isinstance(state_key, tuple)
    assert len(state_key) == 8  # Full task-aware state


# ============================================================================
# Reset behavior tests
# ============================================================================

def test_task_aware_training_resets_correctly() -> None:
    """Test that task-aware training resets environment correctly."""
    task = _create_sample_task()
    
    manager = ArchitectureManager.create_default_architecture()
    meta_env = MetaEnvironment(
        manager,
        role_options=["analysis"],
        max_steps=2,
        task=task,
    )
    
    policy = QLearningPolicy(task_aware=True, seed=42, epsilon=0.0)
    trainer = QLearningTrainer(meta_env, policy)
    
    # First episode
    stats_a = trainer.train_episode()
    
    # Second episode (should reset)
    stats_b = trainer.train_episode()
    
    # Both episodes should have same length (same task, same architecture)
    assert stats_a.episode_length == stats_b.episode_length
    assert stats_a.episode == 1
    assert stats_b.episode == 2


# ============================================================================
# Serialization tests
# ============================================================================

def test_task_aware_q_table_serialization() -> None:
    """Test that task-aware Q-table can be serialized."""
    policy = QLearningPolicy(task_aware=True, seed=42)
    env = _default_env()
    env.reset()
    
    task = _create_sample_task()
    task_context = task.to_context()
    
    state_key = policy.get_state_key(env.encoder.encode(), task_context)
    policy.set_q_value(state_key, 0, 1.5)
    
    # Serialize
    q_dict = policy.q_table.to_dict()
    
    assert "default_value" in q_dict
    assert "entries" in q_dict
    assert len(q_dict["entries"]) == 1
    assert q_dict["entries"][0]["q_value"] == 1.5


def test_task_aware_q_table_serialization_has_no_secrets() -> None:
    """Test that task-aware Q-table serialization contains no credentials."""
    policy = QLearningPolicy(task_aware=True, seed=42)
    env = _default_env()
    env.reset()
    
    task = _create_sample_task()
    task_context = task.to_context()
    
    state_key = policy.get_state_key(env.encoder.encode(), task_context)
    policy.set_q_value(state_key, 0, 1.5)
    
    # Serialize
    q_dict = policy.q_table.to_dict()
    serialized = json.dumps(q_dict, default=str)
    
    # Check for credential markers
    forbidden_markers = ["sk-or-", "sk-proj-", "api_key", "openai_api_key", "password"]
    serialized_lower = serialized.lower()
    
    for marker in forbidden_markers:
        assert marker not in serialized_lower, f"Found forbidden marker: {marker}"


def test_task_aware_save_load_roundtrip() -> None:
    """Test that task-aware Q-table save/load preserves values."""
    policy = QLearningPolicy(task_aware=True, seed=42)
    env = _default_env()
    env.reset()
    
    task = _create_sample_task()
    task_context = task.to_context()
    
    state_key = policy.get_state_key(env.encoder.encode(), task_context)
    policy.set_q_value(state_key, 0, 1.5)
    policy.set_q_value(state_key, 1, 2.5)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        policy.save_q_table(temp_path)
        
        # Create new policy and load
        policy_loaded = QLearningPolicy(task_aware=True)
        policy_loaded.load_q_table(temp_path)
        
        # Verify Q-values match
        loaded_q_0 = policy_loaded.get_q_value(state_key, 0)
        loaded_q_1 = policy_loaded.get_q_value(state_key, 1)
        
        assert loaded_q_0 == pytest.approx(1.5)
        assert loaded_q_1 == pytest.approx(2.5)
    finally:
        os.unlink(temp_path)


# ============================================================================
# No LLM/API calls tests
# ============================================================================

def test_task_aware_no_api_imports() -> None:
    """Test that task-aware code doesn't import API-related modules."""
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
        assert f"import {forbidden}" not in source_lower
        assert f"from {forbidden}" not in source_lower


def test_task_aware_training_no_api_calls() -> None:
    """Test that task-aware training doesn't make API calls."""
    task = _create_sample_task()
    
    manager = ArchitectureManager.create_default_architecture()
    meta_env = MetaEnvironment(
        manager,
        role_options=["analysis"],
        max_steps=2,
        task=task,
    )
    
    policy = QLearningPolicy(task_aware=True, seed=42)
    trainer = QLearningTrainer(meta_env, policy)
    
    # This should run without any network calls
    trainer.train(num_episodes=3)
    
    # If we got here without errors, no API calls were made
    assert len(trainer.stats) == 3


# ============================================================================
# Edge case tests
# ============================================================================

def test_task_aware_with_no_task_context() -> None:
    """Test task-aware encoder with no task context (uses defaults)."""
    encoder = TaskAwareStateEncoder()
    env = _default_env()
    env.reset()
    observation = env.encoder.encode()
    
    # Encode without task context
    state_key = encoder.encode(observation, None)
    
    # Should use default values
    assert state_key[4] == ""  # empty task_category
    assert state_key[5] == ()  # empty required_capabilities
    assert state_key[6] == 1  # default difficulty
    assert state_key[7] == ()  # empty context_features


def test_task_aware_deterministic_tie_breaking() -> None:
    """Test that task-aware policy uses deterministic tie-breaking."""
    policy = QLearningPolicy(task_aware=True, seed=42, epsilon=0.0)
    
    env = _default_env()
    env.reset()
    observation = env.encoder.encode()
    valid_actions = [5, 3, 7, 1]
    
    task = _create_sample_task()
    task_context = task.to_context()
    
    # All Q-values are equal (0.0 by default)
    action = policy.select_action(observation, valid_actions, task_context)
    
    # Should select the smallest action_id (deterministic tie-breaking)
    assert action == min(valid_actions)


def test_task_aware_policy_rejects_empty_actions() -> None:
    """Test that task-aware policy raises on empty action list."""
    policy = QLearningPolicy(task_aware=True, seed=42)
    observation = {}
    task = _create_sample_task()
    
    with pytest.raises(ValueError, match="no valid actions available"):
        policy.select_action(observation, [], task.to_context())


# ============================================================================
# Training metrics tests
# ============================================================================

def test_task_aware_training_records_statistics() -> None:
    """Test that task-aware training records statistics correctly."""
    task = _create_sample_task()
    
    manager = ArchitectureManager.create_default_architecture()
    meta_env = MetaEnvironment(
        manager,
        role_options=["analysis"],
        max_steps=2,
        task=task,
    )
    
    policy = QLearningPolicy(task_aware=True, seed=42)
    trainer = QLearningTrainer(meta_env, policy)
    
    trainer.train(num_episodes=3)
    
    assert len(trainer.stats) == 3
    for stat in trainer.stats:
        assert stat.episode > 0
        assert stat.episode_length > 0
        assert isinstance(stat.total_reward, float)


def test_task_aware_training_matches_architecture_only() -> None:
    """Test that task-aware and architecture-only training can both work."""
    # Architecture-only
    env_a = _default_env(max_steps=2)
    policy_a = QLearningPolicy(task_aware=False, seed=42)
    trainer_a = QLearningTrainer(env_a, policy_a)
    stats_a = trainer_a.train(num_episodes=2)
    
    # Task-aware
    task = _create_sample_task()
    manager = ArchitectureManager.create_default_architecture()
    env_b = MetaEnvironment(
        manager,
        role_options=["analysis"],
        max_steps=2,
        task=task,
    )
    policy_b = QLearningPolicy(task_aware=True, seed=42)
    trainer_b = QLearningTrainer(env_b, policy_b)
    stats_b = trainer_b.train(num_episodes=2)
    
    # Both should have 2 episodes
    assert len(stats_a) == 2
    assert len(stats_b) == 2
    
    # Both should have recorded states
    assert policy_a.q_table.num_states() > 0
    assert policy_b.q_table.num_states() > 0
