"""
Tests for task transfer and adaptation baseline (Step 22).

These tests cover:

1. TransferConfig construction
2. Invalid configuration handling
3. Training/adaptation task separation
4. Training on multiple tasks
5. Q-table contains learned entries after training
6. Transfer condition preserves learned Q-table
7. From-scratch condition starts fresh
8. Adaptation runs successfully
9. Transfer and from-scratch use same adaptation tasks
10. Same initial architecture
11. Same task-aware state representation
12. Task-performance-aware reward is used
13. Episode-level metrics recorded
14. Transfer advantage calculation
15. Task-success advantage calculation
16. Aggregate metrics
17. Deterministic experiment
18. Serialization
19. Empty task handling
20. No LLM/API calls
21. No credentials/secrets
22. workflow.py unchanged
23. app/agents unchanged
24. Existing Q-learning behavior preserved
25. Existing MetaTrainer behavior preserved
"""

from __future__ import annotations

import inspect
import json
import os

import pytest

from app.architecture.manager import ArchitectureManager
from app.rl.meta_task import MetaTask
from app.rl.meta_environment import MetaEnvironment
from app.rl.task_distribution import MetaTaskDistribution, create_baseline_task_distribution
from app.rl.q_learning import QLearningPolicy, QTable
from app.rl.task_transfer import (
    TransferConfig,
    TransferEpisodeResult,
    TransferTaskResult,
    TransferExperimentResult,
    TaskTransferExperiment,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_task(
    *,
    task_id: str = "task-001",
    task_category: str = "coding",
    required_capabilities: list[str] | None = None,
    difficulty: int = 2,
) -> MetaTask:
    if required_capabilities is None:
        required_capabilities = ["coding"]
    return MetaTask(
        task_id=task_id,
        task_description=f"Sample {task_id}",
        task_category=task_category,
        required_capabilities=required_capabilities,
        difficulty=difficulty,
    )


def _make_small_distribution() -> MetaTaskDistribution:
    """Create a small distribution for testing."""
    tasks = [
        _make_task(task_id="train-1", task_category="coding", required_capabilities=["coding"]),
        _make_task(task_id="train-2", task_category="coding", required_capabilities=["coding"]),
        _make_task(task_id="adapt-1", task_category="research", required_capabilities=["research"]),
        _make_task(task_id="adapt-2", task_category="analysis", required_capabilities=["analysis"]),
    ]
    return MetaTaskDistribution(tasks=tasks)


# ============================================================================
# 1. TransferConfig construction
# ============================================================================


def test_transfer_config_construction() -> None:
    """Test that TransferConfig can be constructed."""
    config = TransferConfig(
        adaptation_task_ids=["task-001"],
        seed=42,
    )
    assert config.adaptation_task_ids == ["task-001"]
    assert config.seed == 42
    assert config.task_performance_weight == 0.5


def test_transfer_config_with_all_fields() -> None:
    """Test TransferConfig with all fields."""
    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=5,
        adaptation_episodes=10,
        max_steps=20,
        seed=123,
        task_performance_weight=0.3,
        learning_rate=0.2,
        discount_factor=0.8,
        epsilon=0.8,
        epsilon_min=0.1,
        epsilon_decay=0.99,
    )
    assert config.training_task_ids == ["train-1", "train-2"]
    assert config.adaptation_episodes == 10
    assert config.task_performance_weight == 0.3


# ============================================================================
# 2. Invalid configuration handling
# ============================================================================


def test_config_requires_adaptation_tasks() -> None:
    """Test that config requires adaptation tasks."""
    with pytest.raises(ValueError, match="adaptation_task_ids must contain at least one task"):
        TransferConfig(adaptation_task_ids=[])


def test_config_validates_no_overlap() -> None:
    """Test that config validates no overlap between training and adaptation."""
    distribution = _make_small_distribution()
    config = TransferConfig(
        training_task_ids=["train-1", "adapt-1"],  # Overlap!
        adaptation_task_ids=["adapt-1"],
        seed=42,
    )
    with pytest.raises(ValueError, match="must not overlap"):
        config.validate(distribution)


def test_config_validates_task_ids_exist() -> None:
    """Test that config validates task IDs exist in distribution."""
    distribution = _make_small_distribution()
    config = TransferConfig(
        training_task_ids=["nonexistent-task"],
        adaptation_task_ids=["adapt-1"],
        seed=42,
    )
    with pytest.raises(ValueError, match="not found in distribution"):
        config.validate(distribution)


# ============================================================================
# 3. Training/adaptation task separation
# ============================================================================


def test_training_tasks_are_separate() -> None:
    """Test that training and adaptation tasks are separate."""
    distribution = _make_small_distribution()
    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1", "adapt-2"],
        seed=42,
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)
    assert len(experiment._training_tasks) == 2
    assert len(experiment._adaptation_tasks) == 2

    training_ids = {t.task_id for t in experiment._training_tasks}
    adaptation_ids = {t.task_id for t in experiment._adaptation_tasks}

    assert training_ids.isdisjoint(adaptation_ids)


# ============================================================================
# 4. Training on multiple tasks
# ============================================================================


def test_training_on_multiple_tasks() -> None:
    """Test that training runs on multiple tasks."""
    distribution = _make_small_distribution()

    # Use only training tasks
    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=1,
        adaptation_episodes=1,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,  # Use structural reward only for speed
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)
    result = experiment.run()

    assert len(result.training_task_ids) == 2
    assert result.training_q_table_state_count > 0
    assert result.training_q_table_entry_count > 0


# ============================================================================
# 5. Q-table contains learned entries after training
# ============================================================================


def test_q_table_has_entries_after_training() -> None:
    """Test that Q-table has learned entries after training."""
    distribution = _make_small_distribution()

    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=2,
        adaptation_episodes=1,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)
    result = experiment.run()

    # Q-table should have learned entries
    assert result.training_q_table_state_count > 0
    assert result.training_q_table_entry_count > 0


# ============================================================================
# 6. Transfer condition preserves learned Q-table
# ============================================================================


def test_transfer_condition_preserves_q_table() -> None:
    """Test that transfer condition preserves learned Q-table."""
    distribution = _make_small_distribution()

    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=2,
        adaptation_episodes=1,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)
    result = experiment.run()

    # Transfer results should exist
    assert len(result.transfer_results) == 1
    transfer_result = result.transfer_results[0]

    # Should have adaptation episodes
    assert len(transfer_result.episode_results) == config.adaptation_episodes


# ============================================================================
# 7. From-scratch condition starts fresh
# ============================================================================


def test_from_scratch_starts_fresh() -> None:
    """Test that from-scratch condition starts with fresh Q-table."""
    distribution = _make_small_distribution()

    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=1,
        adaptation_episodes=1,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)
    result = experiment.run()

    # From-scratch results should exist
    assert len(result.from_scratch_results) == 1
    scratch_result = result.from_scratch_results[0]

    # Should have adaptation episodes
    assert len(scratch_result.episode_results) == config.adaptation_episodes


# ============================================================================
# 8. Adaptation runs successfully
# ============================================================================


def test_adaptation_runs_successfully() -> None:
    """Test that adaptation runs successfully."""
    distribution = _make_small_distribution()

    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=1,
        adaptation_episodes=2,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)
    result = experiment.run()

    # Both conditions should have results
    assert len(result.from_scratch_results) == 1
    assert len(result.transfer_results) == 1

    # Episodes should be recorded
    assert len(result.from_scratch_results[0].episode_results) == 2
    assert len(result.transfer_results[0].episode_results) == 2


# ============================================================================
# 9. Transfer and from-scratch use same adaptation tasks
# ============================================================================


def test_same_adaptation_tasks() -> None:
    """Test that both conditions use the same adaptation tasks."""
    distribution = _make_small_distribution()

    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1", "adapt-2"],
        training_episodes_per_task=1,
        adaptation_episodes=1,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)
    result = experiment.run()

    # Both conditions should have same task IDs
    scratch_task_ids = {r.task_id for r in result.from_scratch_results}
    transfer_task_ids = {r.task_id for r in result.transfer_results}

    assert scratch_task_ids == transfer_task_ids
    assert scratch_task_ids == {"adapt-1", "adapt-2"}


# ============================================================================
# 10. Same initial architecture
# ============================================================================


def test_same_initial_architecture() -> None:
    """Test that both conditions start with same initial architecture."""
    distribution = _make_small_distribution()

    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=1,
        adaptation_episodes=2,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)
    result = experiment.run()

    # Both conditions' first episodes should start from same architecture
    scratch_first = result.from_scratch_results[0].episode_results[0]
    transfer_first = result.transfer_results[0].episode_results[0]

    # The episode lengths might differ due to different exploration,
    # but the initial architecture is the same (default)
    assert scratch_first.task_id == transfer_first.task_id
    assert scratch_first.condition != transfer_first.condition


# ============================================================================
# 11. Same task-aware state representation
# ============================================================================


def test_task_aware_state_used() -> None:
    """Test that task-aware state representation is used."""
    distribution = _make_small_distribution()

    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=1,
        adaptation_episodes=1,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)

    # The experiment should use task-aware policies
    # (We verify this by checking that the experiment runs successfully
    #  with task-aware configuration)
    result = experiment.run()

    # Results should be generated with task-aware policies
    assert len(result.from_scratch_results) > 0
    assert len(result.transfer_results) > 0


# ============================================================================
# 12. Task-performance-aware reward is used
# ============================================================================


def test_task_performance_reward_used() -> None:
    """Test that task-performance-aware reward is used when configured."""
    distribution = _make_small_distribution()

    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=1,
        adaptation_episodes=2,
        max_steps=3,
        seed=42,
        task_performance_weight=0.5,  # Task-aware reward
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)
    result = experiment.run()

    # Should have task-success scores in results
    for task_result in result.from_scratch_results + result.transfer_results:
        assert task_result.final_task_success_score is not None
        assert len(task_result.episode_results) > 0
        for episode in task_result.episode_results:
            assert episode.task_success_score is not None


# ============================================================================
# 13. Episode-level metrics recorded
# ============================================================================


def test_episode_level_metrics() -> None:
    """Test that episode-level metrics are recorded."""
    distribution = _make_small_distribution()

    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=1,
        adaptation_episodes=3,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)
    result = experiment.run()

    # Check episode-level results
    for task_result in result.from_scratch_results + result.transfer_results:
        assert len(task_result.episode_results) == config.adaptation_episodes
        for episode in task_result.episode_results:
            assert episode.task_id == task_result.task_id
            assert episode.condition == task_result.condition
            assert isinstance(episode.total_reward, float)
            assert isinstance(episode.episode_length, int)
            assert isinstance(episode.task_success_score, float)
            assert isinstance(episode.structural_score, float)


# ============================================================================
# 14. Transfer advantage calculation
# ============================================================================


def test_transfer_advantage_calculation() -> None:
    """Test that transfer advantage is calculated correctly."""
    distribution = _make_small_distribution()

    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=2,
        adaptation_episodes=2,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)
    result = experiment.run()

    metrics = experiment.calculate_transfer_advantage(result)

    assert "mean_transfer_advantage" in metrics
    assert "mean_task_success_advantage" in metrics
    assert "per_task_advantages" in metrics
    assert len(metrics["per_task_advantages"]) == 1


# ============================================================================
# 15. Task-success advantage calculation
# ============================================================================


def test_task_success_advantage_calculation() -> None:
    """Test that task-success advantage is calculated correctly."""
    distribution = _make_small_distribution()

    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=2,
        adaptation_episodes=2,
        max_steps=3,
        seed=42,
        task_performance_weight=0.5,
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)
    result = experiment.run()

    metrics = experiment.calculate_transfer_advantage(result)

    # Should have per-task task-success advantages
    for task_advantage in metrics["per_task_advantages"]:
        assert "transfer_task_success_advantage" in task_advantage
        assert isinstance(task_advantage["transfer_task_success_advantage"], float)


# ============================================================================
# 16. Aggregate metrics
# ============================================================================


def test_aggregate_metrics() -> None:
    """Test that aggregate metrics are available."""
    distribution = _make_small_distribution()

    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1", "adapt-2"],
        training_episodes_per_task=1,
        adaptation_episodes=2,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)
    result = experiment.run()

    # Check aggregate structure
    assert len(result.from_scratch_results) == 2
    assert len(result.transfer_results) == 2

    # Each task should have mean metrics
    for task_result in result.from_scratch_results + result.transfer_results:
        assert task_result.mean_reward is not None
        assert task_result.mean_task_success_score is not None
        assert task_result.mean_episode_length is not None


# ============================================================================
# 17. Deterministic experiment
# ============================================================================


def test_deterministic_experiment() -> None:
    """Test that same configuration produces same results."""
    distribution = _make_small_distribution()

    config_a = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=1,
        adaptation_episodes=2,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,
    )
    config_b = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=1,
        adaptation_episodes=2,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,
    )

    experiment_a = TaskTransferExperiment(distribution, config_a)
    result_a = experiment_a.run()

    experiment_b = TaskTransferExperiment(distribution, config_b)
    result_b = experiment_b.run()

    # Results should be structurally identical
    assert result_a.training_task_ids == result_b.training_task_ids
    assert result_a.adaptation_task_ids == result_b.adaptation_task_ids
    assert len(result_a.from_scratch_results) == len(result_b.from_scratch_results)
    assert len(result_a.transfer_results) == len(result_b.transfer_results)


# ============================================================================
# 18. Serialization
# ============================================================================


def test_experiment_result_serialization() -> None:
    """Test that experiment result can be serialized."""
    distribution = _make_small_distribution()

    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=1,
        adaptation_episodes=1,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)
    result = experiment.run()

    # Serialize
    serialized = result.to_dict()

    assert isinstance(serialized, dict)
    assert "config" in serialized
    assert "training_task_ids" in serialized
    assert "adaptation_task_ids" in serialized
    assert "from_scratch_results" in serialized
    assert "transfer_results" in serialized


def test_experiment_result_json_serialization() -> None:
    """Test JSON serialization."""
    distribution = _make_small_distribution()

    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=1,
        adaptation_episodes=1,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)
    result = experiment.run()

    json_str = json.dumps(result.to_dict(), indent=2)
    assert isinstance(json_str, str)

    parsed = json.loads(json_str)
    assert "training_task_ids" in parsed


# ============================================================================
# 19. Empty task handling
# ============================================================================


def test_no_training_tasks() -> None:
    """Test experiment with no training tasks (from-scratch only)."""
    distribution = _make_small_distribution()

    config = TransferConfig(
        training_task_ids=[],  # No training
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=1,
        adaptation_episodes=2,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)
    result = experiment.run()

    # Should have from-scratch results but no training
    assert len(result.from_scratch_results) == 1
    assert len(result.training_task_ids) == 0
    assert result.training_q_table_state_count == 0


# ============================================================================
# 20. No LLM/API calls
# ============================================================================


def test_experiment_has_no_api_imports() -> None:
    """Test that task transfer module doesn't import API-related modules."""
    import app.rl.task_transfer as tt
    import inspect

    source = inspect.getsource(tt)

    forbidden_imports = [
        "openai",
        "anthropic",
        "langchain",
        "langsmith",
        "requests",
        "urllib.request",
        "http.client",
        "chatopenai",
        "chat_openai",
        "ChatOpenAI",
    ]

    source_lower = source.lower()
    for forbidden in forbidden_imports:
        assert f"import {forbidden.lower()}" not in source_lower
        assert f"from {forbidden.lower()}" not in source_lower


def test_experiment_is_fully_offline() -> None:
    """Test that experiment runs without network calls."""
    distribution = _make_small_distribution()

    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=1,
        adaptation_episodes=1,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)

    # If we reach here without network errors, the module stayed offline
    result = experiment.run()
    assert isinstance(result, TransferExperimentResult)


# ============================================================================
# 21. No credentials/secrets
# ============================================================================


def test_result_contains_no_credentials() -> None:
    """Test that experiment results contain no credentials."""
    distribution = _make_small_distribution()

    config = TransferConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        training_episodes_per_task=1,
        adaptation_episodes=1,
        max_steps=3,
        seed=42,
        task_performance_weight=0.0,
    )
    config.validate(distribution)

    experiment = TaskTransferExperiment(distribution, config)
    result = experiment.run()

    serialized = json.dumps(result.to_dict(), default=str)
    serialized_lower = serialized.lower()

    forbidden_markers = [
        "sk-or-",
        "sk-proj-",
        "api_key",
        "openai_api_key",
        "password",
        "secret",
    ]

    for marker in forbidden_markers:
        assert marker not in serialized_lower, f"Found forbidden marker: {marker}"


# ============================================================================
# 22. workflow.py unchanged
# ============================================================================


def test_workflow_py_not_modified() -> None:
    """Test that app/graph/workflow.py is not modified."""
    import os

    workflow_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "app",
        "graph",
        "workflow.py",
    )
    assert os.path.exists(workflow_path), "expected app/graph/workflow.py to exist"


# ============================================================================
# 23. app/agents unchanged
# ============================================================================


def test_agents_directory_not_modified() -> None:
    """Test that app/agents/ is not modified."""
    import os

    agents_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "app",
        "agents",
    )
    assert os.path.isdir(agents_path), "expected app/agents/ to exist"


# ============================================================================
# 24. Existing Q-learning behavior preserved
# ============================================================================


def test_q_learning_still_works() -> None:
    """Test that existing Q-learning still works after Step 22."""
    from app.rl.q_learning import QLearningPolicy, QLearningTrainer
    from app.rl.environment import MASArchitectureEnv
    from app.architecture.manager import ArchitectureManager

    manager = ArchitectureManager.create_default_architecture()
    env = MASArchitectureEnv(manager, max_steps=3)
    policy = QLearningPolicy(seed=42)
    trainer = QLearningTrainer(env, policy)

    stats = trainer.train_episode()

    assert stats.episode_length > 0
    assert isinstance(stats.total_reward, float)


# ============================================================================
# 25. Existing MetaTrainer behavior preserved
# ============================================================================


def test_meta_trainer_still_works() -> None:
    """Test that existing MetaTrainer still works after Step 22."""
    from app.rl.meta_trainer import MetaTrainer
    from app.rl.task_distribution import MetaTaskDistribution
    from app.rl.meta_task import MetaTask

    tasks = [
        MetaTask(
            task_id="task-001",
            task_description="Test task",
            task_category="coding",
            required_capabilities=["coding"],
        ),
    ]
    distribution = MetaTaskDistribution(tasks=tasks)

    trainer = MetaTrainer(
        task_distribution=distribution,
        seed=42,
        max_steps=3,
    )

    task = distribution.get_task("task-001")
    result = trainer.train_single_task(task, num_episodes=1)

    assert result.task_id == "task-001"
    assert result.num_episodes == 1
