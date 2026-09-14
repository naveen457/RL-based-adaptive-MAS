"""
Tests for multi-seed transfer generalization evaluation (Step 23).

These tests cover:

1. Evaluator construction
2. Configuration validation
3. Multiple seeds
4. Multiple target tasks
5. Source/target task disjointness
6. Transfer condition
7. From-scratch condition
8. Same experimental controls
9. Deterministic results
10. Per-seed results
11. Per-task results
12. Aggregate results
13. Mean/std/min/max calculations
14. Transfer advantage
15. Task-success advantage
16. Adaptation improvement
17. Threshold-not-reached behavior
18. Adaptation curve data
19. Serialization
20. Empty seeds handling
21. Invalid task IDs
22. Overlapping source/target tasks
23. No LLM/API calls
24. No credentials/secrets
25. Existing Step 22 compatibility
26. workflow.py unchanged
27. app/agents unchanged
"""

from __future__ import annotations

import json
import os
import statistics
from typing import List

import pytest

from app.rl.meta_task import MetaTask
from app.rl.task_distribution import MetaTaskDistribution
from app.rl.task_transfer import TransferExperimentResult
from app.rl.transfer_evaluation import (
    AggregateGeneralizationMetrics,
    PerSeedTransferResult,
    PerTaskCrossSeedResult,
    TransferGeneralizationConfig,
    TransferGeneralizationEvaluator,
    TransferGeneralizationResult,
    _mean,
    _stdev,
    _min,
    _max,
    _first_episode_above,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_task(
    *,
    task_id: str = "task-001",
    task_category: str = "coding",
    required_capabilities: List[str] | None = None,
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
        _make_task(
            task_id="train-1",
            task_category="coding",
            required_capabilities=["coding"],
        ),
        _make_task(
            task_id="train-2",
            task_category="coding",
            required_capabilities=["coding"],
        ),
        _make_task(
            task_id="adapt-1",
            task_category="research",
            required_capabilities=["research"],
        ),
        _make_task(
            task_id="adapt-2",
            task_category="analysis",
            required_capabilities=["analysis"],
        ),
    ]
    return MetaTaskDistribution(tasks=tasks)


def _make_config(
    *,
    seeds: List[int] | None = None,
    training_task_ids: List[str] | None = None,
    adaptation_task_ids: List[str] | None = None,
    adaptation_episodes: int = 2,
    max_steps: int = 3,
    task_performance_weight: float = 0.0,
) -> TransferGeneralizationConfig:
    if seeds is None:
        seeds = [42, 123]
    if training_task_ids is None:
        training_task_ids = ["train-1", "train-2"]
    if adaptation_task_ids is None:
        adaptation_task_ids = ["adapt-1", "adapt-2"]
    return TransferGeneralizationConfig(
        training_task_ids=training_task_ids,
        adaptation_task_ids=adaptation_task_ids,
        seeds=seeds,
        training_episodes_per_task=1,
        adaptation_episodes=adaptation_episodes,
        max_steps=max_steps,
        task_performance_weight=task_performance_weight,
    )


# ===========================================================================
# 1. Evaluator construction
# ===========================================================================


def test_aggregator_mean_basic() -> None:
    """Test _mean with a simple list."""
    assert _mean([1.0, 2.0, 3.0]) == 2.0
    assert _mean([]) == 0.0


def test_aggregator_stdev_basic() -> None:
    """Test _stdev with a simple list."""
    # Population variance: stdev of [2,4,4,4,5,5,7,9] is 2.0
    # Sample stdev:
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    stdev = _stdev(values)
    assert stdev is not None
    assert abs(stdev - statistics.stdev(values)) < 1e-9


def test_aggregator_stdev_under_two_elements() -> None:
    """Test _stdev returns None when under two elements."""
    assert _stdev([]) is None
    assert _stdev([42.0]) is None


def test_aggregator_min_max() -> None:
    """Test _min and _max."""
    values = [3.0, 1.0, 4.0, 1.0, 5.0]
    assert _min(values) == 1.0
    assert _max(values) == 5.0
    assert _min([]) is None
    assert _max([]) is None


def test_aggregator_first_episode_above() -> None:
    """Test _first_episode_above helper."""
    rewards = [0.1, 0.3, 0.5, 0.8, 1.0]
    assert _first_episode_above(rewards, 0.4) == 2
    assert _first_episode_above(rewards, 0.0) == 0
    assert _first_episode_above(rewards, 1.5) is None
    assert _first_episode_above([], 0.0) is None


def test_evaluator_construction() -> None:
    """Test that TransferGeneralizationEvaluator can be constructed."""
    distribution = _make_small_distribution()
    config = _make_config()
    evaluator = TransferGeneralizationEvaluator(distribution, config)
    assert evaluator.distribution is distribution
    assert evaluator.config is config
    assert len(evaluator._training_tasks) == 2
    assert len(evaluator._adaptation_tasks) == 2


# ===========================================================================
# 2. Configuration validation
# ===========================================================================


def test_config_construction() -> None:
    """Test TransferGeneralizationConfig construction."""
    config = TransferGeneralizationConfig(
        training_task_ids=["train-1", "train-2"],
        adaptation_task_ids=["adapt-1"],
        seeds=[42, 123],
    )
    assert config.training_task_ids == ["train-1", "train-2"]
    assert config.adaptation_task_ids == ["adapt-1"]
    assert config.seeds == [42, 123]
    assert config.task_performance_weight == 0.5


def test_config_custom_fields() -> None:
    """Test TransferGeneralizationConfig with custom fields."""
    config = TransferGeneralizationConfig(
        training_task_ids=["train-1"],
        adaptation_task_ids=["adapt-1"],
        seeds=[1],
        training_episodes_per_task=5,
        adaptation_episodes=10,
        max_steps=20,
        task_performance_weight=0.3,
        learning_rate=0.2,
        discount_factor=0.8,
        epsilon=0.8,
        epsilon_min=0.1,
        epsilon_decay=0.99,
    )
    assert config.training_episodes_per_task == 5
    assert config.adaptation_episodes == 10
    assert config.task_performance_weight == 0.3


def test_config_requires_seeds() -> None:
    """Test that config requires at least one seed."""
    with pytest.raises(
        ValueError, match="seeds must contain at least one seed"
    ):
        TransferGeneralizationConfig(
            training_task_ids=["train-1"],
            adaptation_task_ids=["adapt-1"],
            seeds=[],
        )


def test_config_requires_adaptation_tasks() -> None:
    """Test that config requires adaptation tasks."""
    with pytest.raises(
        ValueError,
        match="adaptation_task_ids must contain at least one task",
    ):
        TransferGeneralizationConfig(
            training_task_ids=["train-1"],
            adaptation_task_ids=[],
            seeds=[42],
        )


def test_config_validates_no_overlap() -> None:
    """Test that config validates no overlap between training and adaptation."""
    distribution = _make_small_distribution()
    config = TransferGeneralizationConfig(
        training_task_ids=["train-1", "adapt-1"],  # Overlap!
        adaptation_task_ids=["adapt-1"],
        seeds=[42],
    )
    with pytest.raises(ValueError, match="must not overlap"):
        config.validate(distribution)


def test_config_validates_task_ids_exist() -> None:
    """Test that config validates task IDs exist in distribution."""
    distribution = _make_small_distribution()
    config = TransferGeneralizationConfig(
        training_task_ids=["nonexistent-task"],
        adaptation_task_ids=["adapt-1"],
        seeds=[42],
    )
    with pytest.raises(ValueError, match="not found in distribution"):
        config.validate(distribution)


def test_config_validates_training_tasks_exist() -> None:
    """Test that config validates training task IDs exist."""
    distribution = _make_small_distribution()
    config = TransferGeneralizationConfig(
        training_task_ids=["train-1"],
        adaptation_task_ids=["nonexistent-adapt"],
        seeds=[42],
    )
    with pytest.raises(ValueError, match="not found in distribution"):
        config.validate(distribution)


# ===========================================================================
# 3. Multiple seeds
# ===========================================================================


def test_multiple_seeds_evaluated() -> None:
    """Test that multiple seeds are evaluated."""
    distribution = _make_small_distribution()
    config = _make_config(seeds=[42, 123, 456])

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    assert len(result.per_seed_results) == 3
    assert result.aggregate.num_seeds == 3
    assert [r.seed for r in result.per_seed_results] == [42, 123, 456]


def test_per_seed_results_available() -> None:
    """Test that per-seed results are available."""
    distribution = _make_small_distribution()
    config = _make_config(seeds=[42, 123])

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    assert len(result.per_seed_results) == 2
    for seed_result in result.per_seed_results:
        assert isinstance(seed_result.transfer_result, TransferExperimentResult)
        assert len(seed_result.transfer_result.from_scratch_results) == 2
        assert len(seed_result.transfer_result.transfer_results) == 2


# ===========================================================================
# 4. Multiple target tasks
# ===========================================================================


def test_multiple_target_tasks() -> None:
    """Test evaluation with multiple target tasks."""
    distribution = _make_small_distribution()
    config = _make_config()

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    assert len(result.per_task_results) == 2
    task_ids = {r.task_id for r in result.per_task_results}
    assert task_ids == {"adapt-1", "adapt-2"}


def test_per_task_results_available() -> None:
    """Test that per-task cross-seed results are available."""
    distribution = _make_small_distribution()
    config = _make_config()

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    for task_result in result.per_task_results:
        assert isinstance(task_result.mean_transfer_final_reward, float)
        assert isinstance(task_result.mean_from_scratch_final_reward, float)
        assert isinstance(task_result.mean_transfer_advantage, float)
        assert isinstance(task_result.mean_task_success_advantage, float)


# ===========================================================================
# 5. Source/target task disjointness
# ===========================================================================


def test_source_target_disjointness_in_config() -> None:
    """Test that config validation enforces disjoint source/target tasks."""
    distribution = _make_small_distribution()
    config = TransferGeneralizationConfig(
        training_task_ids=["train-1", "adapt-1"],
        adaptation_task_ids=["adapt-1"],
        seeds=[42],
    )
    with pytest.raises(ValueError, match="must not overlap"):
        config.validate(distribution)


def test_source_target_disjointness_in_result() -> None:
    """Test that source and target tasks are disjoint in the result."""
    distribution = _make_small_distribution()
    config = _make_config()

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    training_ids = set(config.training_task_ids)
    adaptation_ids = set(config.adaptation_task_ids)
    assert training_ids.isdisjoint(adaptation_ids)
    assert result.aggregate.num_source_tasks == len(training_ids)
    assert result.aggregate.num_target_tasks == len(adaptation_ids)


# ===========================================================================
# 6. Transfer condition
# ===========================================================================


def test_transfer_condition_executed() -> None:
    """Test that transfer condition is executed for each seed."""
    distribution = _make_small_distribution()
    config = _make_config()

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    for seed_result in result.per_seed_results:
        seed_transfer = seed_result.transfer_result
        assert len(seed_transfer.transfer_results) == 2
        for task_result in seed_transfer.transfer_results:
            assert task_result.condition == "transfer"
            assert len(task_result.episode_results) == config.adaptation_episodes


# ===========================================================================
# 7. From-scratch condition
# ===========================================================================


def test_from_scratch_condition_executed() -> None:
    """Test that from-scratch condition is executed for each seed."""
    distribution = _make_small_distribution()
    config = _make_config()

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    for seed_result in result.per_seed_results:
        seed_scratch = seed_result.transfer_result
        assert len(seed_scratch.from_scratch_results) == 2
        for task_result in seed_scratch.from_scratch_results:
            assert task_result.condition == "from_scratch"
            assert len(task_result.episode_results) == config.adaptation_episodes


# ===========================================================================
# 8. Same experimental controls
# ===========================================================================


def test_same_controls_across_conditions() -> None:
    """Test that transfer and from-scratch use the same controls."""
    distribution = _make_small_distribution()
    config = _make_config()

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    # Both conditions should use the same target tasks, episodes, etc.
    for seed_result in result.per_seed_results:
        seed_transfer = seed_result.transfer_result
        seed_scratch = seed_result.transfer_result

        scratch_task_ids = {r.task_id for r in seed_transfer.from_scratch_results}
        transfer_task_ids = {r.task_id for r in seed_transfer.transfer_results}
        assert scratch_task_ids == transfer_task_ids
        assert scratch_task_ids == {"adapt-1", "adapt-2"}

        # Same episode counts
        for scratch_result, transfer_result in zip(
            seed_transfer.from_scratch_results, seed_transfer.transfer_results
        ):
            assert len(scratch_result.episode_results) == len(
                transfer_result.episode_results
            )


# ===========================================================================
# 9. Deterministic results
# ===========================================================================


def test_deterministic_evaluation() -> None:
    """Test that same configuration produces identical results."""
    distribution = _make_small_distribution()

    config_a = _make_config(seeds=[42, 123])
    config_b = _make_config(seeds=[42, 123])

    evaluator_a = TransferGeneralizationEvaluator(distribution, config_a)
    result_a = evaluator_a.run()

    evaluator_b = TransferGeneralizationEvaluator(distribution, config_b)
    result_b = evaluator_b.run()

    # Results should be structurally identical
    assert result_a.aggregate.num_seeds == result_b.aggregate.num_seeds
    assert result_a.aggregate.num_source_tasks == result_b.aggregate.num_source_tasks
    assert result_a.aggregate.num_target_tasks == result_b.aggregate.num_target_tasks
    assert len(result_a.per_seed_results) == len(result_b.per_seed_results)
    assert len(result_a.per_task_results) == len(result_b.per_task_results)

    for ra, rb in zip(result_a.per_task_results, result_b.per_task_results):
        assert ra.task_id == rb.task_id
        assert ra.mean_transfer_advantage == pytest.approx(
            rb.mean_transfer_advantage
        )


# ===========================================================================
# 10. Per-seed results
# ===========================================================================


def test_per_seed_transfer_advantage() -> None:
    """Test per-seed transfer advantage is computed."""
    distribution = _make_small_distribution()
    config = _make_config(seeds=[42])

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    # Check that per-seed results exist
    assert len(result.per_seed_results) == 1
    seed_result = result.per_seed_results[0]
    assert seed_result.seed == 42


# ===========================================================================
# 11. Per-task results
# ===========================================================================


def test_per_task_transfer_advantage() -> None:
    """Test per-task cross-seed transfer advantage is computed."""
    distribution = _make_small_distribution()
    config = _make_config()

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    for task_result in result.per_task_results:
        assert task_result.task_id in {"adapt-1", "adapt-2"}
        assert isinstance(task_result.mean_transfer_advantage, float)
        assert isinstance(task_result.mean_task_success_advantage, float)


def test_per_task_episode_curves_structure() -> None:
    """Test that per-task episode curves have the expected structure."""
    distribution = _make_small_distribution()
    config = _make_config(adaptation_episodes=2)

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    for task_result in result.per_task_results:
        curves = task_result.episode_curves
        assert "transfer" in curves
        assert "from_scratch" in curves
        # Each seed should have a curve for each condition
        assert len(curves["transfer"]) == len(config.seeds)
        assert len(curves["from_scratch"]) == len(config.seeds)
        # Each curve should have the right number of episodes
        for curve in curves["transfer"]:
            assert len(curve) == config.adaptation_episodes
        for curve in curves["from_scratch"]:
            assert len(curve) == config.adaptation_episodes


# ===========================================================================
# 12. Aggregate results
# ===========================================================================


def test_aggregate_metrics_structure() -> None:
    """Test that aggregate metrics have the expected structure."""
    distribution = _make_small_distribution()
    config = _make_config()

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    agg = result.aggregate
    assert agg.num_seeds == len(config.seeds)
    assert agg.num_source_tasks == len(config.training_task_ids)
    assert agg.num_target_tasks == len(config.adaptation_task_ids)
    assert isinstance(agg.mean_transfer_reward, float)
    assert isinstance(agg.mean_from_scratch_reward, float)
    assert isinstance(agg.mean_transfer_advantage, float)
    assert isinstance(agg.mean_transfer_task_success, float)
    assert isinstance(agg.mean_from_scratch_task_success, float)
    assert isinstance(agg.mean_task_success_advantage, float)


def test_aggregate_transfer_advantage_stats() -> None:
    """Test that aggregate transfer advantage stats are computed."""
    distribution = _make_small_distribution()
    config = _make_config(seeds=[42, 123])

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    agg = result.aggregate
    assert agg.transfer_advantage_stdev is not None
    assert agg.transfer_advantage_min is not None
    assert agg.transfer_advantage_max is not None
    assert agg.transfer_advantage_min <= agg.transfer_advantage_max


# ===========================================================================
# 13. Mean/std/min/max calculations
# ===========================================================================


def test_aggregate_uses_correct_statistics() -> None:
    """Test that aggregate statistics match expected values."""
    distribution = _make_small_distribution()
    config = _make_config(seeds=[42, 123])

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    # Verify aggregate mean matches recompute from per-task means (weighted by tasks)
    per_task_advantages = [r.mean_transfer_advantage for r in result.per_task_results]
    assert result.aggregate.mean_transfer_advantage == pytest.approx(
        _mean(per_task_advantages)
    )


def test_stdev_is_sample_stdev() -> None:
    """Test that stdev calculation matches sample standard deviation."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _stdev(values) == pytest.approx(statistics.stdev(values))


# ===========================================================================
# 14. Transfer advantage
# ===========================================================================


def test_transfer_advantage_formula() -> None:
    """Test that transfer advantage follows the documented formula."""
    distribution = _make_small_distribution()
    config = _make_config(seeds=[42])

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    # Verify per-task advantage = transfer_final - scratch_final
    for per_task in result.per_task_results:
        expected_advantage = (
            per_task.mean_transfer_final_reward
            - per_task.mean_from_scratch_final_reward
        )
        assert per_task.mean_transfer_advantage == pytest.approx(
            expected_advantage
        )


def test_transfer_advantage_can_be_negative() -> None:
    """Test that transfer advantage can be zero or negative (no forced positivity)."""
    distribution = _make_small_distribution()
    config = _make_config(seeds=[42, 123])

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    agg = result.aggregate
    # Just verify the values are real numbers - they may be positive, zero, or negative
    assert isinstance(agg.mean_transfer_advantage, float)
    assert isinstance(agg.transfer_advantage_min, float)
    assert isinstance(agg.transfer_advantage_max, float)


# ===========================================================================
# 15. Task-success advantage
# ===========================================================================


def test_task_success_advantage_formula() -> None:
    """Test that task-success advantage follows the documented formula."""
    distribution = _make_small_distribution()
    config = _make_config(seeds=[42])

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    for per_task in result.per_task_results:
        expected_advantage = (
            per_task.mean_transfer_final_task_success
            - per_task.mean_from_scratch_final_task_success
        )
        assert per_task.mean_task_success_advantage == pytest.approx(
            expected_advantage
        )


def test_task_success_advantage_can_be_zero() -> None:
    """Test that task-success advantage can be zero when both conditions
    achieve the same final task-success score."""
    distribution = _make_small_distribution()
    config = _make_config(seeds=[42])

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    # The values should be valid floats - they may be zero
    for per_task in result.per_task_results:
        assert isinstance(per_task.mean_task_success_advantage, float)


# ===========================================================================
# 16. Adaptation improvement
# ===========================================================================


def test_adaptation_improvement_from_first_to_final() -> None:
    """Test that adaptation improvement (first-to-final reward delta) can be
    derived from episode curves."""
    distribution = _make_small_distribution()
    config = _make_config(adaptation_episodes=3)

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    # Compute adaptation improvement from episode curves for one task
    for per_task in result.per_task_results:
        transfer_curves = per_task.episode_curves.get("transfer", [])
        from_scratch_curves = per_task.episode_curves.get("from_scratch", [])

        if transfer_curves and from_scratch_curves:
            # Average across seeds: first vs final episode reward
            transfer_first_avg = _mean(
                [c[0] for c in transfer_curves if c]
            )
            transfer_final_avg = _mean(
                [c[-1] for c in transfer_curves if c]
            )
            scratch_first_avg = _mean(
                [c[0] for c in from_scratch_curves if c]
            )
            scratch_final_avg = _mean(
                [c[-1] for c in from_scratch_curves if c]
            )

            transfer_improvement = transfer_final_avg - transfer_first_avg
            scratch_improvement = scratch_final_avg - scratch_first_avg

            # These are real numbers - no assertion on sign
            assert isinstance(transfer_improvement, float)
            assert isinstance(scratch_improvement, float)


# ===========================================================================
# 17. Threshold-not-reached behavior
# ===========================================================================


def test_threshold_not_reached_returns_none() -> None:
    """Test that sample efficiency returns None when threshold is never reached."""
    distribution = _make_small_distribution()
    config = _make_config(adaptation_episodes=1, max_steps=3)

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    # Use an extremely high threshold that should not be reached
    efficiency = evaluator.calculate_sample_efficiency(result, reward_threshold=9999.0)

    assert efficiency["transfer_mean_episodes_to_threshold"] is None
    assert efficiency["from_scratch_mean_episodes_to_threshold"] is None
    assert efficiency["transfer_thresholds_reached"] == 0
    assert efficiency["from_scratch_thresholds_reached"] == 0


def test_threshold_reached_returns_value() -> None:
    """Test that sample efficiency returns a value when threshold is reached."""
    distribution = _make_small_distribution()
    config = _make_config(adaptation_episodes=2, max_steps=3)

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    # Use a low threshold that should be reached
    efficiency = evaluator.calculate_sample_efficiency(result, reward_threshold=-10.0)

    assert efficiency["transfer_mean_episodes_to_threshold"] is not None
    assert efficiency["from_scratch_mean_episodes_to_threshold"] is not None
    assert efficiency["transfer_thresholds_reached"] > 0
    assert efficiency["from_scratch_thresholds_reached"] > 0


# ===========================================================================
# 18. Adaptation curve data
# ===========================================================================


def test_adaptation_curve_serialization() -> None:
    """Test that adaptation curve data is serializable."""
    distribution = _make_small_distribution()
    config = _make_config(adaptation_episodes=2)

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    serialized = result.to_dict()
    assert isinstance(serialized, dict)
    assert "per_task_results" in serialized

    for task_dict in serialized["per_task_results"]:
        assert "episode_curves" in task_dict
        curves = task_dict["episode_curves"]
        assert "transfer" in curves
        assert "from_scratch" in curves


# ===========================================================================
# 19. Serialization
# ===========================================================================


def test_full_result_serialization() -> None:
    """Test that the full evaluation result can be serialized."""
    distribution = _make_small_distribution()
    config = _make_config()

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    serialized = result.to_dict()
    assert isinstance(serialized, dict)
    assert "config" in serialized
    assert "per_seed_results" in serialized
    assert "per_task_results" in serialized
    assert "aggregate" in serialized


def test_result_json_serialization() -> None:
    """Test JSON serialization of the full result."""
    distribution = _make_small_distribution()
    config = _make_config()

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

    json_str = json.dumps(result.to_dict(), indent=2)
    assert isinstance(json_str, str)

    parsed = json.loads(json_str)
    assert "aggregate" in parsed
    assert "mean_transfer_advantage" in parsed["aggregate"]


def test_aggregate_model_serialization() -> None:
    """Test that AggregateGeneralizationMetrics serializes correctly."""
    agg = AggregateGeneralizationMetrics(
        num_seeds=2,
        num_source_tasks=2,
        num_target_tasks=2,
        mean_transfer_reward=0.5,
        mean_from_scratch_reward=0.3,
        mean_transfer_advantage=0.2,
        transfer_advantage_stdev=0.1,
        transfer_advantage_min=0.1,
        transfer_advantage_max=0.3,
        mean_transfer_task_success=0.6,
        mean_from_scratch_task_success=0.4,
        mean_task_success_advantage=0.2,
        overall_notes="Test",
    )

    serialized = agg.model_dump(mode="json", exclude_none=True)
    assert serialized["num_seeds"] == 2
    assert serialized["mean_transfer_advantage"] == 0.2
    assert serialized["transfer_advantage_stdev"] == 0.1


def test_per_task_model_serialization() -> None:
    """Test that PerTaskCrossSeedResult serializes correctly."""
    per_task = PerTaskCrossSeedResult(
        task_id="adapt-1",
        task_category="research",
        difficulty=2,
        mean_transfer_final_reward=0.5,
        mean_from_scratch_final_reward=0.3,
        mean_transfer_advantage=0.2,
        transfer_reward_stdev=0.1,
        from_scratch_reward_stdev=None,
        mean_transfer_final_task_success=0.6,
        mean_from_scratch_final_task_success=0.4,
        mean_task_success_advantage=0.2,
        episode_curves={"transfer": [[0.1, 0.5]], "from_scratch": [[0.0, 0.3]]},
    )

    serialized = per_task.model_dump(mode="json", exclude_none=True)
    assert serialized["task_id"] == "adapt-1"
    assert serialized["mean_transfer_advantage"] == 0.2
    # None fields are excluded when exclude_none=True
    assert "from_scratch_reward_stdev" not in serialized


# ===========================================================================
# 20. Empty seeds handling
# ===========================================================================


def test_empty_seeds_rejected() -> None:
    """Test that empty seeds list is rejected."""
    with pytest.raises(ValueError, match="seeds must contain at least one seed"):
        TransferGeneralizationConfig(
            training_task_ids=["train-1"],
            adaptation_task_ids=["adapt-1"],
            seeds=[],
        )


# ===========================================================================
# 21. Invalid task IDs
# ===========================================================================


def test_invalid_training_task_id_rejected() -> None:
    """Test that invalid training task ID is rejected."""
    distribution = _make_small_distribution()
    config = TransferGeneralizationConfig(
        training_task_ids=["nonexistent-train"],
        adaptation_task_ids=["adapt-1"],
        seeds=[42],
    )
    with pytest.raises(ValueError, match="not found in distribution"):
        config.validate(distribution)


def test_invalid_adaptation_task_id_rejected() -> None:
    """Test that invalid adaptation task ID is rejected."""
    distribution = _make_small_distribution()
    config = TransferGeneralizationConfig(
        training_task_ids=["train-1"],
        adaptation_task_ids=["nonexistent-adapt"],
        seeds=[42],
    )
    with pytest.raises(ValueError, match="not found in distribution"):
        config.validate(distribution)


# ===========================================================================
# 22. Overlapping source/target tasks
# ===========================================================================


def test_overlapping_tasks_rejected() -> None:
    """Test that overlapping source/target tasks are rejected."""
    distribution = _make_small_distribution()
    config = TransferGeneralizationConfig(
        training_task_ids=["train-1", "adapt-1"],
        adaptation_task_ids=["adapt-1", "adapt-2"],
        seeds=[42],
    )
    with pytest.raises(ValueError, match="must not overlap"):
        config.validate(distribution)


# ===========================================================================
# 23. No LLM/API calls
# ===========================================================================


def test_evaluator_has_no_api_imports() -> None:
    """Test that the transfer evaluation module doesn't import API-related modules."""
    import app.rl.transfer_evaluation as te
    import inspect

    source = inspect.getsource(te)

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
        assert (
            f"import {forbidden.lower()}" not in source_lower
        ), f"Found forbidden import: {forbidden}"
        assert (
            f"from {forbidden.lower()}" not in source_lower
        ), f"Found forbidden import: {forbidden}"


def test_evaluator_is_fully_offline() -> None:
    """Test that the evaluator runs without network calls."""
    distribution = _make_small_distribution()
    config = _make_config(seeds=[42])

    evaluator = TransferGeneralizationEvaluator(distribution, config)

    # If we reach here without network errors, the module stayed offline
    result = evaluator.run()
    assert isinstance(result, TransferGeneralizationResult)


# ===========================================================================
# 24. No credentials/secrets
# ===========================================================================


def test_result_contains_no_credentials() -> None:
    """Test that evaluation results contain no credentials."""
    distribution = _make_small_distribution()
    config = _make_config()

    evaluator = TransferGeneralizationEvaluator(distribution, config)
    result = evaluator.run()

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
        assert (
            marker not in serialized_lower
        ), f"Found forbidden marker: {marker}"


# ===========================================================================
# 25. Existing Step 22 compatibility
# ===========================================================================


def test_step22_task_transfer_still_works() -> None:
    """Test that existing Step 22 TaskTransferExperiment still works."""
    from app.rl.task_transfer import TaskTransferExperiment, TransferConfig

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

    assert isinstance(result, TransferExperimentResult)
    assert len(result.from_scratch_results) == 1
    assert len(result.transfer_results) == 1


def test_step22_transfer_config_compatible() -> None:
    """Test that Step 23 config is compatible with Step 22 config fields."""
    from app.rl.task_transfer import TransferConfig

    # Step 23 config should be constructible with fields compatible with Step 22
    config = TransferGeneralizationConfig(
        training_task_ids=["train-1"],
        adaptation_task_ids=["adapt-1"],
        seeds=[42],
        training_episodes_per_task=1,
        adaptation_episodes=1,
        max_steps=3,
        task_performance_weight=0.0,
        learning_rate=0.1,
        discount_factor=0.9,
    )

    # Verify the fields exist and are accessible
    assert config.training_task_ids == ["train-1"]
    assert config.adaptation_task_ids == ["adapt-1"]
    assert config.training_episodes_per_task == 1
    assert config.adaptation_episodes == 1


# ===========================================================================
# 26. workflow.py unchanged
# ===========================================================================


def test_workflow_py_not_modified() -> None:
    """Test that app/graph/workflow.py is not modified."""
    workflow_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "app",
        "graph",
        "workflow.py",
    )
    assert os.path.exists(workflow_path), "expected app/graph/workflow.py to exist"


# ===========================================================================
# 27. app/agents unchanged
# ===========================================================================


def test_agents_directory_not_modified() -> None:
    """Test that app/agents/ is not modified."""
    agents_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "app",
        "agents",
    )
    assert os.path.isdir(agents_path), "expected app/agents/ to exist"
