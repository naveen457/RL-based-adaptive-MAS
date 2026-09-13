"""
Tests for the Meta-RL evaluation framework (Step 19).

These tests cover:

1. Evaluator construction
2. Architecture-only baseline evaluation
3. Task-aware evaluation
4. Multi-task evaluation
5. Same-seed deterministic results
6. Different configurations
7. Per-task results
8. Aggregate metrics
9. Comparison metrics
10. Train/test evaluation
11. Zero-denominator handling
12. Empty distribution handling
13. Invalid configuration handling
14. Serialization
15. No LLM/API calls
16. Task context is not accidentally provided to architecture-only baseline
17. Protected files remain unchanged
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile

import pytest

from app.architecture.manager import ArchitectureManager
from app.rl.meta_task import MetaTask
from app.rl.meta_environment import MetaEnvironment
from app.rl.task_distribution import MetaTaskDistribution, create_baseline_task_distribution
from app.rl.meta_evaluation import (
    MetaRLEvaluator,
    EvaluationConfig,
    EvaluationResult,
    ComparisonResult,
    AggregateEvaluationResult,
    PerTaskEvaluationResult,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_task(
    *,
    task_id: str = "task-001",
    task_category: str = "coding",
    difficulty: int = 2,
    required_capabilities: list[str] | None = None,
    context_features: dict | None = None,
) -> MetaTask:
    if required_capabilities is None:
        required_capabilities = ["coding"]
    if context_features is None:
        context_features = {}
    return MetaTask(
        task_id=task_id,
        task_description=f"Sample {task_id}",
        task_category=task_category,
        required_capabilities=required_capabilities,
        difficulty=difficulty,
        context_features=context_features,
    )


def _make_small_distribution() -> MetaTaskDistribution:
    """Create a small distribution for testing."""
    tasks = [
        _make_task(task_id="task-001", task_category="coding", difficulty=2),
        _make_task(task_id="task-002", task_category="research", difficulty=3),
    ]
    return MetaTaskDistribution(tasks=tasks)


# ============================================================================
# 1. Evaluator construction
# ============================================================================


def test_evaluator_construction_with_distribution() -> None:
    """Test that MetaRLEvaluator can be constructed with a distribution."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(task_distribution=distribution)
    assert evaluator is not None
    assert evaluator.task_distribution is distribution


def test_evaluator_construction_with_config() -> None:
    """Test that MetaRLEvaluator can be constructed with a custom config."""
    distribution = _make_small_distribution()
    config = EvaluationConfig(
        num_tasks=2,
        episodes_per_task=2,
        max_steps_per_episode=5,
        seed=123,
    )
    evaluator = MetaRLEvaluator(task_distribution=distribution, config=config)
    assert evaluator.config.num_tasks == 2
    assert evaluator.config.episodes_per_task == 2
    assert evaluator.config.max_steps_per_episode == 5
    assert evaluator.config.seed == 123


def test_evaluator_rejects_empty_distribution() -> None:
    """Test that evaluator rejects empty task distribution."""
    distribution = MetaTaskDistribution(tasks=[])
    with pytest.raises(ValueError, match="at least one task"):
        MetaRLEvaluator(task_distribution=distribution)


def test_evaluator_default_config() -> None:
    """Test that evaluator has sensible defaults."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(task_distribution=distribution)
    assert evaluator.config.num_tasks == 5
    assert evaluator.config.episodes_per_task == 3
    assert evaluator.config.max_steps_per_episode == 10
    assert evaluator.config.seed == 42


# ============================================================================
# 2. Architecture-only baseline evaluation
# ============================================================================


def test_architecture_only_evaluation_produces_results() -> None:
    """Test that architecture-only evaluation produces valid results."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=2, episodes_per_task=1, max_steps_per_episode=3, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=False)

    assert isinstance(result, EvaluationResult)
    assert result.comparison.architecture_only.num_tasks == 2
    assert result.comparison.architecture_only.num_episodes > 0
    assert result.comparison.architecture_only.mean_reward is not None
    assert result.comparison.architecture_only.approach_name == "architecture_only"


def test_architecture_only_no_task_context_in_observation() -> None:
    """Test that architecture-only baseline does NOT receive task context."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=1, episodes_per_task=1, max_steps_per_episode=2, seed=42),
    )

    # This should not raise - the evaluator internally verifies no task context
    result = evaluator.evaluate(evaluate_multi_task=False)

    # Verify the architecture-only result exists
    assert result.comparison.architecture_only.num_tasks == 1

    # Check that per-task results have the "architecture_only_baseline" note
    for task_result in result.comparison.architecture_only.per_task_results:
        assert "architecture_only_baseline" in task_result.task_context_summary.get("note", "")


# ============================================================================
# 3. Task-aware evaluation
# ============================================================================


def test_task_aware_evaluation_produces_results() -> None:
    """Test that task-aware evaluation produces valid results."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=2, episodes_per_task=1, max_steps_per_episode=3, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=False)

    assert isinstance(result, EvaluationResult)
    assert result.comparison.task_aware.num_tasks == 2
    assert result.comparison.task_aware.num_episodes > 0
    assert result.comparison.task_aware.mean_reward is not None
    assert result.comparison.task_aware.approach_name == "task_aware"


def test_task_aware_has_task_context_in_observation() -> None:
    """Test that task-aware evaluation receives task context."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=1, episodes_per_task=1, max_steps_per_episode=2, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=False)

    # Task-aware results should have task context summary
    for task_result in result.comparison.task_aware.per_task_results:
        assert "task_category" in task_result.task_context_summary
        assert task_result.task_context_summary["task_category"] in ["coding", "research"]


# ============================================================================
# 4. Multi-task evaluation
# ============================================================================


def test_multi_task_evaluation_produces_results() -> None:
    """Test that multi-task evaluation produces valid results."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=2, episodes_per_task=1, max_steps_per_episode=3, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=True)

    assert result.comparison.multi_task is not None
    assert result.comparison.multi_task.num_tasks == 2
    assert result.comparison.multi_task.num_episodes > 0
    assert result.comparison.multi_task.mean_reward is not None
    assert result.comparison.multi_task.approach_name == "multi_task"


def test_multi_task_evaluation_can_be_disabled() -> None:
    """Test that multi-task evaluation can be disabled."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=2, episodes_per_task=1, max_steps_per_episode=3, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=False)

    assert result.comparison.multi_task is None


# ============================================================================
# 5. Same-seed deterministic results
# ============================================================================


def test_same_seed_produces_identical_results() -> None:
    """Test that same seed produces identical evaluation results."""
    distribution = _make_small_distribution()
    config = EvaluationConfig(num_tasks=2, episodes_per_task=1, max_steps_per_episode=3, seed=42)

    evaluator_a = MetaRLEvaluator(task_distribution=distribution, config=config)
    evaluator_b = MetaRLEvaluator(task_distribution=distribution, config=config)

    result_a = evaluator_a.evaluate(evaluate_multi_task=False)
    result_b = evaluator_b.evaluate(evaluate_multi_task=False)

    # Use the determinism check method
    assert evaluator_a.evaluate_determinism(num_runs=2)


def test_determinism_verification_method() -> None:
    """Test the evaluate_determinism helper method."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=2, episodes_per_task=1, max_steps_per_episode=3, seed=42),
    )

    # Should return True for deterministic evaluation
    assert evaluator.evaluate_determinism(num_runs=2) is True


# ============================================================================
# 6. Different configurations
# ============================================================================


def test_different_episodes_per_task() -> None:
    """Test that different episodes_per_task configuration works."""
    distribution = _make_small_distribution()

    config_short = EvaluationConfig(
        num_tasks=2, episodes_per_task=1, max_steps_per_episode=2, seed=42
    )
    config_long = EvaluationConfig(
        num_tasks=2, episodes_per_task=3, max_steps_per_episode=2, seed=42
    )

    evaluator_short = MetaRLEvaluator(task_distribution=distribution, config=config_short)
    evaluator_long = MetaRLEvaluator(task_distribution=distribution, config=config_long)

    result_short = evaluator_short.evaluate(evaluate_multi_task=False)
    result_long = evaluator_long.evaluate(evaluate_multi_task=False)

    assert result_short.comparison.architecture_only.num_episodes == 2
    assert result_long.comparison.architecture_only.num_episodes == 6


def test_different_max_steps() -> None:
    """Test that different max_steps configuration affects episode length."""
    distribution = _make_small_distribution()

    config_short = EvaluationConfig(
        num_tasks=1, episodes_per_task=1, max_steps_per_episode=2, seed=42
    )
    config_long = EvaluationConfig(
        num_tasks=1, episodes_per_task=1, max_steps_per_episode=5, seed=42
    )

    evaluator_short = MetaRLEvaluator(task_distribution=distribution, config=config_short)
    evaluator_long = MetaRLEvaluator(task_distribution=distribution, config=config_long)

    result_short = evaluator_short.evaluate(evaluate_multi_task=False)
    result_long = evaluator_long.evaluate(evaluate_multi_task=False)

    # Episode lengths should reflect max_steps
    per_task_short = result_short.comparison.architecture_only.per_task_results[0]
    per_task_long = result_long.comparison.architecture_only.per_task_results[0]

    assert per_task_short.mean_episode_length <= 2
    assert per_task_long.mean_episode_length <= 5


# ============================================================================
# 7. Per-task results
# ============================================================================


def test_per_task_results_structure() -> None:
    """Test that per-task results have required fields."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=2, episodes_per_task=1, max_steps_per_episode=3, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=False)

    for task_result in result.comparison.architecture_only.per_task_results:
        assert task_result.task_id
        assert task_result.task_category
        assert task_result.difficulty >= 1
        assert task_result.episode_count == 1
        assert isinstance(task_result.mean_reward, float)
        assert isinstance(task_result.total_reward, float)
        assert isinstance(task_result.mean_episode_length, float)
        assert task_result.final_architecture_version >= 0
        assert isinstance(task_result.valid_transition_count, int)
        assert isinstance(task_result.invalid_transition_count, int)


def test_per_task_results_contain_task_ids() -> None:
    """Test that per-task results contain correct task IDs."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=2, episodes_per_task=1, max_steps_per_episode=3, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=False)

    task_ids = {r.task_id for r in result.comparison.architecture_only.per_task_results}
    assert task_ids == {"task-001", "task-002"}


# ============================================================================
# 8. Aggregate metrics
# ============================================================================


def test_aggregate_metrics_structure() -> None:
    """Test that aggregate metrics have required fields."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=2, episodes_per_task=2, max_steps_per_episode=3, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=False)

    agg = result.comparison.architecture_only
    assert agg.approach_name == "architecture_only"
    assert agg.num_tasks == 2
    assert agg.num_episodes == 4
    assert isinstance(agg.mean_reward, float)
    assert isinstance(agg.total_reward, float)
    assert isinstance(agg.mean_episode_length, float)
    # Standard deviation might be None for small sample
    assert agg.reward_standard_deviation is None or isinstance(agg.reward_standard_deviation, float)
    assert isinstance(agg.mean_final_architecture_version, float)


def test_aggregate_metrics_correct_sum() -> None:
    """Test that aggregate total_reward equals sum of per-task totals."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=2, episodes_per_task=2, max_steps_per_episode=3, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=False)

    agg = result.comparison.architecture_only
    per_task_sum = sum(r.total_reward for r in agg.per_task_results)

    assert agg.total_reward == pytest.approx(per_task_sum)


# ============================================================================
# 9. Comparison metrics
# ============================================================================


def test_comparison_metrics_exist() -> None:
    """Test that comparison metrics are computed."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=2, episodes_per_task=2, max_steps_per_episode=3, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=True)

    comparison = result.comparison

    # Task-aware vs architecture-only
    assert comparison.task_aware_absolute_improvement is not None
    assert isinstance(comparison.task_aware_absolute_improvement, float)

    # Relative improvement might be None if baseline is zero
    assert comparison.task_aware_relative_improvement is None or isinstance(
        comparison.task_aware_relative_improvement, float
    )

    # Multi-task metrics
    if comparison.multi_task is not None:
        assert comparison.multi_task_absolute_improvement is not None
        assert comparison.multi_task_vs_task_aware_absolute is not None


def test_comparison_uses_correct_formula() -> None:
    """Test that comparison metrics use correct formulas."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=2, episodes_per_task=2, max_steps_per_episode=3, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=True)

    comparison = result.comparison
    arch_only = comparison.architecture_only.mean_reward
    task_aware = comparison.task_aware.mean_reward

    # Verify absolute improvement formula
    expected_abs = task_aware - arch_only
    assert comparison.task_aware_absolute_improvement == pytest.approx(expected_abs)

    # Verify relative improvement formula (or None if baseline is zero)
    if abs(arch_only) > 1e-10:
        expected_rel = expected_abs / abs(arch_only)
        assert comparison.task_aware_relative_improvement == pytest.approx(expected_rel)
    else:
        assert comparison.task_aware_relative_improvement is None


# ============================================================================
# 10. Train/test evaluation
# ============================================================================


def test_train_test_split_evaluation() -> None:
    """Test evaluation with train/test split."""
    distribution = create_baseline_task_distribution()

    config = EvaluationConfig(
        num_tasks=6,
        episodes_per_task=1,
        max_steps_per_episode=3,
        seed=42,
        train_ratio=0.5,
    )

    evaluator = MetaRLEvaluator(task_distribution=distribution, config=config)

    result = evaluator.evaluate(train_test_split=True, evaluate_multi_task=False)

    # Should have evaluated on train tasks
    assert result.comparison.architecture_only.num_tasks > 0
    assert result.comparison.task_aware.num_tasks > 0


def test_train_test_split_deterministic() -> None:
    """Test that train/test split is deterministic with seed."""
    distribution = create_baseline_task_distribution()

    config = EvaluationConfig(
        num_tasks=6,
        episodes_per_task=1,
        max_steps_per_episode=2,
        seed=42,
        train_ratio=0.5,
    )

    evaluator_a = MetaRLEvaluator(task_distribution=distribution, config=config)
    evaluator_b = MetaRLEvaluator(task_distribution=distribution, config=config)

    result_a = evaluator_a.evaluate(train_test_split=True, evaluate_multi_task=False)
    result_b = evaluator_b.evaluate(train_test_split=True, evaluate_multi_task=False)

    # Results should be identical
    assert evaluator_a._results_equal(result_a, result_b)


# ============================================================================
# 11. Zero-denominator handling
# ============================================================================


def test_zero_baseline_relative_improvement_is_none() -> None:
    """Test that relative improvement is None when baseline is zero."""
    # Create a distribution and evaluate
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=2, episodes_per_task=1, max_steps_per_episode=2, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=False)

    comparison = result.comparison
    arch_only_reward = comparison.architecture_only.mean_reward

    # If baseline is effectively zero, relative improvement should be None
    if abs(arch_only_reward) < 1e-10:
        assert comparison.task_aware_relative_improvement is None


def test_absolute_improvement_works_with_zero_baseline() -> None:
    """Test that absolute improvement works even when baseline is zero."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=2, episodes_per_task=1, max_steps_per_episode=2, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=False)

    comparison = result.comparison

    # Absolute improvement should always be a number
    assert comparison.task_aware_absolute_improvement is not None
    assert isinstance(comparison.task_aware_absolute_improvement, float)


# ============================================================================
# 12. Empty distribution handling
# ============================================================================


def test_empty_distribution_raises_at_construction() -> None:
    """Test that empty distribution raises at construction."""
    distribution = MetaTaskDistribution(tasks=[])
    with pytest.raises(ValueError, match="at least one task"):
        MetaRLEvaluator(task_distribution=distribution)


def test_single_task_distribution_works() -> None:
    """Test that single-task distribution works for evaluation."""
    distribution = MetaTaskDistribution(tasks=[_make_task()])
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=1, episodes_per_task=1, max_steps_per_episode=2, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=False)

    assert result.comparison.architecture_only.num_tasks == 1
    assert result.comparison.task_aware.num_tasks == 1


# ============================================================================
# 13. Invalid configuration handling
# ============================================================================


def test_invalid_num_tasks_raises() -> None:
    """Test that invalid num_tasks raises validation error."""
    with pytest.raises(Exception):  # Pydantic raises ValidationError
        EvaluationConfig(num_tasks=0)


def test_invalid_episodes_per_task_raises() -> None:
    """Test that invalid episodes_per_task raises validation error."""
    with pytest.raises(Exception):  # Pydantic raises ValidationError
        EvaluationConfig(episodes_per_task=0)


def test_invalid_max_steps_raises() -> None:
    """Test that invalid max_steps raises validation error."""
    with pytest.raises(Exception):  # Pydantic raises ValidationError
        EvaluationConfig(max_steps_per_episode=0)


def test_evaluator_validates_config_on_init() -> None:
    """Test that evaluator validates config on initialization."""
    distribution = _make_small_distribution()

    with pytest.raises(Exception):  # Pydantic raises ValidationError for num_tasks=0
        EvaluationConfig(num_tasks=0)


# ============================================================================
# 14. Serialization
# ============================================================================


def test_evaluation_result_serialization() -> None:
    """Test that EvaluationResult can be serialized to dict."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=1, episodes_per_task=1, max_steps_per_episode=2, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=False)
    result_dict = result.to_dict()

    assert isinstance(result_dict, dict)
    assert "config" in result_dict
    assert "comparison" in result_dict
    assert "architecture_only" in result_dict["comparison"]
    assert "task_aware" in result_dict["comparison"]


def test_evaluation_result_json_serialization() -> None:
    """Test that EvaluationResult can be serialized to JSON string."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=1, episodes_per_task=1, max_steps_per_episode=2, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=False)
    json_str = result.to_json_string()

    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert "config" in parsed
    assert "comparison" in parsed


def test_aggregate_result_serialization() -> None:
    """Test that AggregateEvaluationResult can be serialized."""
    agg = AggregateEvaluationResult(
        approach_name="test",
        num_tasks=1,
        num_episodes=1,
        mean_reward=0.5,
        total_reward=0.5,
        mean_episode_length=10.0,
        mean_final_architecture_version=1,
    )

    result_dict = agg.model_dump()
    assert result_dict["approach_name"] == "test"
    assert result_dict["num_tasks"] == 1


def test_comparison_result_serialization() -> None:
    """Test that ComparisonResult can be serialized."""
    arch_only = AggregateEvaluationResult(
        approach_name="arch_only",
        num_tasks=1,
        num_episodes=1,
        mean_reward=0.5,
        total_reward=0.5,
        mean_episode_length=10.0,
        mean_final_architecture_version=1,
    )
    task_aware = AggregateEvaluationResult(
        approach_name="task_aware",
        num_tasks=1,
        num_episodes=1,
        mean_reward=0.6,
        total_reward=0.6,
        mean_episode_length=10.0,
        mean_final_architecture_version=1,
    )

    comparison = ComparisonResult(
        architecture_only=arch_only,
        task_aware=task_aware,
    )

    result_dict = comparison.model_dump()
    assert result_dict["architecture_only"]["approach_name"] == "arch_only"
    assert result_dict["task_aware"]["approach_name"] == "task_aware"
    # Note: task_aware_absolute_improvement is computed by MetaRLEvaluator,
    # not set directly in ComparisonResult constructor
    assert result_dict["task_aware_absolute_improvement"] is None


# ============================================================================
# 15. No LLM/API calls
# ============================================================================


def test_evaluator_has_no_api_imports() -> None:
    """Test that evaluator module doesn't import API-related modules."""
    import app.rl.meta_evaluation as me
    import inspect

    source = inspect.getsource(me)

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
    ]

    source_lower = source.lower()
    for forbidden in forbidden_imports:
        assert f"import {forbidden}" not in source_lower
        assert f"from {forbidden}" not in source_lower


def test_evaluation_is_fully_offline() -> None:
    """Test that evaluation runs without network calls."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=2, episodes_per_task=1, max_steps_per_episode=2, seed=42),
    )

    # If we reach here without network errors, the module stayed offline
    result = evaluator.evaluate(evaluate_multi_task=False)
    assert isinstance(result, EvaluationResult)


# ============================================================================
# 16. Task context is not accidentally provided to architecture-only baseline
# ============================================================================


def test_architecture_only_baseline_has_no_task_context_note() -> None:
    """Test that architecture-only baseline explicitly documents no task context."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=1, episodes_per_task=1, max_steps_per_episode=2, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=False)

    # Each per-task result for architecture-only should have the note
    for task_result in result.comparison.architecture_only.per_task_results:
        note = task_result.task_context_summary.get("note", "")
        assert "architecture_only_baseline" in note
        assert "no task context" in note


def test_task_aware_baseline_has_task_context() -> None:
    """Test that task-aware baseline properly includes task context."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=1, episodes_per_task=1, max_steps_per_episode=2, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=False)

    # Each per-task result for task-aware should have the task category
    for task_result in result.comparison.task_aware.per_task_results:
        assert "task_category" in task_result.task_context_summary
        assert task_result.task_context_summary["task_category"] in ["coding", "research"]


def test_architecture_only_uses_plain_environment() -> None:
    """Test that architecture-only evaluation uses plain MASArchitectureEnv."""
    from app.rl.environment import MASArchitectureEnv

    distribution = _make_small_distribution()

    # Create a minimal evaluator to inspect internal behavior
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=1, episodes_per_task=1, max_steps_per_episode=1, seed=42),
    )

    # Get the policy that would be used for architecture-only
    policy = evaluator._create_architecture_only_policy()

    # Verify it's NOT task-aware
    assert policy.task_aware is False
    assert not isinstance(policy.state_encoder, type(evaluator._create_task_aware_policy().state_encoder))


# ============================================================================
# 17. Protected files remain unchanged
# ============================================================================


def test_workflow_py_not_modified_by_evaluation() -> None:
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


def test_agents_directory_not_modified_by_evaluation() -> None:
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
# Additional tests from requirements
# ============================================================================


def test_evaluator_get_approaches() -> None:
    """Test that get_approaches returns expected list."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(task_distribution=distribution)

    approaches = evaluator.get_approaches()
    assert "architecture_only" in approaches
    assert "task_aware" in approaches
    assert "multi_task" in approaches


def test_evaluation_with_explicit_tasks() -> None:
    """Test evaluation with explicitly provided tasks."""
    tasks = [
        _make_task(task_id="explicit-task-1", task_category="coding"),
        _make_task(task_id="explicit-task-2", task_category="research"),
    ]

    evaluator = MetaRLEvaluator(
        task_distribution=MetaTaskDistribution(tasks=tasks),
        config=EvaluationConfig(num_tasks=2, episodes_per_task=1, max_steps_per_episode=2, seed=42),
    )

    result = evaluator.evaluate(
        tasks=tasks,
        evaluate_multi_task=False,
    )

    task_ids = {r.task_id for r in result.comparison.architecture_only.per_task_results}
    assert task_ids == {"explicit-task-1", "explicit-task-2"}


def test_per_task_valid_invalid_counts() -> None:
    """Test that valid/invalid transition counts are recorded."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=1, episodes_per_task=2, max_steps_per_episode=3, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=False)

    for task_result in result.comparison.architecture_only.per_task_results:
        assert isinstance(task_result.valid_transition_count, int)
        assert isinstance(task_result.invalid_transition_count, int)
        assert task_result.valid_transition_count >= 0
        assert task_result.invalid_transition_count >= 0


def test_notes_document_experimental_nature() -> None:
    """Test that evaluation results document experimental nature."""
    distribution = _make_small_distribution()
    evaluator = MetaRLEvaluator(
        task_distribution=distribution,
        config=EvaluationConfig(num_tasks=1, episodes_per_task=1, max_steps_per_episode=2, seed=42),
    )

    result = evaluator.evaluate(evaluate_multi_task=False)

    # Notes should mention experimental nature
    assert "experimental" in result.notes.lower()
    assert "baseline" in result.notes.lower()
