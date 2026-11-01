"""
Comprehensive research metrics and TensorBoard instrumentation test suite.
Validates metric formulas, multi-seed aggregation, baseline comparison,
JSONL schema backward compatibility, and TensorBoard logging.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from app.agents.planner import PlannerOutput
from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.manager import ArchitectureManager
from app.architecture.models import MASArchitecture
from app.evaluation.experiment_reproducibility import (
    SummaryStatistics,
    collect_experiment_metadata,
    compute_summary_statistics,
    format_baseline_comparison_latex,
    format_baseline_comparison_markdown,
    generate_latex_table,
    generate_markdown_table,
)
from app.evaluation.metrics_logger import MetricsLogger
from app.evaluation.theoretical_evaluator import (
    ComprehensiveArchitectureEvaluation,
    TheoreticalArchitectureEvaluator,
)
from app.graph.dynamic_builder import DynamicGraphBuilder
from app.rl.meta_neural_trainer import (
    MetaNeuralTrainer,
    deterministic_train_test_split,
)
from app.rl.task_conditioned_policy import (
    TaskConditionedPolicy,
    create_task_conditioned_policy_from_environment,
)
from app.rl.task_distribution import create_baseline_task_distribution
from app.runtime.orchestrator import DynamicGraphExecutionResult


def _create_mock_exec_result(
    task: str = "Test Task Prompt",
    agents: List[str] | None = None,
    execution_time_ms: float = 125.4,
    build_time_ms: float = 12.3,
    compilation_time_ms: float = 8.1,
    recovery_count: int = 0,
    conditional_route_count: int = 1,
) -> DynamicGraphExecutionResult:
    invoked = agents or ["planner", "coder", "finalizer"]
    arch = {
        "architecture_id": "test_arch_v1",
        "agents": [
            {
                "agent_id": a,
                "role": a,
                "description": f"{a} description",
                "active": True,
            }
            for a in invoked
        ],
        "communication_edges": [
            {"source": invoked[i], "target": invoked[i + 1]}
            for i in range(len(invoked) - 1)
        ],
    }
    return DynamicGraphExecutionResult(
        task=task,
        planner_output={"task": task, "required_capabilities": ["coding"]},
        architecture_decision={"action": "none"},
        initial_architecture=arch,
        final_architecture=arch,
        graph_nodes=invoked,
        graph_edges=[
            {"source": invoked[i], "target": invoked[i + 1]}
            for i in range(len(invoked) - 1)
        ],
        active_agents=invoked,
        architecture_version=1,
        compiled=True,
        graph_representation=" -> ".join(invoked),
        execution_path=invoked,
        agents_invoked=invoked,
        execution_time_ms=execution_time_ms,
        build_time_ms=build_time_ms,
        compilation_time_ms=compilation_time_ms,
        node_count=len(invoked),
        edge_count=max(0, len(invoked) - 1),
        recovery_count=recovery_count,
        conditional_route_count=conditional_route_count,
    )


# ---------------------------------------------------------------------------
# 1. Metric Calculation Unit Tests
# ---------------------------------------------------------------------------

def test_structural_efficiency_calculation_and_bounds() -> None:
    evaluator = TheoreticalArchitectureEvaluator()
    task = create_baseline_task_distribution().get_task("task-research-001")
    manager = ArchitectureManager.create_default_architecture()
    arch = manager.get_architecture()

    eval_result = evaluator.evaluate(
        architecture=arch,
        required_capabilities=task.required_capabilities,
    )
    assert hasattr(eval_result, "structural_efficiency")
    assert 0.0 <= eval_result.structural_efficiency <= 1.0

    # Test edge case: single agent architecture
    single_agent_arch = MASArchitecture(
        version="v_single",
        agents=[
            {
                "agent_id": "planner",
                "role": "planning",
                "description": "Planner agent",
                "active": True,
            }
        ],
        communication_edges=[],
    )
    single_res = evaluator.evaluate(
        architecture=single_agent_arch,
        required_capabilities=task.required_capabilities,
    )
    assert 0.0 <= single_res.structural_efficiency <= 1.0

    # Test edge case: architecture with no active agents
    empty_arch = MASArchitecture(
        version="v_empty",
        agents=[
            {
                "agent_id": "planner",
                "role": "planning",
                "description": "Planner agent",
                "active": False,
            }
        ],
        communication_edges=[],
    )
    empty_res = evaluator.evaluate(
        architecture=empty_arch,
        required_capabilities=task.required_capabilities,
    )
    assert empty_res.structural_efficiency == 0.0


def test_neural_policy_gradient_and_parameter_norm() -> None:
    trainer = MetaNeuralTrainer(
        task_distribution=create_baseline_task_distribution(),
        max_steps=1,
        meta_updates=1,
        adaptation_updates=1,
        seed=42,
    )
    env = trainer._make_environment()
    obs, _ = env.reset()
    policy = trainer.policy
    policy.action_mapper = env.mapper

    # Prior to update
    assert policy.last_update_metrics == {}

    # Perform update with metrics
    step_metrics = policy.update_from_transition_with_metrics(
        observation=obs,
        task_context=trainer.training_tasks[0].to_context(),
        action_id=0,
        reward=1.0,
        next_observation=obs,
        done=True,
        valid_action_ids=list(range(env.mapper.action_count)),
        next_valid_action_ids=list(range(env.mapper.action_count)),
    )

    assert "loss" in step_metrics
    assert "policy_gradient_norm" in step_metrics
    assert "parameter_norm" in step_metrics
    assert "action_distribution_entropy" in step_metrics
    assert "learning_rate" in step_metrics

    assert step_metrics["policy_gradient_norm"] >= 0.0
    assert step_metrics["parameter_norm"] > 0.0
    assert step_metrics["action_distribution_entropy"] >= 0.0
    assert not math.isnan(step_metrics["policy_gradient_norm"])
    assert not math.isinf(step_metrics["policy_gradient_norm"])


def test_reproducibility_summary_statistics() -> None:
    values = [10.0, 12.0, 14.0, 16.0, 18.0]
    stats = compute_summary_statistics(values)

    assert stats.sample_count == 5
    assert stats.mean == 14.0
    assert stats.min == 10.0
    assert stats.max == 18.0
    assert stats.std > 0.0
    assert stats.ci95_low < stats.mean < stats.ci95_high
    assert stats.ci95_margin > 0.0

    # Edge cases
    empty_stats = compute_summary_statistics([])
    assert empty_stats.sample_count == 0
    assert empty_stats.mean == 0.0

    single_stats = compute_summary_statistics([5.0])
    assert single_stats.sample_count == 1
    assert single_stats.mean == 5.0
    assert single_stats.std == 0.0


# ---------------------------------------------------------------------------
# 2. TensorBoard Logging Tests
# ---------------------------------------------------------------------------

def test_metrics_logger_tensorboard_tags_and_fallback(tmp_path: Path) -> None:
    logger = MetricsLogger(log_base_dir=str(tmp_path))

    exec_result = _create_mock_exec_result(
        task="Test Task Prompt",
        agents=["planner", "coder", "finalizer"],
        execution_time_ms=125.4,
        build_time_ms=12.3,
        compilation_time_ms=8.1,
    )

    logged = logger.log_task_run(
        step=1,
        task="Test Task Prompt",
        result=exec_result,
        architecture=exec_result.final_architecture,
        required_capabilities=["coding"],
        experiment_id="test_exp_01",
        seed=42,
        evaluation_mode="test_mode",
        measurement_type="simulated",
        actual_tokens=1500,
        actual_cost_usd=0.00225,
    )

    assert logged["step"] == 1
    assert logged["task"] == "Test Task Prompt"
    assert "experiment_metadata" in logged
    assert logged["experiment_metadata"]["experiment_id"] == "test_exp_01"
    assert "graph_metrics" in logged
    assert logged["graph_metrics"]["build_time_ms"] == 12.3
    assert "action_metrics" in logged
    assert "architecture_metrics" in logged
    assert logged["architecture_metrics"]["active_agent_count"] == 3

    # Log Neural and Meta Metrics
    logger.log_neural_metrics(
        step=1,
        metrics={
            "loss": 0.045,
            "reward": 0.88,
            "gradient_norm": 0.12,
            "parameter_norm": 14.5,
            "entropy": 1.23,
            "learning_rate": 0.001,
        },
    )

    logger.log_meta_metrics(
        step=1,
        metrics={
            "train_task_reward": 0.90,
            "held_out_task_reward": 0.82,
            "generalization_gap": 0.08,
            "initial_adaptation_reward": 0.50,
            "final_adaptation_reward": 0.85,
            "adaptation_gain": 0.35,
            "adaptation_steps": 2.0,
        },
    )

    logger.close()

    # Verify JSONL log was written
    assert logger.jsonl_file.exists()
    with open(logger.jsonl_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record["step"] == 1
        assert record["is_proxy"] is True
        assert record["overall_score"] > 0.0


# ---------------------------------------------------------------------------
# 3. JSONL Backward Compatibility and Missing Field Handling
# ---------------------------------------------------------------------------

def test_jsonl_backward_compatibility_with_legacy_readers(tmp_path: Path) -> None:
    logger = MetricsLogger(log_base_dir=str(tmp_path))

    exec_result = _create_mock_exec_result(
        task="Chit chat greeting",
        agents=["planner", "finalizer"],
    )

    logged = logger.log_task_run(
        step=1,
        task="Chit chat greeting",
        result=exec_result,
        architecture=exec_result.final_architecture,
    )
    logger.close()

    with open(logger.jsonl_file, "r", encoding="utf-8") as f:
        record = json.loads(f.readline())

    # Check that standard keys exist at root
    expected_legacy_keys = [
        "step",
        "task",
        "timestamp",
        "estimated_tokens",
        "estimated_cost_usd",
        "active_agents",
        "classification",
        "net_utility",
        "critical_path_length",
        "coverage_score",
        "structural_efficiency",
        "overall_score",
        "is_proxy",
    ]
    for key in expected_legacy_keys:
        assert key in record, f"Missing key at root: {key}"

    # Missing optional metrics must be None or omitted, never fabricated
    assert record.get("actual_tokens") is None
    assert record.get("actual_cost_usd") is None


# ---------------------------------------------------------------------------
# 4. Baseline Comparison & Task Separation Tests
# ---------------------------------------------------------------------------

def test_four_way_baseline_comparison() -> None:
    distribution = create_baseline_task_distribution()
    train_dist, test_dist = deterministic_train_test_split(
        distribution, train_ratio=0.7, seed=42
    )
    unseen = test_dist.get_tasks()[:1]
    training = [t for t in train_dist.get_tasks() if t.task_id != unseen[0].task_id][:2]

    # Verify task separation: zero overlap
    train_ids = {t.task_id for t in training}
    unseen_ids = {t.task_id for t in unseen}
    assert train_ids.isdisjoint(unseen_ids)

    trainer = MetaNeuralTrainer(
        task_distribution=distribution,
        training_tasks=training,
        unseen_tasks=unseen,
        max_steps=1,
        meta_updates=2,
        adaptation_updates=2,
        learning_rate=1e-3,
        seed=42,
    )
    trainer.train()

    comparison = trainer.run_baseline_comparison(
        unseen_task=unseen[0],
        adaptation_updates=2,
        seeds=[42, 101],
    )

    assert "summary_statistics" in comparison
    stats = comparison["summary_statistics"]
    for method in ["shared_meta_policy", "fresh_scratch_policy", "deterministic_baseline", "random_policy"]:
        assert method in stats
        m_stat = stats[method]
        assert m_stat["sample_count"] == 2
        assert len(m_stat["average_adaptation_curve"]) == 3
        assert 0.0 <= m_stat["mean_final_performance"] <= 1.0

    # Test Markdown and LaTeX table generation
    md_table = format_baseline_comparison_markdown(comparison)
    latex_table = format_baseline_comparison_latex(comparison)

    assert "Shared Neural Policy (Ours)" in md_table
    assert "\\begin{table}" in latex_table
    assert "\\end{table}" in latex_table


def test_cross_task_generalization_evaluation() -> None:
    distribution = create_baseline_task_distribution()
    train_dist, test_dist = deterministic_train_test_split(
        distribution, train_ratio=0.7, seed=42
    )
    unseen = test_dist.get_tasks()[:2]
    training = [t for t in train_dist.get_tasks() if t.task_id not in {u.task_id for u in unseen}][:2]

    trainer = MetaNeuralTrainer(
        task_distribution=distribution,
        training_tasks=training,
        unseen_tasks=unseen,
        max_steps=1,
        meta_updates=2,
        adaptation_updates=2,
        seed=42,
    )
    trainer.train()

    gen_result = trainer.evaluate_generalization(unseen_tasks=unseen, adaptation_updates=2)
    assert gen_result.train_task_reward >= 0.0
    assert gen_result.held_out_task_reward >= 0.0
    assert gen_result.generalization_gap == round(gen_result.train_task_reward - gen_result.held_out_task_reward, 6)
    assert gen_result.adaptation_gain == round(gen_result.final_adaptation_reward - gen_result.initial_adaptation_reward, 6)
    assert len(gen_result.adaptation_curve) == 3


# ---------------------------------------------------------------------------
# 5. Graph Timing & Recovery Instrumentation Tests
# ---------------------------------------------------------------------------

def test_dynamic_graph_builder_instrumentation() -> None:
    builder = DynamicGraphBuilder()
    manager = ArchitectureManager.create_default_architecture()
    arch = manager.get_architecture()

    planner_output = PlannerOutput(
        task_understanding="Test coding prompt",
        requires_coding=True,
        requires_verification=False,
        requires_research=False,
        requires_tools=False,
        subtasks=[],
        selected_agents=["planner", "coder", "finalizer"],
    )
    handlers = {
        "planner": lambda state: {"messages": ["planned"]},
        "coder": lambda state: {"messages": ["coded"]},
        "finalizer": lambda state: {"messages": ["finalized"]},
    }

    result = builder.build(
        architecture=arch,
        planner_output=planner_output,
        node_handlers=handlers,
    )
    assert result.metadata.build_time_ms >= 0.0
    assert result.metadata.compilation_time_ms >= 0.0
    assert result.metadata.node_count > 0
    assert result.metadata.edge_count > 0
    assert result.metadata.recovery_count >= 0
    assert result.metadata.build_success is True
    assert result.metadata.compilation_success is True


# ---------------------------------------------------------------------------
# 6. Experiment Metadata Collection Test
# ---------------------------------------------------------------------------

def test_collect_experiment_metadata_fields() -> None:
    meta = collect_experiment_metadata(seed=123, hyperparameters={"lr": 0.001, "updates": 10})
    assert "git_commit_hash" in meta
    assert "timestamp" in meta
    assert meta["seed"] == 123
    assert "python_version" in meta
    assert "framework_versions" in meta
    assert "hardware_summary" in meta
    assert meta["hyperparameters"]["lr"] == 0.001
