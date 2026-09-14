import json

import pytest

from app.rl.meta_neural_trainer import (
    FastAdaptationResult,
    MetaNeuralTrainer,
    MetaTrainingResult,
    deterministic_train_test_split,
    run_tiny_step27_smoke,
)
from app.rl.task_distribution import create_baseline_task_distribution


def _trainer(seed: int = 17) -> MetaNeuralTrainer:
    distribution = create_baseline_task_distribution()
    train, test = deterministic_train_test_split(
        distribution, train_ratio=0.7, seed=seed
    )
    target = distribution.get_task("task-research-002")
    training_tasks = [
        task for task in train.get_tasks() if task.task_id != target.task_id
    ]
    return MetaNeuralTrainer(
        task_distribution=distribution,
        training_tasks=training_tasks[:3],
        unseen_tasks=[target],
        max_steps=1,
        meta_updates=3,
        adaptation_updates=2,
        learning_rate=1e-3,
        seed=seed,
    )


def test_split_is_deterministic_and_non_overlapping() -> None:
    distribution = create_baseline_task_distribution()
    first = deterministic_train_test_split(distribution, train_ratio=0.7, seed=11)
    second = deterministic_train_test_split(distribution, train_ratio=0.7, seed=11)

    assert first[0].task_ids() == second[0].task_ids()
    assert first[1].task_ids() == second[1].task_ids()
    assert set(first[0].task_ids()).isdisjoint(first[1].task_ids())


def test_shared_policy_updates_across_multiple_tasks() -> None:
    trainer = _trainer()
    before = trainer.policy.clone_parameters()

    result = trainer.meta_train(num_updates=3)

    assert result.update_task_ids == trainer.training_task_ids
    assert trainer.policy.parameters_changed_since(before)


def test_meta_training_records_finite_changed_loss() -> None:
    result = _trainer().train()

    assert len(result.loss_curve) == result.num_updates
    assert all(value >= 0 for value in result.loss_curve)
    assert result.initial_loss != result.final_loss or result.num_updates == 1


def test_meta_training_is_deterministic() -> None:
    first = _trainer(seed=23).train()
    second = _trainer(seed=23).train()

    assert first == second


def test_unseen_task_adaptation_records_before_after_and_scratch() -> None:
    trainer = _trainer(seed=29)
    trainer.train()

    result = trainer.adapt_and_evaluate()

    assert result.target_task_id in trainer.unseen_task_ids
    assert result.perturbed_start_performance < 1.0
    assert result.meta_policy_before_adaptation == result.perturbed_start_performance
    assert 0.0 <= result.meta_policy_after_adaptation <= 1.0
    assert 0.0 <= result.fresh_policy_after_adaptation <= 1.0
    assert len(result.meta_adaptation_losses) == 2
    assert len(result.from_scratch_losses) == 2
    assert result.perturbation["agent_id"] == "researcher"
    assert result.perturbation["new_role"] == "analysis"


def test_target_perturbation_produces_suboptimal_performance() -> None:
    trainer = _trainer(seed=30)
    architecture = trainer._perturbed_target_architecture()

    performance = trainer.evaluator.evaluate(trainer.unseen_tasks[0], architecture)

    assert performance.task_success_score < 1.0
    assert "research" in performance.missing_capabilities


def test_meta_and_fresh_branches_start_from_equivalent_perturbed_architectures() -> (
    None
):
    trainer = _trainer(seed=32)

    meta_start = trainer._perturbed_target_architecture()
    fresh_start = trainer._perturbed_target_architecture()

    assert meta_start.model_dump() == fresh_start.model_dump()


def test_adaptation_evaluates_through_task_performance_evaluator() -> None:
    trainer = _trainer(seed=33)
    calls = []
    original_evaluate = trainer.evaluator.evaluate

    def recording_evaluate(task, architecture, **kwargs):
        calls.append((task.task_id, architecture.model_dump()))
        return original_evaluate(task, architecture, **kwargs)

    trainer.evaluator.evaluate = recording_evaluate
    trainer.train()
    result = trainer.adapt_and_evaluate()

    assert result.target_task_id in {task_id for task_id, _ in calls}
    assert len(calls) >= 3


def test_adaptation_updates_policy_parameters() -> None:
    trainer = _trainer(seed=31)
    trainer.train()
    target = trainer.unseen_tasks[0]
    adapted = trainer._clone_policy(trainer.policy, seed=31)
    before = adapted.clone_parameters()

    trainer._adaptation_update(
        adapted,
        target,
        initial_architecture=trainer._perturbed_target_architecture(),
    )

    assert adapted.parameters_changed_since(before)


def test_adaptation_metrics_are_calculated_correctly() -> None:
    result = _trainer(seed=37).adapt_and_evaluate(adaptation_updates=0)

    assert result.adaptation_improvement == pytest.approx(
        result.post_adaptation_performance - result.pre_adaptation_performance
    )
    assert result.transfer_adaptation_advantage == pytest.approx(
        result.post_adaptation_performance - result.from_scratch_performance
    )


def test_results_are_json_serializable() -> None:
    trainer = _trainer(seed=41)
    training = trainer.train()
    adaptation = trainer.adapt_and_evaluate()

    json.dumps(training.serialize())
    json.dumps(adaptation.serialize())
    assert isinstance(training, MetaTrainingResult)
    assert isinstance(adaptation, FastAdaptationResult)
    assert adaptation.perturbed_start_performance < 1.0


def test_repeated_smoke_runs_are_identical() -> None:
    assert run_tiny_step27_smoke(seed=43) == run_tiny_step27_smoke(seed=43)


def test_training_and_unseen_ids_are_exposed() -> None:
    trainer = _trainer(seed=47)

    assert trainer.training_task_ids
    assert trainer.unseen_task_ids
    assert set(trainer.training_task_ids).isdisjoint(trainer.unseen_task_ids)


def test_invalid_overlap_is_rejected() -> None:
    distribution = create_baseline_task_distribution()
    task = distribution.tasks[0]

    with pytest.raises(ValueError, match="must not overlap"):
        MetaNeuralTrainer(training_tasks=[task], unseen_tasks=[task], seed=53)
