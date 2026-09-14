"""
Multi-seed task transfer and generalization evaluation for the adaptive MAS research project.

This module provides TransferGeneralizationEvaluator, which extends the Step 22
TaskTransferExperiment to evaluate transfer/generalization across:

* Multiple random seeds
* Multiple unseen target tasks
* Multiple source-task configurations where supported

The evaluator compares:

A. FROM-SCRATCH:
    Fresh task-aware Q-learning on the target task.

B. TRANSFER:
    Q-learning first trained on source tasks, then adapted to the unseen target task.

The purpose is to determine whether transfer consistently improves adaptation
rather than relying on a single smoke-test result.

This is an EVALUATION/EXPERIMENTAL step. It does NOT implement a new RL algorithm.
It continues to use the existing tabular Q-learning implementation from Step 12/17.

Design notes
------------
* Uses existing tabular Q-learning (Step 12/17).
* Uses existing task-aware state encoding (Step 17).
* Uses existing MetaTrainer for training phase (Step 18).
* Uses existing TaskTransferExperiment from Step 22.
* Uses existing task-performance evaluation (Step 20).
* Uses existing task-aware reward mechanism (Step 21).
* Does NOT implement PPO, DQN, MAML, Reptile, or neural networks.
* Fully offline and deterministic.
* No LLM calls, no API calls, no task execution.

Descriptive statistics
----------------------
This module computes simple descriptive statistics (mean, stdev, min, max)
using only the Python standard library. These are explicitly descriptive
experimental statistics, not inferential statistical claims.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from app.rl.meta_task import MetaTask
from app.rl.task_distribution import MetaTaskDistribution
from app.rl.task_transfer import (
    TransferConfig,
    TransferExperimentResult,
    TaskTransferExperiment,
)


# ===========================================================================
# Statistical helpers (standard library only)
# ===========================================================================


def _mean(values: List[float]) -> float:
    """Return the arithmetic mean of a non-empty list of floats."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _stdev(values: List[float]) -> Optional[float]:
    """Return the sample standard deviation of a list of floats.

    Returns None when the list has fewer than 2 elements (stdev is undefined).
    """
    if len(values) < 2:
        return None
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _min(values: List[float]) -> Optional[float]:
    """Return the minimum of a non-empty list, or None if empty."""
    if not values:
        return None
    return min(values)


def _max(values: List[float]) -> Optional[float]:
    """Return the maximum of a non-empty list, or None if empty."""
    if not values:
        return None
    return max(values)


# ===========================================================================
# Result models
# ===========================================================================


class PerSeedTransferResult(BaseModel):
    """Result of running the transfer experiment for a single seed.

    Attributes
    ----------
    seed : int
        The random seed used for this run.
    transfer_result : TransferExperimentResult
        The transfer experiment result for this seed.
    """

    seed: int = Field(description="Random seed used for this run")
    transfer_result: TransferExperimentResult = Field(
        description="Transfer experiment result for this seed"
    )


class PerTaskCrossSeedResult(BaseModel):
    """Aggregated result for one adaptation task across all seeds.

    Attributes
    ----------
    task_id : str
        The adaptation task ID.
    task_category : str
        Task category.
    difficulty : int
        Task difficulty.
    mean_transfer_final_reward : float
        Mean transfer final reward across seeds.
    mean_from_scratch_final_reward : float
        Mean from-scratch final reward across seeds.
    mean_transfer_advantage : float
        Mean transfer advantage across seeds.
    transfer_reward_stdev : Optional[float]
        Standard deviation of transfer final reward across seeds.
    from_scratch_reward_stdev : Optional[float]
        Standard deviation of from-scratch final reward across seeds.
    mean_transfer_final_task_success : float
        Mean transfer final task-success score across seeds.
    mean_from_scratch_final_task_success : float
        Mean from-scratch final task-success score across seeds.
    mean_task_success_advantage : float
        Mean task-success advantage across seeds.
    episode_curves : Dict[str, List[List[float]]]
        Episode-level reward curves: {"transfer": [...], "from_scratch": [...]}
        Each inner list is the per-seed per-episode rewards aligned by episode index.
    """

    task_id: str = Field(description="Adaptation task ID")
    task_category: str = Field(description="Task category")
    difficulty: int = Field(description="Task difficulty")
    mean_transfer_final_reward: float = Field(
        description="Mean transfer final reward across seeds"
    )
    mean_from_scratch_final_reward: float = Field(
        description="Mean from-scratch final reward across seeds"
    )
    mean_transfer_advantage: float = Field(
        description="Mean transfer advantage across seeds"
    )
    transfer_reward_stdev: Optional[float] = Field(
        default=None, description="Stdev of transfer final reward across seeds"
    )
    from_scratch_reward_stdev: Optional[float] = Field(
        default=None, description="Stdev of from-scratch final reward across seeds"
    )
    mean_transfer_final_task_success: float = Field(
        description="Mean transfer final task-success score across seeds"
    )
    mean_from_scratch_final_task_success: float = Field(
        description="Mean from-scratch final task-success score across seeds"
    )
    mean_task_success_advantage: float = Field(
        description="Mean task-success advantage across seeds"
    )
    episode_curves: Dict[str, List[List[float]]] = Field(
        default_factory=dict,
        description="Per-seed per-episode reward curves for adaptation visualization",
    )


class TransferGeneralizationResult(BaseModel):
    """Complete result of a multi-seed transfer generalization evaluation.

    Attributes
    ----------
    config : TransferGeneralizationConfig
        Configuration used for the evaluation.
    per_seed_results : List[PerSeedTransferResult]
        Results for each seed.
    per_task_results : List[PerTaskCrossSeedResult]
        Aggregated results per adaptation task across seeds.
    aggregate : AggregateGeneralizationMetrics
        Overall aggregate metrics across all seeds and tasks.
    notes : str
        Human-readable notes about the evaluation.
    """

    config: "TransferGeneralizationConfig" = Field(
        description="Configuration used for the evaluation"
    )
    per_seed_results: List[PerSeedTransferResult] = Field(
        default_factory=list,
        description="Results for each seed",
    )
    per_task_results: List[PerTaskCrossSeedResult] = Field(
        default_factory=list,
        description="Aggregated results per adaptation task across seeds",
    )
    aggregate: "AggregateGeneralizationMetrics" = Field(
        description="Overall aggregate metrics"
    )
    notes: str = Field(
        default="",
        description="Human-readable notes about the evaluation",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        return self.model_dump(mode="json", exclude_none=True)

    def serialize(self) -> Dict[str, Any]:
        """Alias for to_dict for consistency."""
        return self.to_dict()


class AggregateGeneralizationMetrics(BaseModel):
    """Overall aggregate metrics across all seeds and tasks.

    Attributes
    ----------
    num_seeds : int
        Number of seeds evaluated.
    num_source_tasks : int
        Number of source/training tasks.
    num_target_tasks : int
        Number of adaptation/target tasks.
    mean_transfer_reward : float
        Mean transfer final reward across all seeds and tasks.
    mean_from_scratch_reward : float
        Mean from-scratch final reward across all seeds and tasks.
    mean_transfer_advantage : float
        Mean transfer advantage across all seeds and tasks.
    transfer_advantage_stdev : Optional[float]
        Standard deviation of transfer advantage across seeds/tasks.
    transfer_advantage_min : Optional[float]
        Minimum transfer advantage observed.
    transfer_advantage_max : Optional[float]
        Maximum transfer advantage observed.
    mean_transfer_task_success : float
        Mean transfer final task-success score across all seeds and tasks.
    mean_from_scratch_task_success : float
        Mean from-scratch final task-success score across all seeds and tasks.
    mean_task_success_advantage : float
        Mean task-success advantage across all seeds and tasks.
    overall_notes : str
        Human-readable summary notes.
    """

    num_seeds: int = Field(description="Number of seeds evaluated")
    num_source_tasks: int = Field(description="Number of source/training tasks")
    num_target_tasks: int = Field(description="Number of adaptation/target tasks")
    mean_transfer_reward: float = Field(
        description="Mean transfer final reward across all seeds and tasks"
    )
    mean_from_scratch_reward: float = Field(
        description="Mean from-scratch final reward across all seeds and tasks"
    )
    mean_transfer_advantage: float = Field(
        description="Mean transfer advantage across all seeds and tasks"
    )
    transfer_advantage_stdev: Optional[float] = Field(
        default=None,
        description="Stdev of transfer advantage across seeds/tasks",
    )
    transfer_advantage_min: Optional[float] = Field(
        default=None,
        description="Minimum transfer advantage observed",
    )
    transfer_advantage_max: Optional[float] = Field(
        default=None,
        description="Maximum transfer advantage observed",
    )
    mean_transfer_task_success: float = Field(
        description="Mean transfer final task-success score across all seeds and tasks"
    )
    mean_from_scratch_task_success: float = Field(
        description="Mean from-scratch final task-success score across all seeds and tasks"
    )
    mean_task_success_advantage: float = Field(
        description="Mean task-success advantage across all seeds and tasks"
    )
    overall_notes: str = Field(
        default="",
        description="Human-readable summary notes",
    )


class TransferGeneralizationConfig(BaseModel):
    """Configuration for TransferGeneralizationEvaluator.

    Attributes
    ----------
    training_task_ids : List[str]
        Explicit task IDs to use for training.
    adaptation_task_ids : List[str]
        Explicit task IDs to use for adaptation (unseen tasks).
        Must not overlap with training tasks.
    seeds : List[int]
        Random seeds for repeated evaluation.
    training_episodes_per_task : int
        Number of episodes to train on each training task.
    adaptation_episodes : int
        Number of episodes to adapt on each adaptation task.
    max_steps : int
        Maximum steps per episode.
    task_performance_weight : float
        Weight for task-performance reward component (Step 21).
    learning_rate : float
        Q-learning alpha parameter.
    discount_factor : float
        Q-learning gamma parameter.
    epsilon : float
        Initial exploration rate.
    epsilon_min : float
        Minimum exploration rate.
    epsilon_decay : float
        Multiplicative decay factor per episode.
    role_options : Optional[List[str]]
        Optional role vocabulary for environments.
    """

    training_task_ids: List[str] = Field(
        default_factory=list,
        description="Training task IDs",
    )
    adaptation_task_ids: List[str] = Field(
        default_factory=list,
        description="Adaptation (unseen) task IDs",
    )
    seeds: List[int] = Field(
        default_factory=lambda: [42, 123, 456],
        description="Random seeds for repeated evaluation",
    )
    training_episodes_per_task: int = Field(
        default=3,
        ge=1,
        description="Episodes per training task",
    )
    adaptation_episodes: int = Field(
        default=5,
        ge=1,
        description="Episodes per adaptation task",
    )
    max_steps: int = Field(
        default=10,
        ge=1,
        description="Maximum steps per episode",
    )
    task_performance_weight: float = Field(
        default=0.5,
        ge=0.0,
        description="Task-performance reward weight (Step 21)",
    )
    learning_rate: float = Field(
        default=0.1,
        ge=0.001,
        le=1.0,
        description="Q-learning alpha",
    )
    discount_factor: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Q-learning gamma",
    )
    epsilon: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Initial exploration rate",
    )
    epsilon_min: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Minimum exploration rate",
    )
    epsilon_decay: float = Field(
        default=0.995,
        gt=0.0,
        le=1.0,
        description="Epsilon decay factor",
    )
    role_options: Optional[List[str]] = Field(
        default=None,
        description="Optional role vocabulary",
    )

    @field_validator("seeds")
    @classmethod
    def seeds_must_not_be_empty(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("seeds must contain at least one seed")
        return v

    @field_validator("adaptation_task_ids")
    @classmethod
    def adaptation_task_ids_must_not_be_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("adaptation_task_ids must contain at least one task")
        return v

    def validate(self, distribution: MetaTaskDistribution) -> None:
        """Validate configuration against a task distribution.

        Parameters
        ----------
        distribution : MetaTaskDistribution
            The task distribution containing all tasks.

        Raises
        ------
        ValueError
            If configuration is invalid.
        """
        training_ids = set(self.training_task_ids)
        adaptation_ids = set(self.adaptation_task_ids)

        # Check for overlap (data leakage prevention)
        overlap = training_ids & adaptation_ids
        if overlap:
            raise ValueError(
                f"Training and adaptation tasks must not overlap. "
                f"Found overlapping task IDs: {sorted(overlap)}"
            )

        # Validate that all task IDs exist in distribution
        all_distribution_ids = set(distribution.task_ids())
        missing_training = training_ids - all_distribution_ids
        missing_adaptation = adaptation_ids - all_distribution_ids

        if missing_training:
            raise ValueError(
                f"Training task IDs not found in distribution: {sorted(missing_training)}"
            )
        if missing_adaptation:
            raise ValueError(
                f"Adaptation task IDs not found in distribution: {sorted(missing_adaptation)}"
            )


# ===========================================================================
# TransferGeneralizationEvaluator
# ===========================================================================


class TransferGeneralizationEvaluator:
    """Multi-seed transfer/generalization evaluator.

    This evaluator extends the Step 22 TaskTransferExperiment to evaluate
    transfer/generalization across multiple seeds, multiple target tasks,
    and (where supported) multiple source-task configurations.

    The experiment compares:
    - FROM-SCRATCH: Fresh task-aware Q-learning on the target task.
    - TRANSFER: Q-learning first trained on source tasks, then adapted to
      the unseen target task.

    The experiment is fully deterministic when seeds are fixed.
    """

    def __init__(
        self,
        distribution: MetaTaskDistribution,
        config: TransferGeneralizationConfig,
    ) -> None:
        """Initialize the evaluator.

        Parameters
        ----------
        distribution : MetaTaskDistribution
            The task distribution containing all tasks.
        config : TransferGeneralizationConfig
            Evaluation configuration.
        """
        self.distribution = distribution
        self.config = config
        self.config.validate(distribution)

        self._training_tasks = self._get_training_tasks()
        self._adaptation_tasks = self._get_adaptation_tasks()

    def _get_training_tasks(self) -> List[MetaTask]:
        """Get the training tasks based on configuration."""
        return [
            self.distribution.get_task(task_id)
            for task_id in self.config.training_task_ids
        ]

    def _get_adaptation_tasks(self) -> List[MetaTask]:
        """Get the adaptation (unseen) tasks based on configuration."""
        return [
            self.distribution.get_task(task_id)
            for task_id in self.config.adaptation_task_ids
        ]

    def run(self) -> TransferGeneralizationResult:
        """Run the full multi-seed transfer generalization evaluation.

        Returns
        -------
        TransferGeneralizationResult
            Complete evaluation results.
        """
        per_seed_results: List[PerSeedTransferResult] = []
        # Per-task cross-seed accumulators
        task_final_transfer_rewards: Dict[str, List[float]] = {}
        task_final_from_scratch_rewards: Dict[str, List[float]] = {}
        task_final_transfer_success: Dict[str, List[float]] = {}
        task_final_from_scratch_success: Dict[str, List[float]] = {}
        task_advantages: Dict[str, List[float]] = {}
        task_success_advantages: Dict[str, List[float]] = {}
        task_episode_curves: Dict[str, Dict[str, List[List[float]]]] = {}

        for seed in self.config.seeds:
            # Create a seed-specific config wrapping the shared hyperparameters
            seed_config = TransferConfig(
                training_task_ids=self.config.training_task_ids,
                adaptation_task_ids=self.config.adaptation_task_ids,
                training_episodes_per_task=self.config.training_episodes_per_task,
                adaptation_episodes=self.config.adaptation_episodes,
                max_steps=self.config.max_steps,
                seed=seed,
                task_performance_weight=self.config.task_performance_weight,
                learning_rate=self.config.learning_rate,
                discount_factor=self.config.discount_factor,
                epsilon=self.config.epsilon,
                epsilon_min=self.config.epsilon_min,
                epsilon_decay=self.config.epsilon_decay,
                role_options=self.config.role_options,
            )

            experiment = TaskTransferExperiment(self.distribution, seed_config)
            result = experiment.run()
            per_seed_results.append(
                PerSeedTransferResult(seed=seed, transfer_result=result)
            )

            # Accumulate per-task metrics across seeds
            for scratch_result, transfer_result in zip(
                result.from_scratch_results,
                result.transfer_results,
            ):
                task_id = scratch_result.task_id
                # Final episode rewards
                task_final_transfer_rewards.setdefault(task_id, []).append(
                    transfer_result.final_reward
                )
                task_final_from_scratch_rewards.setdefault(task_id, []).append(
                    scratch_result.final_reward
                )
                task_final_transfer_success.setdefault(task_id, []).append(
                    transfer_result.final_task_success_score
                )
                task_final_from_scratch_success.setdefault(task_id, []).append(
                    scratch_result.final_task_success_score
                )
                # Advantages
                task_advantages.setdefault(task_id, []).append(
                    transfer_result.final_reward - scratch_result.final_reward
                )
                task_success_advantages.setdefault(task_id, []).append(
                    transfer_result.final_task_success_score
                    - scratch_result.final_task_success_score
                )
                # Episode curves: store per-seed per-episode reward sequences
                transfer_curve = [
                    ep.total_reward for ep in transfer_result.episode_results
                ]
                scratch_curve = [
                    ep.total_reward for ep in scratch_result.episode_results
                ]
                if task_id not in task_episode_curves:
                    task_episode_curves[task_id] = {
                        "transfer": [],
                        "from_scratch": [],
                    }
                task_episode_curves[task_id]["transfer"].append(transfer_curve)
                task_episode_curves[task_id]["from_scratch"].append(scratch_curve)

        # Build per-task cross-seed results
        per_task_results: List[PerTaskCrossSeedResult] = []
        for task in self._adaptation_tasks:
            task_id = task.task_id
            transfer_rewards = task_final_transfer_rewards.get(task_id, [])
            scratch_rewards = task_final_from_scratch_rewards.get(task_id, [])
            transfer_success = task_final_transfer_success.get(task_id, [])
            scratch_success = task_final_from_scratch_success.get(task_id, [])
            advantages = task_advantages.get(task_id, [])
            success_advantages = task_success_advantages.get(task_id, [])
            curves = task_episode_curves.get(task_id, {"transfer": [], "from_scratch": []})

            mean_transfer_final = _mean(transfer_rewards)
            mean_scratch_final = _mean(scratch_rewards)
            mean_advantage = _mean(advantages)
            stdev_transfer = _stdev(transfer_rewards)
            stdev_scratch = _stdev(scratch_rewards)
            mean_transfer_success = _mean(transfer_success)
            mean_scratch_success = _mean(scratch_success)
            mean_success_advantage = _mean(success_advantages)

            per_task_results.append(
                PerTaskCrossSeedResult(
                    task_id=task_id,
                    task_category=task.task_category,
                    difficulty=task.difficulty,
                    mean_transfer_final_reward=mean_transfer_final,
                    mean_from_scratch_final_reward=mean_scratch_final,
                    mean_transfer_advantage=mean_advantage,
                    transfer_reward_stdev=stdev_transfer,
                    from_scratch_reward_stdev=stdev_scratch,
                    mean_transfer_final_task_success=mean_transfer_success,
                    mean_from_scratch_final_task_success=mean_scratch_success,
                    mean_task_success_advantage=mean_success_advantage,
                    episode_curves=curves,
                )
            )

        # Aggregate metrics across all seeds and tasks
        all_transfer_rewards: List[float] = []
        all_from_scratch_rewards: List[float] = []
        all_advantages: List[float] = []
        all_transfer_success: List[float] = []
        all_from_scratch_success: List[float] = []
        all_success_advantages: List[float] = []

        for task_id in task_final_transfer_rewards:
            all_transfer_rewards.extend(task_final_transfer_rewards[task_id])
            all_from_scratch_rewards.extend(task_final_from_scratch_rewards[task_id])
            all_advantages.extend(task_advantages[task_id])
            all_transfer_success.extend(task_final_transfer_success[task_id])
            all_from_scratch_success.extend(task_final_from_scratch_success[task_id])
            all_success_advantages.extend(task_success_advantages[task_id])

        all_adv = all_advantages
        aggregate = AggregateGeneralizationMetrics(
            num_seeds=len(self.config.seeds),
            num_source_tasks=len(self._training_tasks),
            num_target_tasks=len(self._adaptation_tasks),
            mean_transfer_reward=_mean(all_transfer_rewards),
            mean_from_scratch_reward=_mean(all_from_scratch_rewards),
            mean_transfer_advantage=_mean(all_adv),
            transfer_advantage_stdev=_stdev(all_adv),
            transfer_advantage_min=_min(all_adv),
            transfer_advantage_max=_max(all_adv),
            mean_transfer_task_success=_mean(all_transfer_success),
            mean_from_scratch_task_success=_mean(all_from_scratch_success),
            mean_task_success_advantage=_mean(all_success_advantages),
            overall_notes=(
                "Multi-seed transfer generalization evaluation. "
                "Transfer advantage = transfer_final_reward - from_scratch_final_reward. "
                "Positive advantage indicates better transfer performance under this experiment. "
                "A negative or zero value is also a valid result. "
                "These are descriptive experimental statistics, not inferential claims. "
                "Task-performance scores are deterministic compatibility proxies, NOT actual LLM task execution."
            ),
        )

        notes = (
            "Multi-seed transfer generalization evaluation. "
            f"Evaluated {len(self.config.seeds)} seeds. "
            f"Source tasks: {len(self._training_tasks)}. "
            f"Target tasks: {len(self._adaptation_tasks)}. "
            "Transfer condition uses Q-table learned from source tasks. "
            "From-scratch condition starts with fresh Q-table. "
            "This is a baseline evaluation, NOT a complete Meta-RL algorithm. "
            "Task-performance scores are deterministic compatibility proxies, NOT actual LLM task execution."
        )

        return TransferGeneralizationResult(
            config=self.config,
            per_seed_results=per_seed_results,
            per_task_results=per_task_results,
            aggregate=aggregate,
            notes=notes,
        )

    def calculate_sample_efficiency(
        self,
        result: TransferGeneralizationResult,
        reward_threshold: float,
    ) -> Dict[str, Any]:
        """Calculate simple sample-efficiency metrics.

        For each condition, measures the number of episodes required to reach
        a configurable reward threshold for the first time, averaged across
        seeds and tasks. If the threshold is never reached, returns None for
        that value rather than fabricating a value.

        Parameters
        ----------
        result : TransferGeneralizationResult
            The evaluation result.
        reward_threshold : float
            The reward threshold to measure time-to-threshold.

        Returns
        -------
        Dict[str, Any]
            Dictionary with sample-efficiency metrics.
        """
        # Accumulate time-to-threshold per seed per task per condition
        transfer_times: List[Optional[int]] = []
        from_scratch_times: List[Optional[int]] = []

        for per_task in result.per_task_results:
            transfer_curves = per_task.episode_curves.get("transfer", [])
            from_scratch_curves = per_task.episode_curves.get("from_scratch", [])

            for curve in transfer_curves:
                first_ep = _first_episode_above(curve, reward_threshold)
                transfer_times.append(first_ep)
            for curve in from_scratch_curves:
                first_ep = _first_episode_above(curve, reward_threshold)
                from_scratch_times.append(first_ep)

        # Convert None (never reached) to a sentinel for averaging
        def _avg_or_none(values: List[Optional[int]]) -> Optional[float]:
            valid = [v for v in values if v is not None]
            if not valid:
                return None
            return _mean(valid)

        transfer_avg = _avg_or_none(transfer_times)
        from_scratch_avg = _avg_or_none(from_scratch_times)

        return {
            "reward_threshold": reward_threshold,
            "transfer_mean_episodes_to_threshold": transfer_avg,
            "from_scratch_mean_episodes_to_threshold": from_scratch_avg,
            "transfer_thresholds_reached": sum(1 for v in transfer_times if v is not None),
            "from_scratch_thresholds_reached": sum(
                1 for v in from_scratch_times if v is not None
            ),
            "notes": (
                "Sample-efficiency metric: mean number of episodes to first reach "
                "the reward threshold, averaged across seeds and tasks. "
                "If the threshold is never reached for a run, it is excluded from the mean. "
                "A None value indicates the threshold was never reached in any run. "
                "This is a deterministic descriptive metric, not a claim of superior efficiency."
            ),
        }


def _first_episode_above(
    rewards: List[float],
    threshold: float,
) -> Optional[int]:
    """Return the 0-based episode index of the first reward >= threshold.

    Returns None if no episode reaches the threshold.
    """
    for idx, r in enumerate(rewards):
        if r >= threshold:
            return idx
    return None
