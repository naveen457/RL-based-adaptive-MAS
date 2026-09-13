"""
Task transfer and adaptation baseline for the adaptive MAS research project.

This module provides TaskTransferExperiment, a controlled experiment that
answers: "Does knowledge learned from previously seen tasks help the RL
controller adapt to a new/unseen task?"

The experiment compares two conditions:

BASELINE A — FROM-SCRATCH:
    New/unseen task
        ↓
    Fresh task-aware Q-learning policy (empty Q-table)
        ↓
    Adaptation episodes
        ↓
    Task-performance result

BASELINE B — TRANSFER:
    Training tasks
        ↓
    Task-aware Q-learning with shared Q-table
        ↓
    Preserved learned Q-table
        ↓
    Unseen task
        ↓
    Adaptation episodes (same policy, starts with learned Q-table)
        ↓
    Task-performance result

This is a TRANSFER/ADAPTATION BASELINE. It is NOT yet the final advanced
Meta-RL algorithm.

Design notes
------------
* Uses existing tabular Q-learning (Step 12/17).
* Uses existing task-aware state encoding (Step 17).
* Uses existing MetaTrainer for training phase (Step 18).
* Uses existing task-performance evaluation (Step 20).
* Uses existing task-aware reward mechanism (Step 21).
* Does NOT implement PPO, DQN, MAML, Reptile, or neural networks.
* Fully offline and deterministic.
* No LLM calls, no API calls, no task execution.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from app.architecture.manager import ArchitectureManager
from app.rl.meta_task import MetaTask
from app.rl.meta_environment import MetaEnvironment
from app.rl.meta_trainer import MetaTrainer
from app.rl.q_learning import (
    QLearningPolicy,
    QLearningTrainer,
    QTable,
    TaskAwareStateEncoder,
)
from app.rl.task_distribution import MetaTaskDistribution
from app.evaluation.task_performance import TaskPerformanceEvaluator


# ============================================================================
# Configuration
# ============================================================================


class TransferConfig(BaseModel):
    """Configuration for TaskTransferExperiment.

    Attributes
    ----------
    training_task_ids : Optional[List[str]]
        Explicit task IDs to use for training. If None, sampled from
        training_distribution.
    adaptation_task_ids : List[str]
        Explicit task IDs to use for adaptation (unseen tasks).
        Must not overlap with training tasks.
    training_distribution : Optional[MetaTaskDistribution]
        Distribution to sample training tasks from if training_task_ids
        is None.
    training_episodes_per_task : int
        Number of episodes to train on each training task.
    adaptation_episodes : int
        Number of episodes to adapt on each adaptation (unseen) task.
    max_steps : int
        Maximum steps per episode.
    seed : int
        Random seed for reproducibility.
    task_performance_weight : float
        Weight for task-performance reward component (Step 21).
        Default 0.5 for task-aware transfer experiment.
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

    training_task_ids: Optional[List[str]] = Field(
        default=None,
        description="Explicit training task IDs (optional)",
    )
    adaptation_task_ids: List[str] = Field(
        default_factory=list,
        description="Explicit adaptation (unseen) task IDs",
    )
    training_distribution: Optional[MetaTaskDistribution] = Field(
        default=None,
        description="Distribution to sample training tasks from",
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
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
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
        # Get training task IDs
        if self.training_task_ids is not None:
            training_ids = set(self.training_task_ids)
        elif self.training_distribution is not None:
            training_ids = set(self.training_distribution.task_ids())
        else:
            training_ids = set()

        # Get adaptation task IDs
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


# ============================================================================
# Result Models
# ============================================================================


class TransferEpisodeResult(BaseModel):
    """Result of a single adaptation episode.

    Attributes
    ----------
    task_id : str
        The adaptation task ID.
    episode : int
        Episode index within this task.
    condition : str
        "from_scratch" or "transfer".
    total_reward : float
        Total reward for this episode.
    episode_length : int
        Number of steps taken.
    task_success_score : float
        Task-performance score at end of episode.
    structural_score : float
        Structural evaluation score at end of episode.
    """

    task_id: str = Field(description="Adaptation task ID")
    episode: int = Field(description="Episode index within this task")
    condition: str = Field(
        description="Condition: 'from_scratch' or 'transfer'"
    )
    total_reward: float = Field(description="Total reward for this episode")
    episode_length: int = Field(description="Number of steps taken")
    task_success_score: float = Field(
        description="Task-performance score at end of episode"
    )
    structural_score: float = Field(
        description="Structural evaluation score at end of episode"
    )


class TransferTaskResult(BaseModel):
    """Aggregate result for one adaptation task across all episodes.

    Attributes
    ----------
    task_id : str
        The adaptation task ID.
    task_category : str
        Task category.
    difficulty : int
        Task difficulty.
    condition : str
        "from_scratch" or "transfer".
    initial_reward : float
        Reward from first adaptation episode.
    final_reward : float
        Reward from last adaptation episode.
    mean_reward : float
        Mean reward across all adaptation episodes.
    initial_task_success_score : float
        Task-success score from first episode.
    final_task_success_score : float
        Task-success score from last episode.
    mean_task_success_score : float
        Mean task-success score across episodes.
    initial_episode_length : int
        Episode length from first episode.
    final_episode_length : int
        Episode length from last episode.
    mean_episode_length : float
        Mean episode length across episodes.
    episode_results : List[TransferEpisodeResult]
        All episode-level results for adaptation curve.
    """

    task_id: str = Field(description="Adaptation task ID")
    task_category: str = Field(description="Task category")
    difficulty: int = Field(description="Task difficulty")
    condition: str = Field(description="Condition: 'from_scratch' or 'transfer'")
    initial_reward: float = Field(description="Reward from first episode")
    final_reward: float = Field(description="Reward from last episode")
    mean_reward: float = Field(description="Mean reward across episodes")
    initial_task_success_score: float = Field(
        description="Task-success score from first episode"
    )
    final_task_success_score: float = Field(
        description="Task-success score from last episode"
    )
    mean_task_success_score: float = Field(
        description="Mean task-success score across episodes"
    )
    initial_episode_length: int = Field(
        description="Episode length from first episode"
    )
    final_episode_length: int = Field(
        description="Episode length from last episode"
    )
    mean_episode_length: float = Field(
        description="Mean episode length across episodes"
    )
    episode_results: List[TransferEpisodeResult] = Field(
        default_factory=list,
        description="All episode-level results for adaptation curve",
    )


class TransferExperimentResult(BaseModel):
    """Complete result of a task transfer experiment.

    Attributes
    ----------
    config : TransferConfig
        Configuration used for the experiment.
    training_task_ids : List[str]
        Task IDs used for training.
    adaptation_task_ids : List[str]
        Task IDs used for adaptation (unseen).
    from_scratch_results : List[TransferTaskResult]
        Results for from-scratch condition, one per adaptation task.
    transfer_results : List[TransferTaskResult]
        Results for transfer condition, one per adaptation task.
    training_q_table_state_count : int
        Number of states in Q-table after training (transfer condition).
    training_q_table_entry_count : int
        Number of entries in Q-table after training (transfer condition).
    seed : int
        Random seed used.
    notes : str
        Human-readable notes about the experiment.
    """

    config: TransferConfig = Field(description="Configuration used")
    training_task_ids: List[str] = Field(description="Training task IDs")
    adaptation_task_ids: List[str] = Field(description="Adaptation task IDs")
    from_scratch_results: List[TransferTaskResult] = Field(
        default_factory=list,
        description="From-scratch condition results",
    )
    transfer_results: List[TransferTaskResult] = Field(
        default_factory=list,
        description="Transfer condition results",
    )
    training_q_table_state_count: int = Field(
        description="Q-table states after training (transfer condition)"
    )
    training_q_table_entry_count: int = Field(
        description="Q-table entries after training (transfer condition)"
    )
    seed: int = Field(description="Random seed used")
    notes: str = Field(
        default="",
        description="Human-readable notes about the experiment",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        return self.model_dump(mode="json", exclude_none=True)

    def serialize(self) -> Dict[str, Any]:
        """Alias for to_dict."""
        return self.to_dict()


# ============================================================================
# TaskTransferExperiment
# ============================================================================


class TaskTransferExperiment:
    """Controlled experiment comparing from-scratch vs transfer adaptation.

    This experiment answers: "Does knowledge learned from previously seen
    tasks help the RL controller adapt to a new/unseen task?"

    The experiment:
    1. Trains a task-aware Q-learning policy on training tasks (transfer
       condition only).
    2. Preserves the learned Q-table.
    3. Creates a fresh policy for from-scratch condition.
    4. Runs adaptation episodes on unseen tasks for both conditions.
    5. Compares results.

    The experiment is fully deterministic when seed is fixed.
    """

    def __init__(
        self,
        distribution: MetaTaskDistribution,
        config: TransferConfig,
    ) -> None:
        """Initialize the experiment.

        Parameters
        ----------
        distribution : MetaTaskDistribution
            The task distribution containing all tasks.
        config : TransferConfig
            Experiment configuration.
        """
        self.distribution = distribution
        self.config = config
        self.config.validate(distribution)

        # RNG for any stochastic choices
        self._rng = random.Random(config.seed)

        # Get training and adaptation tasks
        self._training_tasks = self._get_training_tasks()
        self._adaptation_tasks = self._get_adaptation_tasks()

        # Shared components
        self._task_performance_evaluator = TaskPerformanceEvaluator()

    def _get_training_tasks(self) -> List[MetaTask]:
        """Get the training tasks based on configuration."""
        if self.config.training_task_ids is not None:
            # Use explicit training task IDs
            return [
                self.distribution.get_task(task_id)
                for task_id in self.config.training_task_ids
            ]
        elif self.config.training_distribution is not None:
            # Use training distribution
            return self.config.training_distribution.get_tasks()
        else:
            # No training tasks (from-scratch only)
            return []

    def _get_adaptation_tasks(self) -> List[MetaTask]:
        """Get the adaptation (unseen) tasks based on configuration."""
        return [
            self.distribution.get_task(task_id)
            for task_id in self.config.adaptation_task_ids
        ]

    def _create_policy(
        self,
        *,
        task_aware: bool = True,
        seed: Optional[int] = None,
        existing_q_table: Optional[QTable] = None,
    ) -> QLearningPolicy:
        """Create a Q-learning policy.

        Parameters
        ----------
        task_aware : bool
            Whether to use task-aware state encoding.
        seed : Optional[int]
            Random seed for the policy.
        existing_q_table : Optional[QTable]
            If provided, use this Q-table (for transfer condition).

        Returns
        -------
        QLearningPolicy
            The created policy.
        """
        if existing_q_table is not None:
            # Create policy with existing Q-table (transfer condition)
            policy = QLearningPolicy(
                alpha=self.config.learning_rate,
                gamma=self.config.discount_factor,
                epsilon=self.config.epsilon,
                epsilon_min=self.config.epsilon_min,
                epsilon_decay=self.config.epsilon_decay,
                seed=seed,
                task_aware=task_aware,
            )
            # Replace the Q-table with the existing one
            policy.q_table = existing_q_table
            return policy
        else:
            # Create fresh policy (from-scratch condition)
            return QLearningPolicy(
                alpha=self.config.learning_rate,
                gamma=self.config.discount_factor,
                epsilon=self.config.epsilon,
                epsilon_min=self.config.epsilon_min,
                epsilon_decay=self.config.epsilon_decay,
                seed=seed,
                task_aware=task_aware,
            )

    def _train_on_tasks(
        self,
        policy: QLearningPolicy,
        tasks: List[MetaTask],
    ) -> QTable:
        """Train a policy on multiple tasks.

        Parameters
        ----------
        policy : QLearningPolicy
            The policy to train.
        tasks : List[MetaTask]
            Tasks to train on.

        Returns
        -------
        QTable
            The trained Q-table.
        """
        if not tasks:
            return policy.q_table

        trainer = MetaTrainer(
            task_distribution=self.distribution,
            policy=policy,
            seed=self.config.seed,
            role_options=self.config.role_options,
            max_steps=self.config.max_steps,
        )

        # Train on all training tasks
        for task in tasks:
            trainer.train_single_task(
                task,
                num_episodes=self.config.training_episodes_per_task,
            )

        return trainer.q_table

    def _run_adaptation_episodes(
        self,
        policy: QLearningPolicy,
        task: MetaTask,
        num_episodes: int,
        condition: str,
    ) -> List[TransferEpisodeResult]:
        """Run adaptation episodes on a single task.

        Parameters
        ----------
        policy : QLearningPolicy
            The policy to use for adaptation.
        task : MetaTask
            The adaptation task.
        num_episodes : int
            Number of adaptation episodes.
        condition : str
            "from_scratch" or "transfer".

        Returns
        -------
        List[TransferEpisodeResult]
            Episode-level results.
        """
        episode_results: List[TransferEpisodeResult] = []

        for episode_idx in range(num_episodes):
            # Create fresh environment for this episode
            manager = ArchitectureManager.create_default_architecture()
            env = MetaEnvironment(
                manager,
                role_options=self.config.role_options,
                max_steps=self.config.max_steps,
                task=task,
                task_performance_weight=self.config.task_performance_weight,
            )

            # Create trainer with this policy
            trainer = QLearningTrainer(env, policy)

            # Run one episode
            stats = trainer.train_episode()

            # Get task-performance evaluation at end of episode
            architecture = env.manager.get_architecture()
            task_perf = self._task_performance_evaluator.evaluate(
                task, architecture
            )

            # Get structural evaluation at end of episode
            structural_eval = env.current_evaluation

            # Record episode result
            episode_result = TransferEpisodeResult(
                task_id=task.task_id,
                episode=episode_idx,
                condition=condition,
                total_reward=stats.total_reward,
                episode_length=stats.episode_length,
                task_success_score=task_perf.task_success_score,
                structural_score=structural_eval.overall_score,
            )
            episode_results.append(episode_result)

        return episode_results

    def run(self) -> TransferExperimentResult:
        """Run the full transfer experiment.

        Returns
        -------
        TransferExperimentResult
            Complete experiment results.
        """
        # ============================================================
        # PHASE 1: Training (transfer condition only)
        # ============================================================
        training_task_ids = [t.task_id for t in self._training_tasks]

        # Create policy for transfer condition
        transfer_policy = self._create_policy(
            task_aware=True,
            seed=self.config.seed,
        )

        # Train on training tasks
        trained_q_table = self._train_on_tasks(
            transfer_policy,
            self._training_tasks,
        )

        # Record Q-table state after training
        training_q_table_state_count = trained_q_table.num_states()
        training_q_table_entry_count = trained_q_table.num_state_action_pairs()

        # ============================================================
        # PHASE 2: Adaptation on unseen tasks
        # ============================================================
        from_scratch_results: List[TransferTaskResult] = []
        transfer_results: List[TransferTaskResult] = []

        for task in self._adaptation_tasks:
            # --------------------------------------------------------
            # From-scratch condition: fresh policy
            # --------------------------------------------------------
            from_scratch_policy = self._create_policy(
                task_aware=True,
                seed=self.config.seed,
                # No existing Q-table - starts fresh
            )

            from_scratch_episodes = self._run_adaptation_episodes(
                from_scratch_policy,
                task,
                self.config.adaptation_episodes,
                condition="from_scratch",
            )

            # Aggregate from-scratch results
            from_scratch_result = self._aggregate_task_results(
                task,
                from_scratch_episodes,
                condition="from_scratch",
            )
            from_scratch_results.append(from_scratch_result)

            # --------------------------------------------------------
            # Transfer condition: policy with learned Q-table
            # --------------------------------------------------------
            transfer_policy_copy = self._create_policy(
                task_aware=True,
                seed=self.config.seed,
                existing_q_table=trained_q_table,
            )

            transfer_episodes = self._run_adaptation_episodes(
                transfer_policy_copy,
                task,
                self.config.adaptation_episodes,
                condition="transfer",
            )

            # Aggregate transfer results
            transfer_result = self._aggregate_task_results(
                task,
                transfer_episodes,
                condition="transfer",
            )
            transfer_results.append(transfer_result)

        # ============================================================
        # Build result
        # ============================================================
        return TransferExperimentResult(
            config=self.config,
            training_task_ids=training_task_ids,
            adaptation_task_ids=[t.task_id for t in self._adaptation_tasks],
            from_scratch_results=from_scratch_results,
            transfer_results=transfer_results,
            training_q_table_state_count=training_q_table_state_count,
            training_q_table_entry_count=training_q_table_entry_count,
            seed=self.config.seed,
            notes=(
                "Task transfer and adaptation baseline experiment. "
                "Compares from-scratch vs transfer adaptation on unseen tasks. "
                "Transfer condition uses Q-table learned from training tasks. "
                "This is a baseline experiment, NOT a complete Meta-RL algorithm. "
                "Task-performance scores are deterministic compatibility proxies, "
                "NOT actual LLM task execution."
            ),
        )

    def _aggregate_task_results(
        self,
        task: MetaTask,
        episode_results: List[TransferEpisodeResult],
        condition: str,
    ) -> TransferTaskResult:
        """Aggregate episode results for one task.

        Parameters
        ----------
        task : MetaTask
            The adaptation task.
        episode_results : List[TransferEpisodeResult]
            Episode-level results.
        condition : str
            "from_scratch" or "transfer".

        Returns
        -------
        TransferTaskResult
            Aggregated task result.
        """
        if not episode_results:
            raise ValueError("episode_results must not be empty")

        # Sort by episode index to ensure correct ordering
        sorted_results = sorted(episode_results, key=lambda r: r.episode)

        initial = sorted_results[0]
        final = sorted_results[-1]

        n = len(sorted_results)
        mean_reward = sum(r.total_reward for r in sorted_results) / n
        mean_task_success = sum(r.task_success_score for r in sorted_results) / n
        mean_length = sum(r.episode_length for r in sorted_results) / n

        return TransferTaskResult(
            task_id=task.task_id,
            task_category=task.task_category,
            difficulty=task.difficulty,
            condition=condition,
            initial_reward=initial.total_reward,
            final_reward=final.total_reward,
            mean_reward=mean_reward,
            initial_task_success_score=initial.task_success_score,
            final_task_success_score=final.task_success_score,
            mean_task_success_score=mean_task_success,
            initial_episode_length=initial.episode_length,
            final_episode_length=final.episode_length,
            mean_episode_length=mean_length,
            episode_results=sorted_results,
        )

    # ============================================================================
    # Helper methods for analysis
    # ============================================================================

    def calculate_transfer_advantage(self, result: TransferExperimentResult) -> Dict[str, Any]:
        """Calculate transfer advantage metrics from experiment result.

        Parameters
        ----------
        result : TransferExperimentResult
            The experiment result.

        Returns
        -------
        Dict[str, Any]
            Dictionary with transfer advantage metrics.
        """
        metrics: Dict[str, Any] = {
            "mean_transfer_advantage": 0.0,
            "mean_task_success_advantage": 0.0,
            "per_task_advantages": [],
        }

        total_transfer_advantage = 0.0
        total_task_success_advantage = 0.0
        task_count = 0

        for scratch_result, transfer_result in zip(
            result.from_scratch_results,
            result.transfer_results,
        ):
            # Verify same task
            assert scratch_result.task_id == transfer_result.task_id

            # Transfer advantage = transfer_final - scratch_final
            reward_advantage = (
                transfer_result.final_reward - scratch_result.final_reward
            )
            task_success_advantage = (
                transfer_result.final_task_success_score
                - scratch_result.final_task_success_score
            )

            metrics["per_task_advantages"].append(
                {
                    "task_id": scratch_result.task_id,
                    "task_category": scratch_result.task_category,
                    "transfer_reward_advantage": reward_advantage,
                    "transfer_task_success_advantage": task_success_advantage,
                }
            )

            total_transfer_advantage += reward_advantage
            total_task_success_advantage += task_success_advantage
            task_count += 1

        if task_count > 0:
            metrics["mean_transfer_advantage"] = (
                total_transfer_advantage / task_count
            )
            metrics["mean_task_success_advantage"] = (
                total_task_success_advantage / task_count
            )

        return metrics
