"""
Deterministic meta-RL evaluation framework for the adaptive MAS research project.

This module provides MetaRLEvaluator, a deterministic evaluation framework that
compares existing RL approaches and establishes a research baseline for the
proposed adaptive MAS architecture.

The comparison includes:
1. Architecture-only Q-learning baseline (Step 12/17)
2. Task-aware Q-learning (Step 17)
3. Multi-task Q-learning trainer (Step 18)

This module does NOT implement:
- New RL algorithms (PPO, DQN, etc.)
- Neural networks or gradient-based Meta-RL
- MAML, Reptile, or learned task embeddings
- Dynamic LangGraph rebuilding
- LLM calls or API interactions
- Persistent memory

Design notes
------------
* The evaluator is fully offline and deterministic when seeds are fixed.
* All three approaches are evaluated under identical experimental conditions.
* Results are represented with Pydantic models for serialization.
* Improvement metrics handle zero-denominator cases explicitly.
* This is an evaluation/baseline step, NOT a complete Meta-RL algorithm.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.architecture.manager import ArchitectureManager
from app.rl.environment import MASArchitectureEnv
from app.rl.meta_environment import MetaEnvironment
from app.rl.meta_task import MetaTask, MetaTaskContext
from app.rl.q_learning import (
    QLearningPolicy,
    QLearningTrainer,
    StateEncoder,
    TaskAwareStateEncoder,
)
from app.rl.task_distribution import MetaTaskDistribution
from app.rl.meta_trainer import MetaTrainer


# ============================================================================
# Result representations
# ============================================================================


class PerTaskEvaluationResult(BaseModel):
    """Per-task evaluation results for a single approach."""

    task_id: str = Field(description="MetaTask.task_id")
    task_category: str = Field(description="MetaTask.task_category")
    difficulty: int = Field(description="MetaTask.difficulty")
    episode_count: int = Field(description="Number of episodes evaluated")
    mean_reward: float = Field(description="Mean episode total reward")
    total_reward: float = Field(description="Sum of episode total rewards")
    mean_episode_length: float = Field(description="Mean episode length in steps")
    final_architecture_version: int = Field(
        description="Architecture version at end of evaluation"
    )
    valid_transition_count: int = Field(
        default=0, description="Number of valid transitions if available"
    )
    invalid_transition_count: int = Field(
        default=0, description="Number of invalid transitions if available"
    )
    task_context_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task context summary for reproducibility",
    )


class AggregateEvaluationResult(BaseModel):
    """Aggregate evaluation results across all tasks for one approach."""

    approach_name: str = Field(description="Name of the evaluated approach")
    num_tasks: int = Field(description="Number of tasks evaluated")
    num_episodes: int = Field(description="Total number of episodes")
    mean_reward: float = Field(description="Mean episode total reward")
    total_reward: float = Field(description="Sum of episode total rewards")
    mean_episode_length: float = Field(description="Mean episode length")
    reward_standard_deviation: Optional[float] = Field(
        default=None,
        description="Standard deviation of episode rewards (when computable)",
    )
    mean_final_architecture_version: float = Field(
        description="Mean final architecture version across tasks"
    )
    train_task_mean_reward: Optional[float] = Field(
        default=None,
        description="Mean reward on train tasks (when train/test split used)",
    )
    test_task_mean_reward: Optional[float] = Field(
        default=None,
        description="Mean reward on test tasks (when train/test split used)",
    )
    per_task_results: List[PerTaskEvaluationResult] = Field(
        default_factory=list,
        description="Per-task evaluation details",
    )


class ComparisonResult(BaseModel):
    """Comparison results between approaches."""

    architecture_only: AggregateEvaluationResult = Field(
        description="Architecture-only baseline results"
    )
    task_aware: AggregateEvaluationResult = Field(
        description="Task-aware Q-learning results"
    )
    multi_task: Optional[AggregateEvaluationResult] = Field(
        default=None,
        description="Multi-task trainer results (if evaluated)",
    )

    # Improvement: task-aware vs architecture-only
    task_aware_absolute_improvement: Optional[float] = Field(
        default=None,
        description="task_aware.mean_reward - architecture_only.mean_reward",
    )
    task_aware_relative_improvement: Optional[float] = Field(
        default=None,
        description="(task_aware - arch_only) / abs(arch_only), or None if baseline is zero",
    )

    # Improvement: multi-task vs architecture-only (if available)
    multi_task_absolute_improvement: Optional[float] = Field(
        default=None,
        description="multi_task.mean_reward - architecture_only.mean_reward",
    )
    multi_task_relative_improvement: Optional[float] = Field(
        default=None,
        description="(multi_task - arch_only) / abs(arch_only), or None if baseline is zero",
    )

    # Improvement: multi-task vs task-aware (if available)
    multi_task_vs_task_aware_absolute: Optional[float] = Field(
        default=None,
        description="multi_task.mean_reward - task_aware.mean_reward",
    )
    multi_task_vs_task_aware_relative: Optional[float] = Field(
        default=None,
        description="(multi_task - task_aware) / abs(task_aware), or None if baseline is zero",
    )


class EvaluationConfig(BaseModel):
    """Configuration for MetaRLEvaluator runs."""

    num_tasks: int = Field(
        default=5,
        ge=1,
        description="Number of tasks to evaluate (from distribution)",
    )
    episodes_per_task: int = Field(
        default=3,
        ge=1,
        description="Number of episodes per task",
    )
    max_steps_per_episode: int = Field(
        default=10,
        ge=1,
        description="Maximum steps per episode",
    )
    seed: int = Field(
        default=42,
        description="Random seed for deterministic evaluation",
    )
    role_options: Optional[List[str]] = Field(
        default=None,
        description="Optional role vocabulary for environments",
    )
    train_ratio: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=0.99,
        description="If set, split distribution into train/test",
    )

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.num_tasks < 1:
            raise ValueError("num_tasks must be >= 1")
        if self.episodes_per_task < 1:
            raise ValueError("episodes_per_task must be >= 1")
        if self.max_steps_per_episode < 1:
            raise ValueError("max_steps_per_episode must be >= 1")


class EvaluationResult(BaseModel):
    """Complete evaluation result containing all approaches and metadata."""

    config: EvaluationConfig = Field(description="Evaluation configuration used")
    comparison: ComparisonResult = Field(description="Comparison between approaches")
    timestamp: str = Field(
        default="",
        description="Optional timestamp of evaluation run",
    )
    notes: str = Field(
        default="",
        description="Optional human-readable notes about the evaluation",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "config": self.config.model_dump(),
            "comparison": {
                "architecture_only": self.comparison.architecture_only.model_dump(),
                "task_aware": self.comparison.task_aware.model_dump(),
                "multi_task": (
                    self.comparison.multi_task.model_dump()
                    if self.comparison.multi_task is not None
                    else None
                ),
                "task_aware_absolute_improvement": (
                    self.comparison.task_aware_absolute_improvement
                ),
                "task_aware_relative_improvement": (
                    self.comparison.task_aware_relative_improvement
                ),
                "multi_task_absolute_improvement": (
                    self.comparison.multi_task_absolute_improvement
                ),
                "multi_task_relative_improvement": (
                    self.comparison.multi_task_relative_improvement
                ),
                "multi_task_vs_task_aware_absolute": (
                    self.comparison.multi_task_vs_task_aware_absolute
                ),
                "multi_task_vs_task_aware_relative": (
                    self.comparison.multi_task_vs_task_aware_relative
                ),
            },
            "notes": self.notes,
        }

    def to_json_string(self) -> str:
        """Serialize to JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=2, default=str)


# ============================================================================
# MetaRLEvaluator
# ============================================================================


class MetaRLEvaluator:
    """
    Deterministic evaluator for comparing RL approaches on a MetaTaskDistribution.

    This evaluator evaluates three approaches:
    1. Architecture-only Q-learning (no task context)
    2. Task-aware Q-learning (with task context)
    3. Multi-task Q-learning trainer (shared Q-table across tasks)

    All approaches are evaluated under identical conditions:
    - Same task distribution
    - Same task selection
    - Same initial architecture
    - Same max episode steps
    - Same reward definition
    - Same evaluation configuration
    - Same random seed where applicable

    The evaluator is designed for reproducibility:
    - Uses fixed random seed
    - Does not modify the original task distribution
    - Supports train/test task separation
    - Can be run repeatedly without side effects
    """

    def __init__(
        self,
        task_distribution: MetaTaskDistribution,
        *,
        config: Optional[EvaluationConfig] = None,
    ) -> None:
        """
        Initialize the evaluator.

        Parameters
        ----------
        task_distribution : MetaTaskDistribution
            The task distribution to evaluate on.
        config : EvaluationConfig, optional
            Evaluation configuration. If None, uses defaults.
        """
        if task_distribution.task_count() < 1:
            raise ValueError(
                "MetaRLEvaluator requires a task distribution with at least one task."
            )

        self.task_distribution = task_distribution
        self.config = config or EvaluationConfig()
        self.config.validate()

        # Create a dedicated RNG for task sampling (does not affect policies)
        self._task_rng = random.Random(self.config.seed)

    def _sample_tasks(
        self,
        n: Optional[int] = None,
        train_test_split: bool = False,
    ) -> Tuple[List[MetaTask], Optional[List[MetaTask]], Optional[List[MetaTask]]]:
        """
        Sample tasks from the distribution.

        Parameters
        ----------
        n : int, optional
            Number of tasks to sample. If None, uses config.num_tasks.
        train_test_split : bool
            If True and config.train_ratio is set, perform train/test split.

        Returns
        -------
        Tuple of (train_tasks, test_tasks_or_None, all_sampled_tasks)
        """
        n = n or self.config.num_tasks

        if train_test_split and self.config.train_ratio is not None:
            # Perform train/test split on the distribution
            train_dist, test_dist = self.task_distribution.split(
                train_ratio=self.config.train_ratio,
                seed=self.config.seed,
            )
            # Sample from train distribution
            train_tasks = train_dist.sample(
                n=min(n, train_dist.task_count()),
                seed=self.config.seed,
            )
            test_tasks = test_dist.sample(
                n=min(max(1, n // 2), test_dist.task_count()),
                seed=self.config.seed + 1,
            )
            return list(train_tasks), list(test_tasks), list(train_tasks)
        else:
            # Sample tasks from the full distribution
            sampled = self.task_distribution.sample(
                n=min(n, self.task_distribution.task_count()),
                seed=self.config.seed,
            )
            return list(sampled), None, list(sampled)

    def _create_architecture_only_policy(self) -> QLearningPolicy:
        """Create an architecture-only Q-learning policy (no task context)."""
        return QLearningPolicy(
            alpha=0.1,
            gamma=0.9,
            epsilon=0.5,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            seed=self.config.seed,
            task_aware=False,  # Architecture-only: no task context
        )

    def _create_task_aware_policy(self) -> QLearningPolicy:
        """Create a task-aware Q-learning policy (with task context)."""
        return QLearningPolicy(
            alpha=0.1,
            gamma=0.9,
            epsilon=0.5,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            seed=self.config.seed,
            task_aware=True,  # Task-aware: includes task context
        )

    def _evaluate_architecture_only(
        self,
        tasks: List[MetaTask],
    ) -> AggregateEvaluationResult:
        """
        Evaluate the architecture-only Q-learning baseline.

        IMPORTANT: This policy does NOT receive task context.
        The state encoder is architecture-only (StateEncoder, not TaskAwareStateEncoder).
        """
        policy = self._create_architecture_only_policy()
        per_task_results: List[PerTaskEvaluationResult] = []
        all_rewards: List[float] = []
        total_valid = 0
        total_invalid = 0

        for task in tasks:
            # Create a plain MASArchitectureEnv (no task context)
            manager = ArchitectureManager.create_default_architecture()
            env = MASArchitectureEnv(
                manager,
                role_options=self.config.role_options,
                max_steps=self.config.max_steps_per_episode,
            )

            trainer = QLearningTrainer(env, policy)
            task_rewards: List[float] = []
            task_lengths: List[int] = []
            valid_count = 0
            invalid_count = 0

            for episode_idx in range(self.config.episodes_per_task):
                # Reset environment
                observation, info = env.reset()

                # Verify observation does NOT contain task_context
                # This ensures the architecture-only baseline is not accidentally
                # receiving task context
                if "task_context" in observation:
                    raise RuntimeError(
                        "Architecture-only evaluation received task_context in observation. "
                        "This indicates a bug in the evaluation setup."
                    )

                # Run episode
                state_key = policy.get_state_key(observation)
                valid_actions = list(range(env.mapper.action_count))

                episode_reward = 0.0
                episode_steps = 0
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    # Select action (no task_context passed - architecture-only)
                    action_id = policy.select_action(
                        observation=observation,
                        valid_action_ids=valid_actions,
                        task_context=None,  # Explicitly None for architecture-only
                    )

                    # Execute step
                    next_observation, reward, terminated, truncated, step_info = (
                        env.step(action_id)
                    )

                    # Track valid/invalid transitions
                    transition = step_info.get("transition", {})
                    if transition.get("valid", False):
                        valid_count += 1
                    else:
                        invalid_count += 1

                    # Update Q-table
                    next_state_key = policy.get_state_key(next_observation)
                    next_valid_actions = list(range(env.mapper.action_count))

                    policy.update(
                        state_key=state_key,
                        action_id=action_id,
                        reward=reward,
                        next_state_key=next_state_key,
                        next_valid_actions=next_valid_actions,
                        terminated=terminated,
                        truncated=truncated,
                    )

                    episode_reward += reward
                    episode_steps += 1

                    # Move to next state
                    observation = next_observation
                    state_key = next_state_key
                    valid_actions = next_valid_actions

                task_rewards.append(episode_reward)
                task_lengths.append(episode_steps)

                # Decay epsilon after each episode
                policy.reset_for_episode()

            # Record per-task results
            mean_reward = sum(task_rewards) / len(task_rewards)
            all_rewards.extend(task_rewards)

            # Get final architecture version
            try:
                final_version = env.current_evaluation.architecture_version or 0
            except (RuntimeError, AttributeError):
                final_version = 0

            per_task_results.append(
                PerTaskEvaluationResult(
                    task_id=task.task_id,
                    task_category=task.task_category,
                    difficulty=task.difficulty,
                    episode_count=self.config.episodes_per_task,
                    mean_reward=mean_reward,
                    total_reward=sum(task_rewards),
                    mean_episode_length=sum(task_lengths) / len(task_lengths),
                    final_architecture_version=final_version,
                    valid_transition_count=valid_count,
                    invalid_transition_count=invalid_count,
                    task_context_summary={
                        "task_category": task.task_category,
                        "difficulty": task.difficulty,
                        "note": "architecture_only_baseline - no task context provided",
                    },
                )
            )

            total_valid += valid_count
            total_invalid += invalid_count

        # Compute aggregate metrics
        num_episodes = len(all_rewards)
        mean_reward = sum(all_rewards) / num_episodes if num_episodes > 0 else 0.0
        total_reward = sum(all_rewards)

        # Standard deviation (population)
        if num_episodes > 1:
            variance = sum((r - mean_reward) ** 2 for r in all_rewards) / num_episodes
            std_dev = variance ** 0.5
        else:
            std_dev = None

        mean_final_version = (
            sum(r.final_architecture_version for r in per_task_results)
            / len(per_task_results)
            if per_task_results
            else 0.0
        )

        return AggregateEvaluationResult(
            approach_name="architecture_only",
            num_tasks=len(tasks),
            num_episodes=num_episodes,
            mean_reward=mean_reward,
            total_reward=total_reward,
            mean_episode_length=sum(task_lengths) / len(task_lengths) if task_lengths else 0.0,
            reward_standard_deviation=std_dev,
            mean_final_architecture_version=mean_final_version,
            per_task_results=per_task_results,
        )

    def _evaluate_task_aware(
        self,
        tasks: List[MetaTask],
    ) -> AggregateEvaluationResult:
        """
        Evaluate the task-aware Q-learning approach.

        This policy receives task context through the TaskAwareStateEncoder.
        """
        policy = self._create_task_aware_policy()
        per_task_results: List[PerTaskEvaluationResult] = []
        all_rewards: List[float] = []
        total_valid = 0
        total_invalid = 0

        for task in tasks:
            # Create a MetaEnvironment with task context
            manager = ArchitectureManager.create_default_architecture()
            env = MetaEnvironment(
                manager,
                role_options=self.config.role_options,
                max_steps=self.config.max_steps_per_episode,
                task=task,
            )

            trainer = QLearningTrainer(env, policy)
            task_rewards: List[float] = []
            task_lengths: List[int] = []
            valid_count = 0
            invalid_count = 0

            task_context = env.task_context

            for episode_idx in range(self.config.episodes_per_task):
                # Reset environment
                observation, info = env.reset()

                # Verify observation DOES contain task_context
                if "task_context" not in observation:
                    raise RuntimeError(
                        "Task-aware evaluation did not receive task_context in observation."
                    )

                # Run episode
                state_key = policy.get_state_key(
                    observation, task_context=task_context
                )
                valid_actions = list(range(env.mapper.action_count))

                episode_reward = 0.0
                episode_steps = 0
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    # Select action WITH task_context (task-aware)
                    action_id = policy.select_action(
                        observation=observation,
                        valid_action_ids=valid_actions,
                        task_context=task_context,
                    )

                    # Execute step
                    next_observation, reward, terminated, truncated, step_info = (
                        env.step(action_id)
                    )

                    # Track valid/invalid transitions
                    transition = step_info.get("transition", {})
                    if transition.get("valid", False):
                        valid_count += 1
                    else:
                        invalid_count += 1

                    # Update Q-table
                    next_state_key = policy.get_state_key(
                        next_observation, task_context=task_context
                    )
                    next_valid_actions = list(range(env.mapper.action_count))

                    policy.update(
                        state_key=state_key,
                        action_id=action_id,
                        reward=reward,
                        next_state_key=next_state_key,
                        next_valid_actions=next_valid_actions,
                        terminated=terminated,
                        truncated=truncated,
                    )

                    episode_reward += reward
                    episode_steps += 1

                    # Move to next state
                    observation = next_observation
                    state_key = next_state_key
                    valid_actions = next_valid_actions

                task_rewards.append(episode_reward)
                task_lengths.append(episode_steps)

                # Decay epsilon after each episode
                policy.reset_for_episode()

            # Record per-task results
            mean_reward = sum(task_rewards) / len(task_rewards)
            all_rewards.extend(task_rewards)

            # Get final architecture version
            try:
                final_version = env.current_evaluation.architecture_version or 0
            except (RuntimeError, AttributeError):
                final_version = 0

            per_task_results.append(
                PerTaskEvaluationResult(
                    task_id=task.task_id,
                    task_category=task.task_category,
                    difficulty=task.difficulty,
                    episode_count=self.config.episodes_per_task,
                    mean_reward=mean_reward,
                    total_reward=sum(task_rewards),
                    mean_episode_length=sum(task_lengths) / len(task_lengths),
                    final_architecture_version=final_version,
                    valid_transition_count=valid_count,
                    invalid_transition_count=invalid_count,
                    task_context_summary=task_context.to_dict()
                    if task_context
                    else {},
                )
            )

            total_valid += valid_count
            total_invalid += invalid_count

        # Compute aggregate metrics
        num_episodes = len(all_rewards)
        mean_reward = sum(all_rewards) / num_episodes if num_episodes > 0 else 0.0
        total_reward = sum(all_rewards)

        # Standard deviation (population)
        if num_episodes > 1:
            variance = sum((r - mean_reward) ** 2 for r in all_rewards) / num_episodes
            std_dev = variance ** 0.5
        else:
            std_dev = None

        mean_final_version = (
            sum(r.final_architecture_version for r in per_task_results)
            / len(per_task_results)
            if per_task_results
            else 0.0
        )

        return AggregateEvaluationResult(
            approach_name="task_aware",
            num_tasks=len(tasks),
            num_episodes=num_episodes,
            mean_reward=mean_reward,
            total_reward=total_reward,
            mean_episode_length=sum(task_lengths) / len(task_lengths) if task_lengths else 0.0,
            reward_standard_deviation=std_dev,
            mean_final_architecture_version=mean_final_version,
            per_task_results=per_task_results,
        )

    def _evaluate_multi_task(
        self,
        tasks: List[MetaTask],
    ) -> AggregateEvaluationResult:
        """
        Evaluate the multi-task Q-learning trainer.

        This uses the existing MetaTrainer with a shared Q-table across tasks.
        The policy is task-aware (uses TaskAwareStateEncoder).
        """
        # Create MetaTrainer with task-aware policy
        trainer = MetaTrainer(
            task_distribution=self.task_distribution,
            policy=QLearningPolicy(
                alpha=0.1,
                gamma=0.9,
                epsilon=0.5,
                epsilon_min=0.01,
                epsilon_decay=0.995,
                seed=self.config.seed,
                task_aware=True,
            ),
            seed=self.config.seed,
            role_options=self.config.role_options,
            max_steps=self.config.max_steps_per_episode,
        )

        # Train on the provided tasks
        result = trainer.train_multi_task(
            tasks=tasks,
            num_episodes_per_task=self.config.episodes_per_task,
            task_sample_seed=self.config.seed,
        )

        # Convert MultiTaskTrainingResult to AggregateEvaluationResult
        per_task_results: List[PerTaskEvaluationResult] = []
        all_rewards: List[float] = []

        for task_result in result.task_results:
            # Expand per-task results to per-episode
            # (MetaTrainer aggregates by task, we need per-episode for metrics)
            for _ in range(task_result.num_episodes):
                all_rewards.append(task_result.mean_episode_reward)

            per_task_results.append(
                PerTaskEvaluationResult(
                    task_id=task_result.task_id,
                    task_category=task_result.task_category,
                    difficulty=task_result.difficulty,
                    episode_count=task_result.num_episodes,
                    mean_reward=task_result.mean_episode_reward,
                    total_reward=task_result.total_reward,
                    mean_episode_length=task_result.mean_episode_length,
                    final_architecture_version=task_result.final_architecture_version,
                    valid_transition_count=0,  # Not tracked by MetaTrainer
                    invalid_transition_count=0,  # Not tracked by MetaTrainer
                    task_context_summary=task_result.task_context_summary,
                )
            )

        num_episodes = len(all_rewards)
        mean_reward = sum(all_rewards) / num_episodes if num_episodes > 0 else 0.0
        total_reward = sum(all_rewards)

        # Standard deviation
        if num_episodes > 1:
            variance = sum((r - mean_reward) ** 2 for r in all_rewards) / num_episodes
            std_dev = variance ** 0.5
        else:
            std_dev = None

        mean_final_version = (
            sum(r.final_architecture_version for r in per_task_results)
            / len(per_task_results)
            if per_task_results
            else 0.0
        )

        return AggregateEvaluationResult(
            approach_name="multi_task",
            num_tasks=result.num_tasks_trained,
            num_episodes=num_episodes,
            mean_reward=mean_reward,
            total_reward=total_reward,
            mean_episode_length=result.mean_episode_length,
            reward_standard_deviation=std_dev,
            mean_final_architecture_version=mean_final_version,
            per_task_results=per_task_results,
        )

    def evaluate(
        self,
        *,
        tasks: Optional[List[MetaTask]] = None,
        train_test_split: bool = False,
        evaluate_multi_task: bool = True,
    ) -> EvaluationResult:
        """
        Run the full evaluation comparing all approaches.

        Parameters
        ----------
        tasks : List[MetaTask], optional
            Explicit tasks to evaluate. If None, sampled from distribution.
        train_test_split : bool
            If True and config.train_ratio is set, evaluate on train tasks and
            report test task metrics separately.
        evaluate_multi_task : bool
            Whether to evaluate the multi-task trainer approach.

        Returns
        -------
        EvaluationResult
            Complete evaluation results.
        """
        # Sample tasks
        train_tasks, test_tasks, all_tasks = self._sample_tasks(
            n=self.config.num_tasks,
            train_test_split=train_test_split,
        )

        # Determine which tasks to use for evaluation
        eval_tasks = all_tasks
        if train_test_split and test_tasks is not None:
            # Use train tasks for main evaluation
            eval_tasks = train_tasks

        if not eval_tasks:
            raise ValueError("No tasks available for evaluation")

        # Evaluate architecture-only baseline
        arch_only_result = self._evaluate_architecture_only(eval_tasks)

        # Evaluate task-aware approach
        task_aware_result = self._evaluate_task_aware(eval_tasks)

        # Evaluate multi-task trainer (if requested)
        multi_task_result = None
        if evaluate_multi_task:
            multi_task_result = self._evaluate_multi_task(eval_tasks)

        # Compute comparison metrics
        comparison = self._compute_comparison(
            arch_only=arch_only_result,
            task_aware=task_aware_result,
            multi_task=multi_task_result,
            test_tasks=test_tasks,
        )

        return EvaluationResult(
            config=self.config,
            comparison=comparison,
            notes=(
                "Experimental baseline evaluation. These metrics are for research "
                "comparison purposes and do not claim superiority of any approach "
                "unless the actual experiment demonstrates it."
            ),
        )

    def _compute_comparison(
        self,
        arch_only: AggregateEvaluationResult,
        task_aware: AggregateEvaluationResult,
        multi_task: Optional[AggregateEvaluationResult],
        test_tasks: Optional[List[MetaTask]],
    ) -> ComparisonResult:
        """Compute comparison metrics between approaches."""

        # Task-aware vs architecture-only
        task_aware_abs_improvement = (
            task_aware.mean_reward - arch_only.mean_reward
        )

        # Relative improvement: handle zero-denominator
        if abs(arch_only.mean_reward) > 1e-10:
            task_aware_rel_improvement = (
                task_aware_abs_improvement / abs(arch_only.mean_reward)
            )
        else:
            task_aware_rel_improvement = None

        # Multi-task vs architecture-only (if available)
        multi_abs_improvement = None
        multi_rel_improvement = None
        multi_vs_task_abs = None
        multi_vs_task_rel = None

        if multi_task is not None:
            multi_abs_improvement = (
                multi_task.mean_reward - arch_only.mean_reward
            )

            if abs(arch_only.mean_reward) > 1e-10:
                multi_rel_improvement = (
                    multi_abs_improvement / abs(arch_only.mean_reward)
                )

            # Multi-task vs task-aware
            if task_aware.mean_reward != 0:
                multi_vs_task_abs = (
                    multi_task.mean_reward - task_aware.mean_reward
                )
                if abs(task_aware.mean_reward) > 1e-10:
                    multi_vs_task_rel = (
                        multi_vs_task_abs / abs(task_aware.mean_reward)
                    )

        return ComparisonResult(
            architecture_only=arch_only,
            task_aware=task_aware,
            multi_task=multi_task,
            task_aware_absolute_improvement=task_aware_abs_improvement,
            task_aware_relative_improvement=task_aware_rel_improvement,
            multi_task_absolute_improvement=multi_abs_improvement,
            multi_task_relative_improvement=multi_rel_improvement,
            multi_task_vs_task_aware_absolute=multi_vs_task_abs,
            multi_task_vs_task_aware_relative=multi_vs_task_rel,
        )

    def evaluate_determinism(
        self,
        num_runs: int = 2,
    ) -> bool:
        """
        Verify that the evaluator produces deterministic results.

        Runs the evaluation multiple times with the same seed and verifies
        that results are identical.

        Parameters
        ----------
        num_runs : int
            Number of times to run evaluation for comparison.

        Returns
        -------
        bool
            True if all runs produced identical results.
        """
        results: List[EvaluationResult] = []

        for _ in range(num_runs):
            result = self.evaluate()
            results.append(result)

        # Compare all results
        baseline = results[0]
        for other in results[1:]:
            if not self._results_equal(baseline, other):
                return False

        return True

    @staticmethod
    def _results_equal(a: EvaluationResult, b: EvaluationResult) -> bool:
        """Compare two evaluation results for equality."""
        # Compare config
        if a.config != b.config:
            return False

        # Compare comparison results
        comp_a = a.comparison
        comp_b = b.comparison

        if comp_a.architecture_only != comp_b.architecture_only:
            return False
        if comp_a.task_aware != comp_b.task_aware:
            return False

        # Multi-task might be None
        if (comp_a.multi_task is None) != (comp_b.multi_task is None):
            return False
        if comp_a.multi_task is not None and comp_b.multi_task is not None:
            if comp_a.multi_task != comp_b.multi_task:
                return False

        # Compare improvement metrics
        if comp_a.task_aware_absolute_improvement != comp_b.task_aware_absolute_improvement:
            return False
        if comp_a.task_aware_relative_improvement != comp_b.task_aware_relative_improvement:
            return False

        return True

    def get_approaches(self) -> List[str]:
        """Return list of approach names that can be evaluated."""
        return ["architecture_only", "task_aware", "multi_task"]
