"""
Multi-task / meta-training foundation for the adaptive MAS research project.

This module provides a lightweight baseline multi-task training loop that
trains a single task-aware Q-learning policy across multiple MetaTasks drawn
from a MetaTaskDistribution.

This module is intentionally a BASELINE multi-task / meta-learning foundation.
It does NOT implement a complete state-of-the-art Meta-RL algorithm, and it
does NOT implement any of the following:

* PPO, DQN, or other deep RL algorithms
* Neural networks or function approximation
* Gradient-based meta-learning (MAML, Reptile, etc.)
* Learned task embeddings
* LLM reward judges
* Dynamic LangGraph rebuilding
* Persistent memory
* Actual LLM task execution

Design overview
---------------

The module separates two levels of loops explicitly:

* *Inner/task loop*: execute a single task / environment episode, allow the
  task-aware Q-learning policy to update from (state, action, reward, next_state)
  transitions, and record per-step and per-episode statistics.

* *Outer/meta-training loop*: iterate over multiple tasks drawn from a
  MetaTaskDistribution, run task-specific episodes, update the shared
  task-aware Q-table, and collect aggregate multi-task metrics.

Task context affects the state representation. The Q-learning policy configured
with ``task_aware=True`` uses ``TaskAwareStateEncoder`` from
``app.rl.q_learning``, which incorporates the ``MetaTaskContext`` into the
state key. The same underlying MAS architecture under different MetaTasks
therefore produces different state keys and may learn different Q-values.

All randomness is localized: the trainer owns a ``random.Random`` instance
for any stochastic choices (for example, shuffling task order) and never
relies on uncontrolled global randomness. Deterministic execution is achieved
by seeding that RNG and by relying on the deterministic task sampler from
``MetaTaskDistribution`` / ``MetaTaskSampler``.

No LLM or external API calls are made. This module is fully offline.

Results are represented with small Pydantic models so that per-task and
aggregate metrics are easy to inspect, serialize, and extend.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.architecture.manager import ArchitectureManager
from app.rl.environment import MASArchitectureEnv
from app.rl.meta_environment import MetaEnvironment
from app.rl.meta_task import MetaTask, MetaTaskContext
from app.rl.policy import BasePolicy
from app.rl.q_learning import (
    QLearningPolicy,
    QLearningTrainer,
    QTable,
    StateEncoder,
    TaskAwareStateEncoder,
    TrainingStats,
)
from app.rl.task_distribution import MetaTaskDistribution

# ---------------------------------------------------------------------------
# Result representations
# ---------------------------------------------------------------------------


class TaskEpisodeResult(BaseModel):
    """Result of a single task episode during multi-task training.

    This is intentionally lightweight: it records enough information to
    compare per-task performance and to reconstruct what happened in an
    episode without storing full transition logs.
    """

    task_id: str = Field(description="MetaTask.task_id for this episode.")
    episode: int = Field(description="Episode index within this task.")
    total_reward: float = Field(description="Sum of rewards over the episode.")
    episode_length: int = Field(description="Number of environment steps taken.")
    final_architecture_version: int = Field(
        description="Architecture version at the end of the episode.",
    )
    final_architecture_id: str = Field(
        description="Architecture id at the end of the episode.",
    )
    success: bool = Field(
        description="True when the episode ended with terminated=True.",
    )
    terminated: bool = Field(description="Episode reached a terminal condition.")
    truncated: bool = Field(description="Episode reached max_steps.")
    task_context_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Minimal deterministic task-context summary used for the state key.",
    )
    notes: str = Field(
        default="",
        description="Optional human-readable notes.",
    )


class TaskTrainingResult(BaseModel):
    """Aggregate per-task result after training on one MetaTask for multiple episodes.

    This aggregates the per-episode ``TaskEpisodeResult`` objects for a single
    task into a compact summary.
    """

    task_id: str = Field(description="MetaTask.task_id.")
    task_category: str = Field(description="MetaTask.task_category.")
    difficulty: int = Field(description="MetaTask.difficulty.")
    num_episodes: int = Field(description="Number of episodes run on this task.")
    total_reward: float = Field(description="Sum of episode total rewards.")
    mean_episode_reward: float = Field(description="Mean episode total reward.")
    mean_episode_length: float = Field(description="Mean episode length in steps.")
    final_architecture_version: int = Field(
        description="Architecture version at the end of the last episode.",
    )
    final_architecture_id: str = Field(
        description="Architecture id at the end of the last episode.",
    )
    task_context_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task-context summary used for state keys.",
    )
    notes: str = Field(
        default="",
        description="Optional human-readable notes.",
    )


class MultiTaskTrainingResult(BaseModel):
    """Aggregate multi-task / meta-training result.

    This contains per-task results plus simple aggregate metrics across all
    tasks that were trained.
    """

    task_results: List[TaskTrainingResult] = Field(
        default_factory=list,
        description="Per-task training results.",
    )
    total_episodes: int = Field(
        default=0,
        description="Total number of episodes run across all tasks.",
    )
    total_reward: float = Field(
        default=0.0,
        description="Sum of episode total rewards across all tasks.",
    )
    mean_reward: float = Field(
        default=0.0,
        description="Mean episode total reward across all tasks.",
    )
    mean_episode_length: float = Field(
        default=0.0,
        description="Mean episode length across all tasks.",
    )
    num_tasks_trained: int = Field(
        default=0,
        description="Number of tasks trained.",
    )
    q_table_state_count: int = Field(
        default=0,
        description="Number of unique state keys in the shared Q-table after training.",
    )
    q_table_entry_count: int = Field(
        default=0,
        description="Number of (state, action) entries in the shared Q-table.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed used for this run, if any.",
    )
    train_task_ids: Optional[List[str]] = Field(
        default=None,
        description="Task ids used for training, if train/test split was used.",
    )
    test_task_ids: Optional[List[str]] = Field(
        default=None,
        description="Task ids reserved for evaluation, if train/test split was used.",
    )
    notes: str = Field(
        default="",
        description="Optional human-readable notes.",
    )


# ---------------------------------------------------------------------------
# Helper: build a MetaEnvironment from a MetaTask without side effects
# ---------------------------------------------------------------------------


def _build_meta_environment(
    *,
    task: MetaTask,
    manager: Optional[ArchitectureManager] = None,
    role_options: Optional[List[str]] = None,
    max_steps: Optional[int] = None,
) -> MetaEnvironment:
    """Create a fresh MetaEnvironment configured for *task*.

    This helper centralizes environment construction so the rest of the
    trainer does not need to know how the baseline default architecture is
    created.
    """
    env_target_manager = manager
    if env_target_manager is None:
        env_target_manager = ArchitectureManager.create_default_architecture()
    return MetaEnvironment(
        env_target_manager,
        role_options=role_options,
        max_steps=max_steps,
        task=task,
    )


def _default_task_context_summary(task_context: Optional[MetaTaskContext]) -> Dict[str, Any]:
    """Create a compact deterministic task-context summary for logging.

    This is deliberately limited to a few scalar/string fields so it is easy
    to compare across tasks without storing the full observation dict.
    """
    if task_context is None:
        return {
            "task_category": "",
            "difficulty": 1,
            "required_capability_count": 0,
            "context_feature_count": 0,
        }
    return {
        "task_category": task_context.task_category,
        "difficulty": task_context.difficulty,
        "required_capability_count": len(task_context.required_capabilities),
        "context_feature_count": len(task_context.context_features),
    }


# ---------------------------------------------------------------------------
# MetaTrainer
# ---------------------------------------------------------------------------


class MetaTrainer:
    """Baseline multi-task trainer for a task-aware Q-learning policy.

    This trainer trains a single shared Q-table across multiple tasks drawn
    from a ``MetaTaskDistribution``. It is a baseline multi-task / meta-RL
    foundation, not a complete advanced Meta-RL algorithm.

    The trainer owns its own randomness via ``self.rng``. If ``seed`` is
    provided, task sampling order is deterministic. The underlying
    ``MetaTaskDistribution`` / ``MetaTaskSampler`` is also used with seeds
    where applicable so that task sequences are reproducible.

    The policy is expected to be a ``QLearningPolicy`` with
    ``task_aware=True``. The trainer does not enforce this by type at
    construction time, but the public training methods assume the policy
    supports an optional ``task_context`` argument in ``select_action``.
    """

    def __init__(
        self,
        *,
        task_distribution: MetaTaskDistribution,
        policy: Optional[QLearningPolicy] = None,
        seed: Optional[int] = None,
        role_options: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
        shared_q_table: Optional[QTable] = None,
    ) -> None:
        """Initialize the meta-trainer.

        Parameters
        ----------
        task_distribution:
            Distribution of MetaTasks to train across.
        policy:
            Task-aware Q-learning policy. If None, a default
            ``QLearningPolicy(task_aware=True)`` is created.
        seed:
            Optional random seed for deterministic multi-task training.
        role_options:
            Optional role vocabulary passed to each MetaEnvironment.
        max_steps:
            Optional episode step limit passed to each MetaEnvironment.
        shared_q_table:
            Optional QTable to reuse across runs. If None, a new QTable is
            created inside the policy default constructor.
        """
        if task_distribution.task_count() < 1:
            raise ValueError(
                "MetaTrainer requires a task distribution with at least one task."
            )

        self.task_distribution = task_distribution
        self.seed = seed
        self.rng = random.Random(seed)
        self.role_options = list(role_options) if role_options is not None else None
        self.max_steps = max_steps

        if policy is None:
            policy = QLearningPolicy(task_aware=True)
        self.policy = policy
        self._trainer: Optional[QLearningTrainer] = None
        self._last_result: Optional[MultiTaskTrainingResult] = None

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def q_table(self) -> QTable:
        """Return the shared Q-table used by the current policy."""
        return self.policy.q_table

    def reset(self, *, keep_seed: bool = True) -> None:
        """Reset the trainer to a fresh shared Q-table and clear prior results.

        This does not recreate the task distribution. It clears the shared
        Q-table and the last training result so a new multi-task run can be
        performed with the same configuration.
        """
        self.policy.clear_q_table()
        self._trainer = None
        self._last_result = None
        if not keep_seed:
            self.rng = random.Random()

    # ------------------------------------------------------------------
    # Inner-loop helpers
    # ------------------------------------------------------------------

    def run_task_episode(
        self,
        task: MetaTask,
        *,
        episode_index: int = 0,
        manager: Optional[ArchitectureManager] = None,
    ) -> TaskEpisodeResult:
        """Run a single episode for one task.

        This is the inner/task loop. It creates a fresh MetaEnvironment for
        the task, runs one episode using the task-aware Q-learning policy,
        updates the shared Q-table through the underlying
        ``QLearningTrainer.train_episode`` path, and returns a
        ``TaskEpisodeResult``.

        The environment is reset at the start of the episode and isolated
        from other episodes by construction.

        Parameters
        ----------
        task:
            The MetaTask to run.
        episode_index:
            Human-readable episode index for this task.
        manager:
            Optional architecture manager to initialize the environment. If
            None, the baseline default architecture is used.

        Returns
        -------
        TaskEpisodeResult
            Per-episode result.
        """
        env = _build_meta_environment(
            task=task,
            manager=manager,
            role_options=self.role_options,
            max_steps=self.max_steps,
        )
        trainer = QLearningTrainer(env, self.policy)
        self._trainer = trainer

        stats = trainer.train_episode()

        task_context = env.task_context
        try:
            current_evaluation = env.current_evaluation
            final_version = current_evaluation.architecture_version or 0
            final_architecture_id = current_evaluation.architecture_id
        except (RuntimeError, AttributeError):
            final_version = 0
            final_architecture_id = ""

        return TaskEpisodeResult(
            task_id=task.task_id,
            episode=episode_index,
            total_reward=stats.total_reward,
            episode_length=stats.episode_length,
            final_architecture_version=final_version,
            final_architecture_id=final_architecture_id,
            success=bool(stats.terminated),
            terminated=bool(stats.terminated),
            truncated=bool(stats.truncated),
            task_context_summary=_default_task_context_summary(task_context),
            notes="",
        )

    # ------------------------------------------------------------------
    # Single-task training
    # ------------------------------------------------------------------

    def train_single_task(
        self,
        task: MetaTask,
        *,
        num_episodes: int = 1,
        manager: Optional[ArchitectureManager] = None,
    ) -> TaskTrainingResult:
        """Train the shared policy on a single MetaTask for multiple episodes.

        Parameters
        ----------
        task:
            The MetaTask to train on.
        num_episodes:
            Number of episodes to run on this task.
        manager:
            Optional architecture manager to use for each episode.

        Returns
        -------
        TaskTrainingResult
            Aggregated per-task training result.
        """
        if num_episodes < 1:
            raise ValueError("num_episodes must be >= 1")

        episodes: List[TaskEpisodeResult] = []
        for episode_index in range(num_episodes):
            episode_result = self.run_task_episode(
                task,
                episode_index=episode_index,
                manager=manager,
            )
            episodes.append(episode_result)

        last = episodes[-1]
        total_reward = sum(e.total_reward for e in episodes)
        mean_reward = total_reward / num_episodes
        mean_length = sum(e.episode_length for e in episodes) / num_episodes

        task_context_summary = episodes[0].task_context_summary

        return TaskTrainingResult(
            task_id=task.task_id,
            task_category=task.task_category,
            difficulty=task.difficulty,
            num_episodes=num_episodes,
            total_reward=total_reward,
            mean_episode_reward=mean_reward,
            mean_episode_length=mean_length,
            final_architecture_version=last.final_architecture_version,
            final_architecture_id=last.final_architecture_id,
            task_context_summary=task_context_summary,
            notes="",
        )

    # ------------------------------------------------------------------
    # Multi-task training
    # ------------------------------------------------------------------

    def train_multi_task(
        self,
        *,
        tasks: Optional[List[MetaTask]] = None,
        num_episodes_per_task: int = 1,
        task_sample_seed: Optional[int] = None,
        manager: Optional[ArchitectureManager] = None,
        train_distribution: Optional[MetaTaskDistribution] = None,
        test_distribution: Optional[MetaTaskDistribution] = None,
    ) -> MultiTaskTrainingResult:
        """Train the shared policy across multiple tasks.

        This is the outer/meta-training loop. It runs
        ``num_episodes_per_task`` episodes for each task, updates the shared
        Q-table, and returns aggregate metrics.

        If *tasks* is provided, those tasks are used directly. Otherwise tasks
        are sampled from the trainer's task distribution using the provided
        seed (or the trainer seed) so the task sequence is reproducible. If a
        train/test split is provided, training is performed on the train split
        and metrics are reported separately for the train and test splits.

        Parameters
        ----------
        tasks:
            Explicit task list to train. If None, sampled from the
            distribution.
        num_episodes_per_task:
            Episodes to run per task.
        task_sample_seed:
            Optional seed for task sampling when *tasks* is None.
        manager:
            Optional architecture manager shared across episodes.
        train_distribution:
            Optional train split used to report train vs test metrics.
        test_distribution:
            Optional test split used to report train vs test metrics.

        Returns
        -------
        MultiTaskTrainingResult
            Aggregate multi-task result.
        """
        if num_episodes_per_task < 1:
            raise ValueError("num_episodes_per_task must be >= 1")

        if tasks is not None:
            if len(tasks) < 1:
                raise ValueError("explicit task list must contain at least one task")
            task_sequence = list(tasks)
        else:
            sample_seed = task_sample_seed if task_sample_seed is not None else self.seed
            sampled = self.task_distribution.sample(
                n=self.task_distribution.task_count(),
                seed=sample_seed,
            )
            task_sequence = list(sampled)

        results: List[TaskTrainingResult] = []
        for task in task_sequence:
            task_result = self.train_single_task(
                task,
                num_episodes=num_episodes_per_task,
                manager=manager,
            )
            results.append(task_result)

        aggregate = self._aggregate(results)
        aggregate.train_task_ids = self._task_ids_from_distribution(train_distribution)
        aggregate.test_task_ids = self._task_ids_from_distribution(test_distribution)
        self._last_result = aggregate
        return aggregate

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def _aggregate(self, task_results: List[TaskTrainingResult]) -> MultiTaskTrainingResult:
        """Aggregate per-task results into a MultiTaskTrainingResult."""
        total_episodes = sum(r.num_episodes for r in task_results)
        total_reward = sum(r.total_reward for r in task_results)
        mean_reward = total_reward / total_episodes if total_episodes > 0 else 0.0
        mean_episode_length = (
            sum(r.mean_episode_length * r.num_episodes for r in task_results) / total_episodes
            if total_episodes > 0
            else 0.0
        )

        return MultiTaskTrainingResult(
            task_results=task_results,
            total_episodes=total_episodes,
            total_reward=total_reward,
            mean_reward=mean_reward,
            mean_episode_length=mean_episode_length,
            num_tasks_trained=len(task_results),
            q_table_state_count=self.q_table.num_states(),
            q_table_entry_count=self.q_table.num_state_action_pairs(),
            seed=self.seed,
            notes="",
        )

    @staticmethod
    def _task_ids_from_distribution(
        distribution: Optional[MetaTaskDistribution],
    ) -> Optional[List[str]]:
        """Extract task ids from a distribution, if any."""
        if distribution is None:
            return None
        return list(distribution.task_ids())

    # ------------------------------------------------------------------
    # Train/test reporting helpers
    # ------------------------------------------------------------------

    def train_test_split_result(
        self,
        *,
        train_distribution: MetaTaskDistribution,
        test_distribution: MetaTaskDistribution,
        num_episodes_per_task: int = 1,
        task_sample_seed: Optional[int] = None,
        manager: Optional[ArchitectureManager] = None,
    ) -> MultiTaskTrainingResult:
        """Train on a train split and report train vs test task ids.

        This is a convenience wrapper around ``train_multi_task`` that makes
        the train/test split explicit in the result object. It does not train
        on the test tasks; it only records them for later evaluation.
        """
        return self.train_multi_task(
            tasks=train_distribution.get_tasks(),
            num_episodes_per_task=num_episodes_per_task,
            task_sample_seed=task_sample_seed,
            manager=manager,
            train_distribution=train_distribution,
            test_distribution=test_distribution,
        )

    # ------------------------------------------------------------------
    # Determinism helpers
    # ------------------------------------------------------------------

    def sample_task_sequence(
        self,
        *,
        n: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[MetaTask]:
        """Sample a deterministic task sequence from the distribution.

        This uses the distribution sampler so the sampled sequence is
        reproducible when the same seed is used.
        """
        use_seed = seed if seed is not None else self.seed
        count = n if n is not None else self.task_distribution.task_count()
        sampled = self.task_distribution.sample(n=count, seed=use_seed)
        return list(sampled)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def last_result(self) -> Optional[MultiTaskTrainingResult]:
        """Return the last completed multi-task training result, if any."""
        return self._last_result
