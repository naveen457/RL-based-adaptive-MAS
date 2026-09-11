"""
Multi-task RL experiment / controller layer for the adaptive MAS research project.

This module provides the MetaController: an algorithm-agnostic orchestrator that
runs multiple task episodes using the existing MetaTask / MetaEnvironment /
BasePolicy / RLController stack.

The MetaController is the foundation required before implementing actual Meta-RL
learning. It does NOT implement:

* PPO, DQN, neural networks, or policy gradients.
* Meta-RL training loops (inner-loop / outer-loop).
* Learned task embeddings.
* Persistent memory.
* Dynamic LangGraph rebuilding.
* LLM calls or API interactions.
* Actual task-success evaluation.

Design notes
------------

* The controller is algorithm-agnostic: it accepts any ``BasePolicy`` and any
  ``MetaTaskDistribution`` and executes episodes.
* Task isolation is enforced by creating a fresh ``MetaEnvironment`` (with a
  fresh ``ArchitectureManager`` copied from the initial manager) for every task.
* Determinism is supported via an optional seed that controls task selection
  order from the distribution.
* Result representations use Pydantic models consistent with the rest of the
  project.
* Serialization is safe: no API keys, passwords, OpenRouter credentials,
  LangSmith credentials, or environment secrets are included.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.architecture.manager import ArchitectureManager
from app.rl.controller import RLController
from app.rl.environment import MASArchitectureEnv
from app.rl.meta_environment import MetaEnvironment
from app.rl.meta_task import MetaTask, MetaTaskContext
from app.rl.policy import BasePolicy
from app.rl.trajectory import Trajectory
from app.evaluation.evaluator import ArchitectureEvaluator, EvaluationResult


# ---------------------------------------------------------------------------
# Result representations
# ---------------------------------------------------------------------------

class TaskEpisodeResult(BaseModel):
    """Result of executing one task episode.

    Parameters
    ----------
    task_id:
        The ID of the task that was executed.
    task_category:
        The category of the task.
    difficulty:
        The difficulty rating of the task (1-5).
    trajectory:
        The full trajectory recorded during the episode.
    total_reward:
        Sum of rewards across all transitions in the trajectory.
    episode_length:
        Number of transitions (steps) in the episode.
    final_architecture_id:
        Architecture ID at the end of the episode.
    final_architecture_version:
        Architecture version at the end of the episode.
    final_evaluation:
        Evaluation result of the final architecture.
    terminated:
        Whether the episode reached a terminal condition.
    truncated:
        Whether the episode was truncated (e.g., max steps reached).
    task_context:
        The task context features used during execution.
    """

    task_id: str = Field(description="The ID of the task that was executed.")
    task_category: str = Field(description="The category of the task.")
    difficulty: int = Field(description="The difficulty rating of the task (1-5).")
    trajectory: Trajectory = Field(description="The full trajectory recorded during the episode.")
    total_reward: float = Field(description="Sum of rewards across all transitions.")
    episode_length: int = Field(description="Number of transitions (steps) in the episode.")
    final_architecture_id: str = Field(description="Architecture ID at the end of the episode.")
    final_architecture_version: Optional[int] = Field(
        default=None,
        description="Architecture version at the end of the episode.",
    )
    final_evaluation: EvaluationResult = Field(
        description="Evaluation result of the final architecture.",
    )
    terminated: bool = Field(description="Whether the episode reached a terminal condition.")
    truncated: bool = Field(description="Whether the episode was truncated.")
    task_context: MetaTaskContext = Field(
        description="The task context features used during execution.",
    )

    def serialize(self) -> Dict[str, Any]:
        """Return a safe JSON-friendly serialization.

        This excludes any fields that could contain secrets. The result
        representation is built from architecture-level and task-level data
        only.
        """
        return {
            "task_id": self.task_id,
            "task_category": self.task_category,
            "difficulty": self.difficulty,
            "trajectory": self.trajectory.to_dict(),
            "total_reward": self.total_reward,
            "episode_length": self.episode_length,
            "final_architecture_id": self.final_architecture_id,
            "final_architecture_version": self.final_architecture_version,
            "final_evaluation": self.final_evaluation.model_dump(
                mode="json",
                exclude_none=True,
            ),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "task_context": self.task_context.serialize(),
        }


class MultiTaskExperimentResult(BaseModel):
    """Result of executing a multi-task experiment.

    Parameters
    ----------
    task_results:
        Ordered list of task episode results.
    task_ids:
        Ordered list of task IDs that were executed.
    num_tasks:
        Number of tasks executed.
    total_reward:
        Sum of total rewards across all task episodes.
    average_reward:
        Mean total reward per task episode.
    average_episode_length:
        Mean episode length across all task episodes.
    seed:
        The seed used for task selection, if any.
    distribution_name:
        Name of the task distribution used.
    """

    task_results: List[TaskEpisodeResult] = Field(
        description="Ordered list of task episode results.",
    )
    task_ids: List[str] = Field(
        description="Ordered list of task IDs that were executed.",
    )
    num_tasks: int = Field(
        description="Number of tasks executed.",
    )
    total_reward: float = Field(
        description="Sum of total rewards across all task episodes.",
    )
    average_reward: float = Field(
        description="Mean total reward per task episode.",
    )
    average_episode_length: float = Field(
        description="Mean episode length across all task episodes.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="The seed used for task selection, if any.",
    )
    distribution_name: str = Field(
        default="",
        description="Name of the task distribution used.",
    )

    def serialize(self) -> Dict[str, Any]:
        """Return a safe JSON-friendly serialization.

        This excludes any fields that could contain secrets.
        """
        return {
            "task_results": [r.serialize() for r in self.task_results],
            "task_ids": list(self.task_ids),
            "num_tasks": self.num_tasks,
            "total_reward": self.total_reward,
            "average_reward": self.average_reward,
            "average_episode_length": self.average_episode_length,
            "seed": self.seed,
            "distribution_name": self.distribution_name,
        }


# ---------------------------------------------------------------------------
# MetaController
# ---------------------------------------------------------------------------

class MetaController:
    """Algorithm-agnostic multi-task RL experiment controller.

    This controller orchestrates the execution of one or more task episodes
    using the existing MetaTask / MetaEnvironment / BasePolicy / RLController
    stack. It is the foundation required before implementing actual Meta-RL
    learning.

    The controller:

    * Does NOT implement PPO, DQN, neural networks, or policy gradients.
    * Does NOT implement Meta-RL training loops.
    * Does NOT make LLM calls or API interactions.
    * Enforces task isolation: every task begins from a fresh initial
      architecture/environment state.
    * Supports deterministic task selection via an optional seed.

    Parameters
    ----------
    initial_manager:
        The initial architecture manager. A copy of this manager is used to
        create a fresh MetaEnvironment for each task, ensuring task isolation.
    policy:
        A policy implementing BasePolicy. This policy is used for all task
        episodes.
    role_options:
        Optional finite role vocabulary passed to each MetaEnvironment.
    max_steps:
        Optional maximum number of architecture transitions per episode.
        If None, the environment default (50) is used.
    seed:
        Optional random seed for deterministic task selection from the
        distribution.
    """

    def __init__(
        self,
        initial_manager: ArchitectureManager,
        policy: BasePolicy,
        *,
        role_options: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if initial_manager is None:
            raise ValueError("initial_manager must not be None")
        if policy is None:
            raise ValueError("policy must not be None")
        if not isinstance(policy, BasePolicy):
            raise TypeError(
                f"policy must be a BasePolicy instance, got {type(policy).__name__}"
            )

        self._initial_manager = initial_manager
        self._policy = policy
        self._role_options = list(role_options) if role_options is not None else None
        self._max_steps = max_steps
        self._seed = seed

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def initial_manager(self) -> ArchitectureManager:
        """The initial architecture manager used to create per-task environments."""
        return self._initial_manager

    @property
    def policy(self) -> BasePolicy:
        """The policy used for all task episodes."""
        return self._policy

    @property
    def role_options(self) -> Optional[List[str]]:
        """Optional role vocabulary passed to environments."""
        return list(self._role_options) if self._role_options is not None else None

    @property
    def max_steps(self) -> Optional[int]:
        """Optional max steps per episode."""
        return self._max_steps

    @property
    def seed(self) -> Optional[int]:
        """Optional seed for deterministic task selection."""
        return self._seed

    # ------------------------------------------------------------------
    # Single-task execution
    # ------------------------------------------------------------------

    def run_task(
        self,
        task: MetaTask,
        *,
        environment: Optional[MetaEnvironment] = None,
    ) -> TaskEpisodeResult:
        """Execute one task episode and return the result.

        Given a MetaTask, this method:

        * Creates/resets the corresponding MetaEnvironment (unless an existing
          environment is provided).
        * Executes an episode using the existing policy interface.
        * Records the trajectory, total reward, episode length, final
          architecture, final evaluation, and termination/truncation
          information.

        Parameters
        ----------
        task:
            The MetaTask to execute.
        environment:
            Optional pre-configured MetaEnvironment. If provided, this
            environment is used directly (the caller is responsible for
            configuring it with the correct task). If None, a fresh
            MetaEnvironment is created from the task.

        Returns
        -------
        TaskEpisodeResult:
            The result of the task episode.

        Raises
        ------
        ValueError:
            If the task is invalid or the policy is invalid.
        RuntimeError:
            If the environment is already terminated/truncated.
        """
        if task is None:
            raise ValueError("task must not be None")

        task_context = task.to_context()

        if environment is None:
            manager = ArchitectureManager(self._initial_manager.to_architecture_model())
            env = MetaEnvironment(
                manager,
                role_options=self._role_options,
                max_steps=self._max_steps,
                task=task,
            )
        else:
            env = environment

        # Use RLController to execute the episode.
        controller = RLController(
            env,
            self._policy,
            max_steps=self._max_steps,
            record_trajectory=True,
        )
        trajectory = controller.run_episode()

        # Gather final state information.
        final_architecture = env.manager.get_architecture()
        final_evaluation = env.current_evaluation

        # Architecture version: prefer the trajectory's last transition info,
        # fall back to the final evaluation's architecture_version.
        final_version: Optional[int] = None
        if trajectory.transitions:
            last_info = trajectory.transitions[-1].info.environment_info
            if isinstance(last_info, dict) and "architecture_version" in last_info:
                raw = last_info["architecture_version"]
                if raw is not None:
                    final_version = int(raw)
        if final_version is None and final_evaluation is not None:
            raw = final_evaluation.architecture_version
            if raw is not None:
                final_version = int(raw)

        return TaskEpisodeResult(
            task_id=task.task_id,
            task_category=task.task_category,
            difficulty=task.difficulty,
            trajectory=trajectory,
            total_reward=trajectory.total_reward,
            episode_length=trajectory.length,
            final_architecture_id=final_architecture.architecture_id,
            final_architecture_version=final_version,
            final_evaluation=final_evaluation,
            terminated=trajectory.terminated,
            truncated=trajectory.truncated,
            task_context=task_context,
        )

    # ------------------------------------------------------------------
    # Multi-task execution
    # ------------------------------------------------------------------

    def run_experiment(
        self,
        distribution: MetaTaskDistribution,
        *,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
    ) -> MultiTaskExperimentResult:
        """Execute multiple task episodes and return the experiment result.

        Given a MetaTaskDistribution:

        * Select/sample multiple tasks (either by explicit task IDs or by
          sampling from the distribution).
        * Execute one episode per task.
        * Keep each task's environment state isolated.
        * Store results for every task.
        * Produce an experiment-level result.

        Parameters
        ----------
        distribution:
            The MetaTaskDistribution to draw tasks from.
        n:
            Number of tasks to sample from the distribution. Required if
            task_ids is not provided. Must be >= 1.
        seed:
            Optional random seed for task selection. If None, uses the
            controller's seed if set.
        task_ids:
            Optional explicit list of task IDs to execute in order. If
            provided, n is ignored.

        Returns
        -------
        MultiTaskExperimentResult:
            The result of the multi-task experiment.

        Raises
        ------
        ValueError:
            If the distribution is empty, n < 1, or both n and task_ids are
            None.
        TypeError:
            If distribution is not a MetaTaskDistribution.
        """
        if distribution is None:
            raise ValueError("distribution must not be None")
        from app.rl.task_distribution import MetaTaskDistribution as MTD

        if not isinstance(distribution, MTD):
            raise TypeError(
                f"distribution must be a MetaTaskDistribution, got {type(distribution).__name__}"
            )

        if distribution.task_count() == 0:
            raise ValueError(
                "cannot run experiment with an empty task distribution"
            )

        # Resolve which tasks to execute.
        resolved_tasks = self._resolve_tasks(
            distribution,
            n=n,
            seed=seed,
            task_ids=task_ids,
        )

        # Execute each task in order, keeping environments isolated.
        task_results: List[TaskEpisodeResult] = []
        for task in resolved_tasks:
            result = self.run_task(task)
            task_results.append(result)

        # Build experiment-level summary.
        total_reward = sum(r.total_reward for r in task_results)
        avg_reward = total_reward / len(task_results) if task_results else 0.0
        total_length = sum(r.episode_length for r in task_results)
        avg_length = total_length / len(task_results) if task_results else 0.0
        task_ids_list = [r.task_id for r in task_results]

        # Use the effective seed that was used for task resolution.
        effective_seed = seed if seed is not None else self._seed

        return MultiTaskExperimentResult(
            task_results=task_results,
            task_ids=task_ids_list,
            num_tasks=len(task_results),
            total_reward=total_reward,
            average_reward=avg_reward,
            average_episode_length=avg_length,
            seed=effective_seed,
            distribution_name=distribution.name,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_tasks(
        self,
        distribution: MetaTaskDistribution,
        *,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
    ) -> List[MetaTask]:
        """Resolve the ordered list of tasks to execute.

        Parameters
        ----------
        distribution:
            The task distribution.
        n:
            Number of tasks to sample.
        seed:
            Optional seed for sampling.
        task_ids:
            Optional explicit task IDs.

        Returns
        -------
        List[MetaTask]:
            Ordered list of tasks to execute.
        """
        if task_ids is not None:
            # Explicit task IDs: fetch each task in order.
            if not task_ids:
                raise ValueError("task_ids must not be empty")
            tasks = []
            for tid in task_ids:
                try:
                    tasks.append(distribution.get_task(tid))
                except KeyError:
                    raise ValueError(f"task_id '{tid}' not found in distribution") from None
            return tasks

        if n is None:
            raise ValueError(
                "either n or task_ids must be provided"
            )
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")

        effective_seed = seed if seed is not None else self._seed
        sampled = distribution.sample(n=n, seed=effective_seed)
        return list(sampled)
