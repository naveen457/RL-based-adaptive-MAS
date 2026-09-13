"""
Meta-RL environment: task-aware wrapper around the MAS architecture environment.

This module provides a lightweight task-aware wrapper around
``MASArchitectureEnv`` that makes the environment "task-aware" for Meta-RL.

The MetaEnvironment:

* Accepts a ``MetaTask`` that describes the current adaptation problem.
* Exposes the task context derived from the MetaTask.
* Combines the architecture observation with the task context into a
  deterministic meta-observation.
* Optionally integrates task-performance evaluation for task-aware rewards.
* Preserves the existing environment step/reset behavior.
* Does NOT call an LLM.
* Does NOT execute the actual LangGraph workflow.
* Does NOT implement learning.

This is a pure observation-level wrapper. The underlying environment still
uses the same action space, reward calculator, and evaluation pipeline. The
only addition is that the observation now includes task context.

Design notes
------------

* The meta-observation is a deterministic combination of:

  1. The architecture observation from ``ArchitectureStateEncoder.encode()``.
  2. The task context from ``MetaTaskContext.to_dict()``.

* The combination is done by merging the two dicts with a "task_context" key.
  This keeps the architecture observation intact and adds the task context as
  a nested structure.

* The wrapper delegates all environment operations (step, reset, evaluation,
  action mapping) to the underlying ``MASArchitectureEnv``.

* The wrapper is designed to be compatible with future Meta-RL controllers
  that will consume the meta-observation and produce architecture actions.

* The wrapper does not change the action space, reward function, or environment
  dynamics. It only adds task context to the observation.

* Step 21: Task-performance-aware reward integration.
  When task_performance_weight > 0, the environment computes:
    - Structural evaluation (Step 9/10)
    - Task-performance evaluation (Step 20)
    - Combined reward = structural_delta + weight * task_delta

Meta-RL formulation (repeated for clarity)
-------------------------------------------

State
~~~~~
The meta-RL state (meta-observation) is:

    meta_observation = {
        **architecture_observation,  # from ArchitectureStateEncoder
        "task_context": task_context_dict,  # from MetaTaskContext.to_dict()
    }

where architecture_observation contains:

* agent_ids: sorted list of agent IDs.
* active_agent_indices: indices of active agents.
* activity_vector: binary activity vector.
* role_vector: role strings for each agent.
* adjacency_matrix: directed communication adjacency matrix.
* agent_count: total agent count.
* active_agent_count: active agent count.

and task_context_dict contains:

* task_category: task category string.
* required_capabilities: list of required capabilities.
* difficulty: difficulty rating (1-5).
* context_features: additional deterministic features.

Action
~~~~~~
ArchitectureAction selected through the existing ArchitectureActionMapper.

Reward
~~~~~~
Default (task_performance_weight=0.0):
    reward = current.evaluation.overall_score - previous.evaluation.overall_score

Task-aware (task_performance_weight > 0):
    structural_delta = current.overall_score - previous.overall_score
    task_delta = current_task_success - previous_task_success
    combined_reward = structural_delta + task_performance_weight * task_delta

for valid transitions, and INVALID_TRANSITION_PENALTY for invalid transitions.

Task
~~~~
A MetaTask representing a specific MAS adaptation problem.

Episode
~~~~~~~
A sequence of architecture adaptations for one task, from reset to
termination/truncation.

Inner-loop (NOT IMPLEMENTED)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Future task-specific adaptation: given a new task, adapt the architecture
through a sequence of architecture actions.

Outer-loop (NOT IMPLEMENTED)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Future learning across multiple tasks: learn meta-parameters or a meta-policy
that enables fast adaptation to new tasks.

Limitations
-----------

* The task context is lightweight and does not include semantic embeddings.
* The meta-observation is a plain Python dict, not a tensor.
* No Meta-RL training loops are implemented.
* The wrapper does not change the underlying environment behavior.
* Task-performance scores are deterministic compatibility proxies, NOT actual LLM execution.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from app.architecture.manager import ArchitectureManager
from app.rl.environment import MASArchitectureEnv
from app.rl.meta_task import MetaTask, MetaTaskContext
from app.evaluation.task_performance import (
    TaskPerformanceEvaluator,
    TaskPerformanceResult,
)


class MetaEnvironment:
    """Task-aware wrapper around MASArchitectureEnv for Meta-RL.

    This wrapper adds task context to the environment observation while
    preserving the existing environment behavior.

    Parameters
    ----------
    manager:
        Initial architecture manager. The environment keeps its own copy and
        resets to this initial architecture.
    role_options:
        Optional finite role vocabulary used for CHANGE_ROLE actions.
    max_steps:
        Maximum number of architecture transitions per episode.
    task:
        Optional MetaTask describing the current adaptation problem. If None,
        the environment will have no task context.
    """

    def __init__(
        self,
        manager: ArchitectureManager,
        *,
        role_options: Optional[list[str]] = None,
        max_steps: Optional[int] = None,
        task: Optional[MetaTask] = None,
        task_performance_weight: float = 0.0,
    ) -> None:
        self._env = MASArchitectureEnv(
            manager,
            role_options=role_options,
            max_steps=max_steps,
        )
        self._task: Optional[MetaTask] = None
        self._task_context: Optional[MetaTaskContext] = None
        self._task_performance_weight = task_performance_weight
        
        # Task-performance evaluation components
        self._task_performance_evaluator = TaskPerformanceEvaluator()
        self._previous_task_performance: Optional[TaskPerformanceResult] = None
        self._current_task_performance: Optional[TaskPerformanceResult] = None
        
        if task is not None:
            self.set_task(task)

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def set_task(self, task: MetaTask) -> None:
        """Set or change the current task.

        This updates the task and derives the task context. The environment
        is NOT reset automatically; the caller should call reset() if they
        want to start a new episode with the new task.

        Parameters
        ----------
        task:
            The MetaTask to set as the current task.
        """
        self._task = task
        self._task_context = task.to_context()
        # Reset task-performance evaluations when task changes
        self._previous_task_performance = None
        self._current_task_performance = None

    def get_task(self) -> Optional[MetaTask]:
        """Return the current task, or None if no task is set."""
        return self._task

    def get_task_context(self) -> Optional[MetaTaskContext]:
        """Return the current task context, or None if no task is set."""
        return self._task_context

    @property
    def task(self) -> Optional[MetaTask]:
        """Current task, or None if no task is set."""
        return self._task

    @property
    def task_context(self) -> Optional[MetaTaskContext]:
        """Current task context, or None if no task is set."""
        return self._task_context

    # ------------------------------------------------------------------
    # Environment delegation
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment and return the meta-observation.

        This delegates to the underlying environment's reset() and then
        combines the architecture observation with the task context (if any)
        to produce the meta-observation.

        Parameters
        ----------
        seed:
            Ignored in the deterministic baseline.
        options:
            Ignored in the deterministic baseline.

        Returns
        -------
        meta_observation:
            The combined architecture + task context observation.
        info:
            Safe debug information from the underlying environment.
        """
        del seed, options  # unused in the deterministic baseline

        architecture_observation, info = self._env.reset()
        meta_observation = self._make_meta_observation(architecture_observation)
        
        # Reset task-performance evaluations
        self._previous_task_performance = None
        self._current_task_performance = None
        
        # Evaluate initial architecture for task performance if task is set
        if self._task is not None:
            architecture = self._env.manager.get_architecture()
            self._current_task_performance = self._task_performance_evaluator.evaluate(
                self._task, architecture
            )
            self._previous_task_performance = self._current_task_performance
        
        return meta_observation, info

    def step(self, action_id: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one architecture transition.

        This delegates to the underlying environment's step() and then
        combines the resulting architecture observation with the task context
        (if any) to produce the meta-observation.

        If a task is set and task_performance_weight > 0, the reward is
        computed as a combined structural + task-performance reward.

        Parameters
        ----------
        action_id:
            Integer action selected by the agent/controller.

        Returns
        -------
        meta_observation:
            The combined architecture + task context observation.
        reward:
            Baseline reward for the transition (possibly task-aware).
        terminated:
            True when the episode reached a terminal condition.
        truncated:
            True when the episode exceeded max_steps.
        info:
            Safe debug information from the underlying environment.
        """
        # Store previous task performance before step
        previous_task_perf = self._current_task_performance
        
        # Delegate to underlying environment
        architecture_observation, structural_reward, terminated, truncated, info = (
            self._env.step(action_id)
        )
        meta_observation = self._make_meta_observation(architecture_observation)
        
        # Compute task-aware reward if task is set and weight > 0
        combined_reward = structural_reward
        if self._task is not None and self._task_performance_weight > 0:
            # Evaluate current architecture for task performance
            architecture = self._env.manager.get_architecture()
            self._current_task_performance = self._task_performance_evaluator.evaluate(
                self._task, architecture
            )
            
            # Compute combined reward
            from app.evaluation.task_performance import TaskPerformanceRewardCalculator
            task_reward_calc = TaskPerformanceRewardCalculator(
                task_performance_weight=self._task_performance_weight
            )
            combined_reward = task_reward_calc.calculate(
                previous_result=self._env.current_evaluation,
                current_result=self._env.current_evaluation,
                valid_transition=info.get("transition", {}).get("valid", False),
                previous_task_performance=previous_task_perf,
                current_task_performance=self._current_task_performance,
            )
            
            # Add task-aware reward info to info dict
            info["reward"] = task_reward_calc.calculate_with_context(
                previous_result=self._env.current_evaluation,
                current_result=self._env.current_evaluation,
                valid_transition=info.get("transition", {}).get("valid", False),
                previous_task_performance=previous_task_perf,
                current_task_performance=self._current_task_performance,
            )
        
        return meta_observation, combined_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation space placeholders
    # ------------------------------------------------------------------

    def observation_space(self):
        """Placeholder for a Gymnasium observation space."""
        return self._env.observation_space()

    def action_space(self):
        """Placeholder for a Gymnasium action space."""
        return self._env.action_space()

    # ------------------------------------------------------------------
    # Public accessors (delegated to underlying environment)
    # ------------------------------------------------------------------

    @property
    def manager(self) -> ArchitectureManager:
        """Current architecture manager (live, mutated by step)."""
        return self._env.manager

    @property
    def encoder(self) -> Any:
        """Current state encoder."""
        return self._env.encoder

    @property
    def mapper(self) -> Any:
        """Current action mapper."""
        return self._env.mapper

    @property
    def max_steps(self) -> int:
        """Episode step limit."""
        return self._env.max_steps

    @property
    def step_count(self) -> int:
        """Number of steps taken in the current episode."""
        return self._env.step_count

    @property
    def terminated(self) -> bool:
        """Episode termination flag."""
        return self._env.terminated

    @property
    def truncated(self) -> bool:
        """Episode truncation flag (max steps reached)."""
        return self._env.truncated

    @property
    def current_evaluation(self) -> Any:
        """Current architecture evaluation result."""
        return self._env.current_evaluation

    @property
    def evaluator(self) -> Any:
        """Current architecture evaluator."""
        return self._env.evaluator

    @property
    def reward_calculator(self) -> Any:
        """Current reward calculator."""
        return self._env.reward_calculator

    @property
    def task_performance_weight(self) -> float:
        """Current task-performance weight for combined rewards."""
        return self._task_performance_weight

    @property
    def task_performance_evaluator(self) -> TaskPerformanceEvaluator:
        """Task-performance evaluator."""
        return self._task_performance_evaluator

    @property
    def previous_task_performance(self) -> Optional[TaskPerformanceResult]:
        """Previous task-performance evaluation result."""
        return self._previous_task_performance

    @property
    def current_task_performance(self) -> Optional[TaskPerformanceResult]:
        """Current task-performance evaluation result."""
        return self._current_task_performance

    def get_possible_actions(self) -> list:
        """Return the current valid action set through the manager."""
        return self._env.get_possible_actions()

    def encode_action(self, action: Any) -> int:
        """Encode an action against the current action mapping."""
        return self._env.encode_action(action)

    def decode_action(self, action_id: int) -> Any:
        """Decode an action ID to an ArchitectureAction."""
        return self._env.decode_action(action_id)

    def evaluate_architecture(self, architecture: Any) -> Any:
        """Evaluate an architecture using the current evaluator."""
        return self._env.evaluate_architecture(architecture)

    def evaluate_task_performance(self, task: MetaTask, architecture: Any) -> TaskPerformanceResult:
        """Evaluate task performance for a given task and architecture.

        Parameters
        ----------
        task:
            The MetaTask to evaluate against.
        architecture:
            The architecture to evaluate.

        Returns
        -------
        TaskPerformanceResult
            The task-performance evaluation result.
        """
        return self._task_performance_evaluator.evaluate(task, architecture)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_meta_observation(self, architecture_observation: Dict[str, Any]) -> Dict[str, Any]:
        """Combine architecture observation with task context.

        If no task is set, the meta-observation is just the architecture
        observation with an empty task_context.

        Parameters
        ----------
        architecture_observation:
            The architecture observation from the underlying environment.

        Returns
        -------
        meta_observation:
            The combined observation with task context.
        """
        if self._task_context is not None:
            task_context_dict = self._task_context.to_dict()
        else:
            task_context_dict = {
                "task_category": "",
                "required_capabilities": [],
                "difficulty": 1,
                "context_features": {},
            }

        meta_observation = {
            **architecture_observation,
            "task_context": task_context_dict,
        }
        return meta_observation

    def _update_task_performance_after_step(self) -> None:
        """Update task-performance evaluation after a step.

        This is called internally after each step to keep task-performance
        evaluation current. Normally called automatically in step().
        """
        if self._task is not None:
            architecture = self._env.manager.get_architecture()
            self._previous_task_performance = self._current_task_performance
            self._current_task_performance = self._task_performance_evaluator.evaluate(
                self._task, architecture
            )
