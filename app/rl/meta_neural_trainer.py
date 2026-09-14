"""
Task-conditioned neural meta-training and fast-adaptation experiment.

This is the Step 27 experimental foundation. It trains one shared
TaskConditionedPolicy across multiple MetaTasks, then compares fast adaptation
on an unseen task against a freshly initialized policy with the same update
budget.

This is not PEARL, RL^2, MAML, Reptile, PPO, DQN, or recurrent meta-learning.
It is task-conditioned meta-training with fast adaptation using the Step 26
neural policy and deterministic architecture-task compatibility proxies.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.manager import ArchitectureManager
from app.architecture.models import MASArchitecture
from app.evaluation.task_performance import TaskPerformanceEvaluator
from app.rl.environment import MASArchitectureEnv
from app.rl.meta_task import MetaTask
from app.rl.task_conditioned_policy import (
    TaskConditionedPolicy,
    create_task_conditioned_policy_from_environment,
)
from app.rl.task_distribution import (
    MetaTaskDistribution,
    create_baseline_task_distribution,
)

PROXY_NOTE = (
    "Deterministic architecture-task compatibility proxy; "
    "this is NOT actual LLM task execution."
)


@dataclass(frozen=True)
class MetaTrainingResult:
    """Serializable result for shared task-conditioned neural meta-training."""

    training_task_ids: List[str]
    unseen_task_ids: List[str]
    update_task_ids: List[str]
    num_updates: int
    initial_loss: float
    final_loss: float
    loss_curve: List[float]
    performance_curve: List[float]
    deterministic_seed: Optional[int]
    proxy_note: str = PROXY_NOTE

    def serialize(self) -> Dict[str, Any]:
        return asdict(self)

    def to_dict(self) -> Dict[str, Any]:
        return self.serialize()


@dataclass(frozen=True)
class FastAdaptationResult:
    """Serializable result for unseen-task fast adaptation."""

    target_task_id: str
    perturbation: Dict[str, str]
    perturbed_start_performance: float
    meta_policy_before_adaptation: float
    meta_policy_after_adaptation: float
    fresh_policy_after_adaptation: float
    meta_adaptation_improvement: float
    meta_vs_scratch_advantage: float
    pre_adaptation_performance: float
    post_adaptation_performance: float
    from_scratch_performance: float
    adaptation_improvement: float
    transfer_adaptation_advantage: float
    adaptation_updates: int
    seed: Optional[int]
    meta_adaptation_losses: List[float] = field(default_factory=list)
    from_scratch_losses: List[float] = field(default_factory=list)
    proxy_note: str = PROXY_NOTE

    def serialize(self) -> Dict[str, Any]:
        return asdict(self)

    def to_dict(self) -> Dict[str, Any]:
        return self.serialize()


def deterministic_train_test_split(
    distribution: Optional[MetaTaskDistribution] = None,
    *,
    train_ratio: float = 0.7,
    seed: Optional[int] = None,
) -> Tuple[MetaTaskDistribution, MetaTaskDistribution]:
    """Create a deterministic non-overlapping task split."""

    source = distribution or create_baseline_task_distribution()
    return source.split(train_ratio=train_ratio, seed=seed)


class MetaNeuralTrainer:
    """Train one shared task-conditioned neural architecture policy."""

    def __init__(
        self,
        *,
        policy: Optional[TaskConditionedPolicy] = None,
        task_distribution: Optional[MetaTaskDistribution] = None,
        training_tasks: Optional[Sequence[MetaTask]] = None,
        unseen_tasks: Optional[Sequence[MetaTask]] = None,
        train_ratio: float = 0.7,
        role_options: Optional[Sequence[str]] = None,
        max_steps: int = 2,
        learning_rate: float = 1e-3,
        meta_updates: int = 6,
        adaptation_updates: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        if meta_updates < 1:
            raise ValueError("meta_updates must be >= 1")
        if adaptation_updates < 0:
            raise ValueError("adaptation_updates must be >= 0")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        self.task_distribution = (
            task_distribution or create_baseline_task_distribution()
        )
        self.train_ratio = train_ratio
        self.role_options = list(role_options or ["analysis", "planning"])
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.meta_updates = meta_updates
        self.adaptation_updates = adaptation_updates
        self.seed = seed
        self.rng = random.Random(seed)
        if seed is not None:
            torch.manual_seed(seed)
        self.evaluator = TaskPerformanceEvaluator()

        if training_tasks is None or unseen_tasks is None:
            train_dist, test_dist = deterministic_train_test_split(
                self.task_distribution,
                train_ratio=train_ratio,
                seed=seed,
            )
            if training_tasks is None:
                training_tasks = train_dist.get_tasks()
            if unseen_tasks is None:
                unseen_tasks = test_dist.get_tasks()

        self.training_tasks = list(training_tasks)
        self.unseen_tasks = list(unseen_tasks)
        if not self.training_tasks:
            raise ValueError("training_tasks must contain at least one task")
        if not self.unseen_tasks:
            raise ValueError("unseen_tasks must contain at least one task")

        overlap = set(self.training_task_ids) & set(self.unseen_task_ids)
        if overlap:
            raise ValueError(
                f"training and unseen tasks must not overlap: {sorted(overlap)}"
            )

        self._template_env = self._make_environment()
        self.policy = policy or create_task_conditioned_policy_from_environment(
            self._template_env,
            learning_rate=learning_rate,
            seed=seed,
        )
        self.policy.action_mapper = self._template_env.mapper
        if self.policy.action_dim != self._template_env.mapper.action_count:
            raise ValueError("policy action_dim must match environment action space")

    @property
    def training_task_ids(self) -> List[str]:
        return [task.task_id for task in self.training_tasks]

    @property
    def unseen_task_ids(self) -> List[str]:
        return [task.task_id for task in self.unseen_tasks]

    def train(self, *, num_updates: Optional[int] = None) -> MetaTrainingResult:
        """Run shared-policy task-conditioned meta-training."""

        updates = self.meta_updates if num_updates is None else num_updates
        if updates < 1:
            raise ValueError("num_updates must be >= 1")

        losses: List[float] = []
        performances: List[float] = []
        update_task_ids: List[str] = []

        for update_index in range(updates):
            task = self.training_tasks[update_index % len(self.training_tasks)]
            step = self._adaptation_update(self.policy, task)
            losses.append(step["loss"])
            performances.append(step["post_performance"])
            update_task_ids.append(task.task_id)

        return MetaTrainingResult(
            training_task_ids=self.training_task_ids,
            unseen_task_ids=self.unseen_task_ids,
            update_task_ids=update_task_ids,
            num_updates=updates,
            initial_loss=losses[0],
            final_loss=losses[-1],
            loss_curve=losses,
            performance_curve=performances,
            deterministic_seed=self.seed,
        )

    def meta_train(self, *, num_updates: Optional[int] = None) -> MetaTrainingResult:
        """Explicit Step 27 name for the shared meta-training operation."""

        return self.train(num_updates=num_updates)

    def adapt_and_evaluate(
        self,
        unseen_task: Optional[MetaTask] = None,
        *,
        adaptation_updates: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> FastAdaptationResult:
        """Compare meta-trained fast adaptation with same-budget fresh training."""

        target = unseen_task or self.unseen_tasks[0]
        if target.task_id in set(self.training_task_ids):
            raise ValueError("unseen_task must not be part of the training tasks")

        updates = (
            self.adaptation_updates
            if adaptation_updates is None
            else adaptation_updates
        )
        if updates < 0:
            raise ValueError("adaptation_updates must be >= 0")

        compare_seed = self.seed if seed is None else seed
        meta_policy = self._clone_policy(self.policy, seed=compare_seed)
        fresh_policy = create_task_conditioned_policy_from_environment(
            self._make_environment(),
            learning_rate=self.policy.learning_rate,
            seed=compare_seed,
        )

        perturbed_architecture = self._perturbed_target_architecture()
        perturbed_start = self._evaluate_task_performance(
            target, perturbed_architecture
        )
        before = perturbed_start

        meta_losses: List[float] = []
        meta_architecture = perturbed_architecture
        for _ in range(updates):
            update = self._adaptation_update(
                meta_policy,
                target,
                initial_architecture=meta_architecture,
            )
            meta_losses.append(update["loss"])
            meta_architecture = update["after_architecture"]
        after = self._evaluate_task_performance(target, meta_architecture)

        fresh_losses: List[float] = []
        fresh_architecture = perturbed_architecture
        for _ in range(updates):
            update = self._adaptation_update(
                fresh_policy,
                target,
                initial_architecture=fresh_architecture,
            )
            fresh_losses.append(update["loss"])
            fresh_architecture = update["after_architecture"]
        from_scratch = self._evaluate_task_performance(target, fresh_architecture)

        improvement = round(after - before, 6)
        advantage = round(after - from_scratch, 6)
        return FastAdaptationResult(
            target_task_id=target.task_id,
            perturbation=self._perturbation_action().serialize(),
            perturbed_start_performance=perturbed_start,
            meta_policy_before_adaptation=before,
            meta_policy_after_adaptation=after,
            fresh_policy_after_adaptation=from_scratch,
            meta_adaptation_improvement=improvement,
            meta_vs_scratch_advantage=advantage,
            pre_adaptation_performance=before,
            post_adaptation_performance=after,
            from_scratch_performance=from_scratch,
            adaptation_improvement=improvement,
            transfer_adaptation_advantage=advantage,
            adaptation_updates=updates,
            seed=compare_seed,
            meta_adaptation_losses=meta_losses,
            from_scratch_losses=fresh_losses,
        )

    def evaluate_policy_on_task(
        self,
        policy: TaskConditionedPolicy,
        task: MetaTask,
    ) -> float:
        """Evaluate the architecture selected by a policy for a target task."""

        env = self._make_environment()
        observation, _ = env.reset()
        policy.action_mapper = env.mapper
        action_id = policy.select_action(
            observation,
            list(range(env.mapper.action_count)),
            task.to_context(),
            deterministic=True,
        )
        _, _, _, _, info = env.step(action_id)
        if not info.get("transition", {}).get("valid", False):
            architecture = env.manager.to_architecture_model()
        else:
            architecture = env.manager.get_architecture()
        result = self.evaluator.evaluate(task, architecture)
        return float(result.task_success_score)

    def _adaptation_update(
        self,
        policy: TaskConditionedPolicy,
        task: MetaTask,
        *,
        initial_architecture: Optional[MASArchitecture] = None,
    ) -> Dict[str, Any]:
        env = self._make_environment(architecture=initial_architecture)
        observation, _ = env.reset()
        policy.action_mapper = env.mapper
        valid_actions = list(range(env.mapper.action_count))
        before_architecture = env.manager.to_architecture_model()
        before_performance = self.evaluator.evaluate(
            task, before_architecture
        ).task_success_score

        action_id = policy.select_action(
            observation,
            valid_actions,
            task.to_context(),
            deterministic=True,
        )
        next_observation, structural_reward, terminated, truncated, info = env.step(
            action_id
        )
        after_architecture = env.manager.to_architecture_model()
        after_performance = self.evaluator.evaluate(
            task, after_architecture
        ).task_success_score
        valid_transition = bool(info.get("transition", {}).get("valid", False))

        task_delta = after_performance - before_performance
        reward = structural_reward + task_delta
        if not valid_transition:
            reward = -1.0

        loss = policy.update_from_transition(
            observation=observation,
            task_context=task.to_context(),
            action_id=action_id,
            reward=reward,
            next_observation=next_observation,
            done=bool(terminated or truncated),
            valid_action_ids=valid_actions,
            next_valid_action_ids=list(range(env.mapper.action_count)),
        )
        return {
            "loss": round(float(loss), 10),
            "pre_performance": float(before_performance),
            "post_performance": float(after_performance),
            "reward": round(float(reward), 10),
            "action_id": float(action_id),
            "after_architecture": after_architecture,
        }

    def _make_environment(
        self,
        *,
        architecture: Optional[MASArchitecture] = None,
    ) -> MASArchitectureEnv:
        manager = (
            ArchitectureManager(architecture)
            if architecture is not None
            else ArchitectureManager.create_default_architecture()
        )
        return MASArchitectureEnv(
            manager,
            role_options=self.role_options,
            max_steps=self.max_steps,
        )

    def _perturbation_action(self) -> ArchitectureAction:
        return ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="researcher",
            new_role="analysis",
        )

    def _perturbed_target_architecture(self) -> MASArchitecture:
        env = self._make_environment()
        env.reset()
        action_id = env.mapper.encode(self._perturbation_action())
        _, _, _, _, info = env.step(action_id)
        if not info.get("transition", {}).get("valid", False):
            raise RuntimeError("Step 27 target perturbation was rejected")
        return env.manager.to_architecture_model()

    def _evaluate_task_performance(
        self,
        task: MetaTask,
        architecture: MASArchitecture,
    ) -> float:
        result = self.evaluator.evaluate(task, architecture)
        return float(result.task_success_score)

    def _clone_policy(
        self,
        policy: TaskConditionedPolicy,
        *,
        seed: Optional[int],
    ) -> TaskConditionedPolicy:
        clone = create_task_conditioned_policy_from_environment(
            self._make_environment(),
            learning_rate=policy.learning_rate,
            exploration_rate=policy.exploration_rate,
            seed=seed,
        )
        clone.model.load_state_dict(
            {
                key: value.detach().clone()
                for key, value in policy.model.state_dict().items()
            }
        )
        clone.model.eval()
        return clone


def run_tiny_step27_smoke(seed: int = 123) -> Dict[str, Any]:
    """Run the tiny deterministic Step 27 smoke experiment."""

    distribution = create_baseline_task_distribution()
    train_dist, test_dist = deterministic_train_test_split(
        distribution,
        train_ratio=0.7,
        seed=seed,
    )
    trainer = MetaNeuralTrainer(
        task_distribution=distribution,
        training_tasks=train_dist.get_tasks()[:3],
        unseen_tasks=[distribution.get_task("task-research-002")],
        meta_updates=4,
        adaptation_updates=2,
        learning_rate=1e-3,
        seed=seed,
    )
    train_result = trainer.train()
    adaptation_result = trainer.adapt_and_evaluate()
    return {
        "train_tasks": train_result.training_task_ids,
        "target_task": adaptation_result.target_task_id,
        "perturbation": adaptation_result.perturbation,
        "perturbed_start_performance": adaptation_result.perturbed_start_performance,
        "meta_before_adaptation": adaptation_result.meta_policy_before_adaptation,
        "meta_after_adaptation": adaptation_result.meta_policy_after_adaptation,
        "meta_initial_loss": train_result.initial_loss,
        "meta_final_loss": train_result.final_loss,
        "from_scratch": adaptation_result.fresh_policy_after_adaptation,
        "adaptation_improvement": adaptation_result.meta_adaptation_improvement,
        "adaptation_advantage": adaptation_result.meta_vs_scratch_advantage,
    }


def print_step27_smoke(seed: int = 123) -> None:
    """Print the required Step 27 smoke experiment and determinism check."""

    first = run_tiny_step27_smoke(seed=seed)
    second = run_tiny_step27_smoke(seed=seed)
    print("STEP27_SMOKE_COMPLETED")
    print(f"train_tasks={first['train_tasks']}")
    print(f"target_task={first['target_task']}")
    print(f"perturbation={first['perturbation']}")
    print(f"perturbed_start_performance={first['perturbed_start_performance']}")
    print(f"meta_before_adaptation={first['meta_before_adaptation']}")
    print(f"meta_after_adaptation={first['meta_after_adaptation']}")
    print(f"from_scratch={first['from_scratch']}")
    print(f"adaptation_improvement={first['adaptation_improvement']}")
    print(f"adaptation_advantage={first['adaptation_advantage']}")
    print("STEP27_DETERMINISM_CHECK_COMPLETED")
    print(f"RESULTS_IDENTICAL={first == second}")


if __name__ == "__main__":
    print_step27_smoke()
