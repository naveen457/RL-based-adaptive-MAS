"""
Task-conditioned neural architecture policy foundation for Step 26.

This module adds the first learned policy component for the adaptive MAS
architecture environment. It is intentionally small and offline:

* no PPO, DQN, MAML, Reptile, PEARL, RL^2, recurrent adaptation, or LLM calls
* no parallel action system
* one neural score per existing ArchitectureActionMapper action ID
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from app.architecture.actions import ArchitectureAction
from app.rl.action_space import ArchitectureActionMapper
from app.rl.meta_task import MetaTask, MetaTaskContext
from app.rl.policy import BasePolicy


DEFAULT_CATEGORY_VOCABULARY = [
    "analysis",
    "coding",
    "debugging",
    "documentation",
    "planning",
    "reasoning",
    "refactoring",
    "research",
    "summarization",
    "synthesis",
    "unknown",
]

DEFAULT_CAPABILITY_VOCABULARY = [
    "analysis",
    "api_design",
    "coding",
    "comparison",
    "critique",
    "data_interpretation",
    "data_structures",
    "diagnostic",
    "final_answer_generation",
    "implementation",
    "information_synthesis",
    "logical_deduction",
    "optimization",
    "performance_evaluation",
    "planning",
    "quality_assessment",
    "reasoning",
    "research",
    "routing",
    "security",
    "security_audit",
    "simplification",
    "summarization",
    "synthesis",
    "task_analysis",
    "testing",
    "verification",
    "unknown",
]

DEFAULT_CONTEXT_FEATURE_KEYS = [
    "complexity",
    "has_dependencies",
    "has_web_search",
    "num_files",
    "num_steps",
    "requires_coding",
    "requires_research",
]

DEFAULT_ROLE_VOCABULARY = [
    "analysis",
    "implementation",
    "planning",
    "research",
    "synthesis",
    "verification",
    "unknown",
]


def _index_map(vocabulary: Sequence[str]) -> Dict[str, int]:
    return {token: idx for idx, token in enumerate(vocabulary)}


def _numeric_feature(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        return min(len(value), 100) / 100.0
    if isinstance(value, (list, tuple, set, dict)):
        return min(len(value), 100) / 100.0
    return 0.0


class DeterministicTaskEncoder:
    """Encode MetaTask / MetaTaskContext into a fixed-size tensor.

    Layout:
    category one-hot with unknown bucket, normalized difficulty,
    required-capability multi-hot with unknown bucket, and a fixed ordered set
    of numeric context features.
    """

    def __init__(
        self,
        *,
        category_vocabulary: Optional[Sequence[str]] = None,
        capability_vocabulary: Optional[Sequence[str]] = None,
        context_feature_keys: Optional[Sequence[str]] = None,
    ) -> None:
        self.category_vocabulary = list(category_vocabulary or DEFAULT_CATEGORY_VOCABULARY)
        self.capability_vocabulary = list(capability_vocabulary or DEFAULT_CAPABILITY_VOCABULARY)
        self.context_feature_keys = list(context_feature_keys or DEFAULT_CONTEXT_FEATURE_KEYS)

        if "unknown" not in self.category_vocabulary:
            self.category_vocabulary.append("unknown")
        if "unknown" not in self.capability_vocabulary:
            self.capability_vocabulary.append("unknown")

        self._category_index = _index_map(self.category_vocabulary)
        self._capability_index = _index_map(self.capability_vocabulary)

    @property
    def output_dim(self) -> int:
        return (
            len(self.category_vocabulary)
            + 1
            + len(self.capability_vocabulary)
            + len(self.context_feature_keys)
        )

    def encode_context(self, context: MetaTaskContext) -> torch.Tensor:
        category = [0.0] * len(self.category_vocabulary)
        category_idx = self._category_index.get(
            context.task_category, self._category_index["unknown"]
        )
        category[category_idx] = 1.0

        difficulty = [(float(context.difficulty) - 1.0) / 4.0]

        capabilities = [0.0] * len(self.capability_vocabulary)
        unknown_capability_seen = False
        for capability in context.required_capabilities:
            idx = self._capability_index.get(capability)
            if idx is None:
                unknown_capability_seen = True
            else:
                capabilities[idx] = 1.0
        if unknown_capability_seen:
            capabilities[self._capability_index["unknown"]] = 1.0

        context_features = [
            _numeric_feature(context.context_features.get(key))
            for key in self.context_feature_keys
        ]

        return torch.tensor(
            [*category, *difficulty, *capabilities, *context_features],
            dtype=torch.float32,
        )

    def encode_task(self, task: MetaTask) -> torch.Tensor:
        return self.encode_context(task.to_context())

    def to_config(self) -> Dict[str, List[str]]:
        return {
            "category_vocabulary": list(self.category_vocabulary),
            "capability_vocabulary": list(self.capability_vocabulary),
            "context_feature_keys": list(self.context_feature_keys),
        }


class ArchitectureStateTensorEncoder:
    """Encode the existing architecture observation into a fixed-size tensor."""

    def __init__(
        self,
        *,
        max_agent_count: int = 5,
        role_vocabulary: Optional[Sequence[str]] = None,
    ) -> None:
        if max_agent_count < 1:
            raise ValueError("max_agent_count must be >= 1")
        self.max_agent_count = max_agent_count
        self.role_vocabulary = list(role_vocabulary or DEFAULT_ROLE_VOCABULARY)
        if "unknown" not in self.role_vocabulary:
            self.role_vocabulary.append("unknown")
        self._role_index = _index_map(self.role_vocabulary)

    @property
    def output_dim(self) -> int:
        n = self.max_agent_count
        return n + n * len(self.role_vocabulary) + n * n + 2

    def encode_observation(self, observation: Dict[str, Any]) -> torch.Tensor:
        n = self.max_agent_count
        agent_count = int(observation.get("agent_count", 0) or 0)
        active_agent_count = int(observation.get("active_agent_count", 0) or 0)

        activity = [0.0] * n
        for idx, value in enumerate(list(observation.get("activity_vector", []))[:n]):
            activity[idx] = 1.0 if value else 0.0

        role_values: List[float] = []
        roles = list(observation.get("role_vector", []))[:n]
        for idx in range(n):
            role_one_hot = [0.0] * len(self.role_vocabulary)
            role = roles[idx] if idx < len(roles) else "unknown"
            role_idx = self._role_index.get(str(role), self._role_index["unknown"])
            role_one_hot[role_idx] = 1.0
            role_values.extend(role_one_hot)

        adjacency_values: List[float] = []
        matrix = list(observation.get("adjacency_matrix", []))
        for row_idx in range(n):
            row = list(matrix[row_idx]) if row_idx < len(matrix) else []
            for col_idx in range(n):
                value = row[col_idx] if col_idx < len(row) else 0
                adjacency_values.append(1.0 if value else 0.0)

        max_count = float(max(1, n))
        count_features = [
            min(float(agent_count), max_count) / max_count,
            min(float(active_agent_count), max_count) / max_count,
        ]

        return torch.tensor(
            [*activity, *role_values, *adjacency_values, *count_features],
            dtype=torch.float32,
        )

    def to_config(self) -> Dict[str, Any]:
        return {
            "max_agent_count": self.max_agent_count,
            "role_vocabulary": list(self.role_vocabulary),
        }


class TaskConditionedArchitectureNet(nn.Module):
    """MLP that scores the existing discrete architecture action IDs."""

    def __init__(
        self,
        architecture_dim: int,
        task_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if architecture_dim < 1 or task_dim < 1 or action_dim < 1:
            raise ValueError("network dimensions must be positive")

        self.architecture_dim = architecture_dim
        self.task_dim = task_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(architecture_dim + task_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, architecture_embedding: torch.Tensor, task_embedding: torch.Tensor) -> torch.Tensor:
        if architecture_embedding.ndim == 1:
            architecture_embedding = architecture_embedding.unsqueeze(0)
        if task_embedding.ndim == 1:
            task_embedding = task_embedding.unsqueeze(0)
        combined = torch.cat([architecture_embedding, task_embedding], dim=-1)
        return self.net(combined)

    def score_actions(
        self, architecture_embedding: torch.Tensor, task_embedding: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(architecture_embedding, task_embedding)


class TaskConditionedPolicy(BasePolicy):
    """Task-conditioned learned scorer over existing architecture action IDs."""

    def __init__(
        self,
        *,
        action_mapper: Optional[ArchitectureActionMapper] = None,
        architecture_encoder: Optional[ArchitectureStateTensorEncoder] = None,
        task_encoder: Optional[DeterministicTaskEncoder] = None,
        action_dim: Optional[int] = None,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        exploration_rate: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.action_mapper = action_mapper
        self.architecture_encoder = architecture_encoder or ArchitectureStateTensorEncoder()
        self.task_encoder = task_encoder or DeterministicTaskEncoder()
        self.action_dim = int(action_dim or (action_mapper.action_count if action_mapper else 1))
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.rng = random.Random(seed)

        if seed is not None:
            torch.manual_seed(seed)

        self.model = TaskConditionedArchitectureNet(
            architecture_dim=self.architecture_encoder.output_dim,
            task_dim=self.task_encoder.output_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def select_action(
        self,
        observation: Dict[str, Any],
        valid_action_ids: List[int],
        task_context: Optional[MetaTaskContext] = None,
        *,
        deterministic: bool = True,
        exploration_rate: Optional[float] = None,
    ) -> int:
        if not valid_action_ids:
            raise ValueError("no valid actions available")
        self._validate_action_ids(valid_action_ids)

        epsilon = self.exploration_rate if exploration_rate is None else exploration_rate
        if not deterministic and epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.choice(valid_action_ids))

        scores = self.action_scores_tensor(observation, task_context).squeeze(0)
        valid_scores = scores[torch.tensor(valid_action_ids, dtype=torch.long)]
        best_local = int(torch.argmax(valid_scores).item())
        return int(valid_action_ids[best_local])

    def select_architecture_action(
        self,
        observation: Dict[str, Any],
        valid_action_ids: List[int],
        task_context: Optional[MetaTaskContext] = None,
        *,
        deterministic: bool = True,
        exploration_rate: Optional[float] = None,
    ) -> ArchitectureAction:
        action_id = self.select_action(
            observation,
            valid_action_ids,
            task_context,
            deterministic=deterministic,
            exploration_rate=exploration_rate,
        )
        return self.decode_action(action_id)

    def decode_action(self, action_id: int) -> ArchitectureAction:
        if self.action_mapper is None:
            raise ValueError("an ArchitectureActionMapper is required to decode actions")
        return self.action_mapper.decode(int(action_id))

    def action_scores_tensor(
        self,
        observation: Dict[str, Any],
        task_context: Optional[MetaTaskContext] = None,
    ) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(
                self.encode_architecture(observation).unsqueeze(0),
                self.encode_task_context(task_context).unsqueeze(0),
            )

    def action_scores(
        self,
        observation: Dict[str, Any],
        task_context: Optional[MetaTaskContext] = None,
    ) -> List[float]:
        return [float(v) for v in self.action_scores_tensor(observation, task_context).squeeze(0)]

    def update_supervised(
        self,
        observation: Dict[str, Any],
        task_context: Optional[MetaTaskContext],
        target_scores: Sequence[float],
    ) -> float:
        if len(target_scores) != self.action_dim:
            raise ValueError("target_scores length must match action_dim")

        self.model.train()
        self.optimizer.zero_grad()
        prediction = self.model(
            self.encode_architecture(observation).unsqueeze(0),
            self.encode_task_context(task_context).unsqueeze(0),
        ).squeeze(0)
        target = torch.tensor(list(target_scores), dtype=torch.float32)
        loss = nn.functional.mse_loss(prediction, target)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_from_transition(
        self,
        observation: Dict[str, Any],
        task_context: Optional[MetaTaskContext],
        action_id: int,
        reward: float,
        next_observation: Dict[str, Any],
        done: bool,
        valid_action_ids: List[int],
        next_valid_action_ids: List[int],
        gamma: float = 0.9,
    ) -> float:
        del valid_action_ids
        self._validate_action_ids([action_id])
        if next_valid_action_ids:
            self._validate_action_ids(next_valid_action_ids)

        self.model.train()
        self.optimizer.zero_grad()
        scores = self.model(
            self.encode_architecture(observation).unsqueeze(0),
            self.encode_task_context(task_context).unsqueeze(0),
        ).squeeze(0)

        with torch.no_grad():
            next_scores = self.model(
                self.encode_architecture(next_observation).unsqueeze(0),
                self.encode_task_context(task_context).unsqueeze(0),
            ).squeeze(0)
            bootstrap = 0.0
            if not done and next_valid_action_ids:
                bootstrap = float(next_scores[torch.tensor(next_valid_action_ids)].max().item())
            target = torch.tensor(float(reward) + gamma * bootstrap, dtype=torch.float32)

        loss = nn.functional.mse_loss(scores[int(action_id)], target)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        payload = {
            "model_state_dict": self.model.state_dict(),
            "architecture_encoder": self.architecture_encoder.to_config(),
            "task_encoder": self.task_encoder.to_config(),
            "action_dim": self.action_dim,
            "hidden_dim": self.model.hidden_dim,
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        action_mapper: Optional[ArchitectureActionMapper] = None,
        seed: Optional[int] = None,
    ) -> "TaskConditionedPolicy":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        architecture_encoder = ArchitectureStateTensorEncoder(**payload["architecture_encoder"])
        task_encoder = DeterministicTaskEncoder(**payload["task_encoder"])
        policy = cls(
            action_mapper=action_mapper,
            architecture_encoder=architecture_encoder,
            task_encoder=task_encoder,
            action_dim=int(payload["action_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            learning_rate=float(payload["learning_rate"]),
            exploration_rate=float(payload["exploration_rate"]),
            seed=seed,
        )
        policy.model.load_state_dict(payload["model_state_dict"])
        policy.model.eval()
        return policy

    def load_state(self, path: str | Path) -> None:
        restored = self.load(path, action_mapper=self.action_mapper)
        self.__dict__.update(restored.__dict__)

    def encode_architecture(self, observation: Dict[str, Any]) -> torch.Tensor:
        return self.architecture_encoder.encode_observation(observation)

    def encode_task_context(self, task_context: Optional[MetaTaskContext]) -> torch.Tensor:
        context = task_context or MetaTaskContext(
            task_category="unknown",
            required_capabilities=[],
            difficulty=1,
            context_features={},
        )
        return self.task_encoder.encode_context(context)

    def parameters_changed_since(self, before: Iterable[torch.Tensor]) -> bool:
        return any(
            not torch.equal(old, new)
            for old, new in zip(before, self.model.parameters())
        )

    def clone_parameters(self) -> List[torch.Tensor]:
        return [parameter.detach().clone() for parameter in self.model.parameters()]

    def _validate_action_ids(self, action_ids: Sequence[int]) -> None:
        for action_id in action_ids:
            if action_id < 0 or action_id >= self.action_dim:
                raise ValueError(f"invalid action id for policy output: {action_id}")
            if self.action_mapper is not None and not self.action_mapper.is_valid_id(int(action_id)):
                raise ValueError(f"action id is not in the current mapper: {action_id}")


def create_task_conditioned_policy_from_environment(
    env: Any,
    *,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    exploration_rate: float = 0.0,
    seed: Optional[int] = None,
) -> TaskConditionedPolicy:
    """Construct a policy sized from an existing architecture environment."""

    observation = env.encoder.encode()
    max_agent_count = int(observation.get("agent_count", 5) or 5)
    architecture_encoder = ArchitectureStateTensorEncoder(max_agent_count=max_agent_count)
    return TaskConditionedPolicy(
        action_mapper=env.mapper,
        architecture_encoder=architecture_encoder,
        task_encoder=DeterministicTaskEncoder(),
        action_dim=env.mapper.action_count,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        exploration_rate=exploration_rate,
        seed=seed,
    )
