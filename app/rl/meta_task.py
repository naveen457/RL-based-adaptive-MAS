"""
Meta-RL task formulation for the adaptive MAS research project.

This module formalizes the Meta-RL problem:

    "Adapt the architecture of an LLM-based Multi-Agent System to different
    task environments by learning which architecture actions improve task
    performance."

It defines:
* MetaTask: A specific MAS adaptation problem with task identity, description,
  capabilities, difficulty, and context features.
* MetaTaskContext: A deterministic numerical/categorical feature vector derived
  from a MetaTask that can be consumed by a Meta-RL controller.

This module does NOT implement:
* Meta-RL training loops (inner-loop / outer-loop).
* Neural networks, PPO, DQN, or policy gradients.
* Dynamic LangGraph rebuilding.
* LLM calls or API interactions.
* Persistent memory.

State/Action/Reward/Task/Episode formulation
---------------------------------------------

State
~~~~~
The meta-RL state is a combination of:

1. Architecture state: the current MAS architecture encoded deterministically
   by ``ArchitectureStateEncoder`` (agent activity, roles, communication
   topology).

2. Task context: deterministic numerical/categorical features derived from the
   current MetaTask (task_category, required_capabilities, difficulty, and any
   additional deterministic features).

Together these form the meta-observation that a future Meta-RL controller will
consume.

Action
~~~~~~
Actions are ``ArchitectureAction`` objects selected through the existing
``ArchitectureActionMapper`` mechanism. The action space is the same as in the
base RL environment; the meta-RL layer adds task context to the observation but
does not change the action space itself.

Reward
~~~~~~
The baseline reward is the structural evaluation delta from Step 9/10::

    reward = current.evaluation.overall_score - previous.evaluation.overall_score

for valid transitions, and a fixed negative penalty for invalid transitions.

This is a structural proxy reward. Future work may add task-performance reward
signals when task execution is available; that is out of scope for this step.

Task
~~~~
A task is a specific MAS adaptation problem represented by a ``MetaTask``:

* A task_id identifying the task.
* A task_description describing what the MAS is expected to do.
* A task_category classifying the task (e.g., "coding", "research",
  "analysis", "synthesis", "planning").
* A required_capabilities list describing which capabilities the MAS must
  provide to attempt the task.
* A difficulty rating (1-5) indicating how challenging the task is expected to
  be for the MAS.
* Additional context/features needed for adaptation.

Episode
~~~~~~~
An episode is a sequence of architecture adaptations for one task. The
environment is stepped until termination or truncation, and the trajectory
records the full sequence of (state, action, reward, next_state) tuples.

The environment is then reset, potentially with a new task, for the next
episode.

Inner-loop
~~~~~~~~~~
The inner-loop is future task-specific adaptation: given a new task, adapt the
architecture through a sequence of architecture actions to improve task
performance. This is NOT implemented in this step.

Outer-loop
~~~~~~~~~~
The outer-loop is future learning across multiple tasks: learn meta-parameters
or a meta-policy that enables fast adaptation to new tasks. This is NOT
implemented in this step.

Design notes
------------

* All task/context representations are Pydantic models with validation.
* Serialization is deterministic and JSON-friendly.
* No LLM calls are made. No embeddings are used.
* The design is compatible with future Meta-RL controllers that will consume
  the meta-observation and produce architecture actions.
* The task context is intentionally lightweight: it captures the information
  that is likely useful for Meta-RL adaptation without over-committing to a
  specific Meta-RL algorithm.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class MetaTask(BaseModel):
    """A specific MAS adaptation problem.

    A MetaTask describes a task that the MAS may be asked to solve, including
    the task identity, description, category, required capabilities, difficulty,
    and any additional context features needed for adaptation.

    This is the unit of "task" in the Meta-RL formulation. A Meta-RL
    controller will eventually observe a MetaTask (via its context) and adapt
    the architecture to improve task performance.

    Parameters
    ----------
    task_id:
        Unique identifier for the task (e.g., "task-coder-001",
        "task-research-042").
    task_description:
        Natural language description of the task. This is human-readable and
        may be used for logging/analysis, but is not directly consumed by the
        Meta-RL controller (the controller consumes the context features
        instead).
    task_category:
        Category classifying the task. Examples: "coding", "research",
        "analysis", "synthesis", "planning", "debugging", "refactoring",
        "documentation".
    required_capabilities:
        List of capability strings that the MAS must provide to attempt the
        task (e.g., ["coding", "testing"], ["research", "synthesis"]).
    difficulty:
        Difficulty rating from 1 (easiest) to 5 (hardest). This is a
        deterministic numerical feature useful for Meta-RL.
    context_features:
        Additional deterministic numerical/categorical features that may be
        useful for Meta-RL adaptation. These are arbitrary key-value pairs
        where values are JSON-serializable and deterministic. Keys should be
        strings. This field is optional and defaults to an empty dict.
    notes:
        Optional human-readable notes about the task. Not consumed by the
        Meta-RL controller.
    """

    task_id: str = Field(
        description="Unique identifier for the task.",
    )
    task_description: str = Field(
        description="Natural language description of the task.",
    )
    task_category: str = Field(
        description="Category classifying the task (e.g., 'coding', 'research').",
    )
    required_capabilities: List[str] = Field(
        default_factory=list,
        description="Capabilities the MAS must provide to attempt the task.",
    )
    difficulty: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Difficulty rating from 1 (easiest) to 5 (hardest).",
    )
    context_features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional deterministic features for Meta-RL adaptation.",
    )
    notes: str = Field(
        default="",
        description="Optional human-readable notes about the task.",
    )

    @field_validator("task_id")
    @classmethod
    def task_id_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("task_id must be a non-empty string")
        return v.strip()

    @field_validator("task_description")
    @classmethod
    def task_description_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("task_description must be a non-empty string")
        return v.strip()

    @field_validator("task_category")
    @classmethod
    def task_category_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("task_category must be a non-empty string")
        return v.strip()

    @field_validator("required_capabilities")
    @classmethod
    def required_capabilities_must_be_unique(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            raise ValueError("required_capabilities must contain unique values")
        return v

    @field_validator("context_features")
    @classmethod
    def context_features_values_must_be_deterministic(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that context feature values are JSON-serializable and deterministic."""
        allowed_types = (str, int, float, bool, list, dict, type(None))
        for key, value in v.items():
            if not isinstance(key, str):
                raise ValueError(f"context_features key must be a string, got {type(key)}")
            if not isinstance(value, allowed_types):
                raise ValueError(
                    f"context_features value for key '{key}' must be JSON-serializable, "
                    f"got {type(value)}"
                )
        return v

    def serialize(self) -> Dict[str, Any]:
        """Return a JSON-friendly serialization of the task."""
        return self.model_dump(mode="json", exclude_none=True)

    def to_context(self) -> MetaTaskContext:
        """Create a MetaTaskContext from this task.

        This is the primary way to derive the context that a Meta-RL controller
        will consume from a MetaTask.
        """
        return MetaTaskContext.from_task(self)


class MetaTaskContext(BaseModel):
    """Deterministic numerical/categorical feature vector for a MetaTask.

    This is the representation that a Meta-RL controller will consume as part
    of the meta-observation. It captures:

    * task_category: one-hot or categorical encoding of the task category.
    * required_capabilities: binary vector over a known capability vocabulary.
    * difficulty: numerical difficulty rating.
    * additional context features: any additional deterministic features.

    The context is designed to be:

    * Deterministic: the same MetaTask always produces the same context.
    * Comparable: different tasks can be compared via their context vectors.
    * Lightweight: only the information likely useful for Meta-RL adaptation.
    * Extensible: additional features can be added without breaking existing
      consumers.

    Parameters
    ----------
    task_category:
        The task category string.
    required_capabilities:
        List of required capability strings.
    difficulty:
        Difficulty rating from 1 to 5.
    context_features:
        Additional deterministic features for Meta-RL.
    category_embedding:
        Optional categorical encoding of the task category. If provided, this
        should be a list of floats representing the category. This is for
        future use when a learned embedding may be available; in the baseline
        this is None.
    capability_vector:
        Optional binary vector over a known capability vocabulary. If provided,
        this should be a list of ints (0 or 1) aligned with a known capability
        ordering. This is for future use; in the baseline this is None.
    """

    task_category: str = Field(
        description="The task category string.",
    )
    required_capabilities: List[str] = Field(
        default_factory=list,
        description="List of required capability strings.",
    )
    difficulty: int = Field(
        ge=1,
        le=5,
        description="Difficulty rating from 1 to 5.",
    )
    context_features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional deterministic features for Meta-RL.",
    )
    category_embedding: Optional[List[float]] = Field(
        default=None,
        description="Optional categorical embedding of the task category.",
    )
    capability_vector: Optional[List[int]] = Field(
        default=None,
        description="Optional binary capability vector over a known vocabulary.",
    )

    @field_validator("task_category")
    @classmethod
    def task_category_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("task_category must be a non-empty string")
        return v.strip()

    @field_validator("context_features")
    @classmethod
    def context_features_values_must_be_deterministic(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that context feature values are JSON-serializable and deterministic."""
        allowed_types = (str, int, float, bool, list, dict, type(None))
        for key, value in v.items():
            if not isinstance(key, str):
                raise ValueError(f"context_features key must be a string, got {type(key)}")
            if not isinstance(value, allowed_types):
                raise ValueError(
                    f"context_features value for key '{key}' must be JSON-serializable, "
                    f"got {type(value)}"
                )
        return v

    @classmethod
    def from_task(cls, task: MetaTask) -> "MetaTaskContext":
        """Create a MetaTaskContext from a MetaTask.

        This derives the context from the task's category, capabilities,
        difficulty, and context_features. The resulting context is deterministic
        and fully determined by the task.
        """
        return cls(
            task_category=task.task_category,
            required_capabilities=list(task.required_capabilities),
            difficulty=task.difficulty,
            context_features=dict(task.context_features),
            category_embedding=None,
            capability_vector=None,
        )

    def serialize(self) -> Dict[str, Any]:
        """Return a JSON-friendly serialization of the context."""
        return self.model_dump(mode="json", exclude_none=True)

    def to_dict(self) -> Dict[str, Any]:
        """Return the context as a plain dict for consumption by a controller.

        This is the primary entry point for a Meta-RL controller to consume the
        task context. The returned dict contains:

        * task_category: the task category string.
        * required_capabilities: the list of required capabilities.
        * difficulty: the difficulty rating.
        * context_features: any additional features.

        The dict is deterministic and suitable for inclusion in a meta-observation.
        """
        return {
            "task_category": self.task_category,
            "required_capabilities": list(self.required_capabilities),
            "difficulty": self.difficulty,
            "context_features": dict(self.context_features),
        }

    def __eq__(self, other: object) -> bool:
        """Two MetaTaskContext objects are equal if they have the same fields.

        This is important for Meta-RL: the same task must always produce the
        same context, and different tasks can be compared via their contexts.
        """
        if not isinstance(other, MetaTaskContext):
            return NotImplemented
        return (
            self.task_category == other.task_category
            and self.required_capabilities == other.required_capabilities
            and self.difficulty == other.difficulty
            and self.context_features == other.context_features
            and self.category_embedding == other.category_embedding
            and self.capability_vector == other.capability_vector
        )
