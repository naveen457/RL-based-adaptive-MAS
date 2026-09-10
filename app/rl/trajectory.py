"""
Transition and trajectory representations for the RL controller.

This module defines data structures for recording experience during
environment interaction:

* Transition: A single step's (state, action, reward, next_state, terminated,
  truncated, info) tuple.
* Trajectory: A sequence of transitions from a single episode.

Design notes
------------
* These representations use Pydantic for validation and dataclasses for
  lightweight containers where appropriate.
* They are designed to be serializable for logging, analysis, and potential
  future use by learned RL policies.
* They do NOT implement any training logic or memory persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ActionInfo(BaseModel):
    """Information about the action taken in a transition.

    This is a lightweight representation that captures the action ID and
    a serialized form of the action for logging/analysis.
    """

    action_id: int = Field(description="Integer action ID selected by the policy.")
    action_serialized: Dict[str, Any] = Field(
        description="JSON-friendly serialization of the action.",
    )


class TransitionInfo(BaseModel):
    """Metadata and info dict from the environment for a transition."""

    environment_info: Dict[str, Any] = Field(
        description="Info dict returned by the environment's step() method.",
    )
    valid_transition: bool = Field(
        description="Whether the transition was accepted by the architecture.",
    )


class Transition(BaseModel):
    """A single RL transition: (s, a, r, s', terminated, truncated, info).

    This represents one step of interaction between the policy and the
    environment.

    Attributes
    ----------
    step:
        Step number within the episode (1-indexed).
    state:
        Observation before the action was taken.
    action:
        Information about the action taken.
    reward:
        Scalar reward received after the action.
    next_state:
        Observation after the action was taken.
    terminated:
        Whether the episode reached a terminal condition.
    truncated:
        Whether the episode was truncated (e.g., max steps reached).
    info:
        Environment info and transition metadata.
    """

    step: int = Field(description="Step number within the episode (1-indexed).")
    state: Dict[str, Any] = Field(description="Observation before the action.")
    action: ActionInfo = Field(description="Action taken in this transition.")
    reward: float = Field(description="Reward received after the action.")
    next_state: Dict[str, Any] = Field(description="Observation after the action.")
    terminated: bool = Field(description="Whether the episode terminated.")
    truncated: bool = Field(description="Whether the episode was truncated.")
    info: TransitionInfo = Field(description="Environment info and metadata.")


class Trajectory(BaseModel):
    """A complete trajectory: sequence of transitions from one episode.

    This represents the full experience of one episode, suitable for:
    * Logging and analysis.
    * Potential future use by learned RL policies.
    * Debugging and visualization.

    Attributes
    ----------
    episode_id:
        Unique identifier for this episode.
    transitions:
        Ordered list of transitions in this episode.
    total_reward:
        Sum of rewards across all transitions.
    length:
        Number of transitions in the trajectory.
    terminated:
        Whether the episode reached a terminal condition.
    truncated:
        Whether the episode was truncated.
    """

    episode_id: str = Field(description="Unique identifier for this episode.")
    transitions: List[Transition] = Field(
        default_factory=list,
        description="Ordered list of transitions in this episode.",
    )

    @property
    def total_reward(self) -> float:
        """Sum of rewards across all transitions."""
        return sum(t.reward for t in self.transitions)

    @property
    def length(self) -> int:
        """Number of transitions in the trajectory."""
        return len(self.transitions)

    @property
    def terminated(self) -> bool:
        """Whether the episode reached a terminal condition."""
        if not self.transitions:
            return False
        return self.transitions[-1].terminated

    @property
    def truncated(self) -> bool:
        """Whether the episode was truncated."""
        if not self.transitions:
            return False
        return self.transitions[-1].truncated

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the trajectory to a plain dict."""
        return {
            "episode_id": self.episode_id,
            "transitions": [t.model_dump() for t in self.transitions],
            "total_reward": self.total_reward,
            "length": self.length,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
