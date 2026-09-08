from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.architecture.actions import ArchitectureAction
from app.architecture.manager import ArchitectureManager
from app.architecture.models import MASArchitecture


@dataclass(frozen=True)
class TransitionRecord:
    """A record of a single architecture transition.

    Captures the step number, the action that produced the transition,
    the previous architecture, and the resulting architecture.
    """

    step: int
    action: ArchitectureAction
    previous_architecture: MASArchitecture
    resulting_architecture: MASArchitecture


class AdaptiveArchitecture:
    """Deterministic adaptive layer over a MAS architecture.

    Responsibilities:
      - Hold the current architecture and its initial state.
      - Apply ArchitectureAction via ArchitectureManager, producing a new
        architecture only when the action succeeds and the result is valid.
      - Maintain an integer version and a transition history.
      - Reset to the initial architecture deterministically.
      - Provide snapshots without credentials.

    This layer deliberately does NOT implement:
      - RL / Meta-RL policy, reward, training, or environment.
      - Dynamic LangGraph graph construction or execution.
      - Persistence, memory, or LLM-based action selection.
    """

    def __init__(self, manager: ArchitectureManager) -> None:
        self._manager = manager
        self._initial_architecture = manager.to_architecture_model()
        self._version = 0
        self._history: List[TransitionRecord] = []

    # ------------------------------------------------------------------
    # Current state
    # ------------------------------------------------------------------

    def current_architecture(self) -> MASArchitecture:
        """Return the current architecture (live copy)."""
        return self._manager.to_architecture_model()

    def get_current_architecture(self) -> MASArchitecture:
        """Public accessor for the current architecture."""
        return self.current_architecture()

    def version(self) -> int:
        """Return the current architecture version.

        The initial architecture is version 0. Each successful transition
        increments the version by one.
        """
        return self._version

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------

    def step(self, action: ArchitectureAction) -> MASArchitecture:
        """Apply *action* to the current architecture.

        Returns the new architecture when the transition succeeds.

        Raises:
            TypeError: If *action* is not an ArchitectureAction.
            ValueError: If the action is rejected (invalid action,
                invalid resulting architecture, or no change produced).
        """
        if not isinstance(action, ArchitectureAction):
            raise TypeError("action must be an ArchitectureAction")

        previous_architecture = self._manager.to_architecture_model()
        try:
            new_architecture = self._manager.apply_action(action)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Transition rejected: {exc}") from exc

        # Defensive check: ensure the resulting architecture is valid.
        # ArchitectureManager.apply_action already validates, but we
        # enforce this at the adaptive layer boundary as well.
        if not new_architecture.is_valid:
            raise ValueError(
                "Resulting architecture is invalid; transition rejected"
            )

        self._version += 1
        record = TransitionRecord(
            step=self._version,
            action=action,
            previous_architecture=previous_architecture,
            resulting_architecture=new_architecture,
        )
        self._history.append(record)
        return new_architecture

    def apply_action(self, action: ArchitectureAction) -> MASArchitecture:
        """Alias for step() for callers that prefer action-oriented naming."""
        return self.step(action)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def history(self) -> List[TransitionRecord]:
        """Return an immutable copy of the transition history."""
        return list(self._history)

    def get_history(self) -> List[TransitionRecord]:
        """Public accessor for the transition history."""
        return self.history()

    def previous_architecture(self, step: int) -> Optional[MASArchitecture]:
        """Return the architecture that existed just before step *step*.

        *step* is 1-indexed: step 1 is the first transition, and the
        previous architecture is the initial architecture.
        """
        if step < 1 or step > len(self._history):
            return None
        return self._history[step - 1].previous_architecture

    def architecture_at_step(self, step: int) -> Optional[MASArchitecture]:
        """Return the architecture that resulted from step *step*.

        *step* is 1-indexed. Step 0 returns the initial architecture.
        Returns None when *step* is out of range.
        """
        if step == 0:
            return self._initial_architecture
        if step < 1 or step > len(self._history):
            return None
        return self._history[step - 1].resulting_architecture

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> MASArchitecture:
        """Reset to the initial architecture deterministically.

        The version is reset to 0 and the history is cleared. The
        architecture is restored to a deep copy of the initial state,
        so subsequent mutations do not affect the recorded initial state.
        """
        self._manager = ArchitectureManager(self._initial_architecture)
        self._version = 0
        self._history = []
        return self._manager.to_architecture_model()

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a safe snapshot of the current architecture.

        The snapshot contains only architecture information. It does NOT
        contain API keys, model credentials, environment secrets, or any
        other sensitive data.
        """
        return self._manager.serialize()

    def get_snapshot(self) -> dict:
        """Public accessor for the current architecture snapshot."""
        return self.snapshot()
