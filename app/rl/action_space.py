"""
Deterministic action-space mapping for the RL environment.

This module provides a stable bidirectional mapping between:

* ArchitectureAction (from Step 6)
* integer action IDs used by a future RL/Meta-RL controller

Design notes
------------
* The mapping is derived from the ArchitectureManager's
  ``get_possible_actions(...)`` method so the environment and the action
  space stay consistent with the validated architecture action model.
* The mapping is deterministic for a given architecture + role vocabulary.
* The same action always maps to the same ID and the same ID always maps
  back to the same action (within one environment construction).
* Invalid IDs are rejected explicitly.
* The mapping is rebuilt when the architecture changes (documented below).

Action ordering
---------------
The action list is sorted deterministically so the ID assignment is
reproducible. The sort key is:

1. action_type value
2. agent_id when present
3. source when present
4. target when present
5. new_role when present

This keeps role-change actions for the same agent adjacent, edge actions
ordered, and so on.

Changing action sets after a transition
----------------------------------------
Architecture actions are state-dependent. A valid action set for one
architecture is not necessarily a valid action set for the next. This
environment therefore rebuilds the action mapping after every successful
step. Existing IDs are not preserved across transitions; an action ID is
only meaningful for the action mapping that produced it. This is a
deliberate, documented limitation for the Step 8 baseline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.manager import ArchitectureManager


def _action_sort_key(action: ArchitectureAction) -> Tuple:
    """Deterministic sort key for architecture actions."""
    return (
        action.action_type.value,
        action.agent_id,
        action.source,
        action.target,
        action.new_role,
    )


class ArchitectureActionMapper:
    """Deterministic integer mapping for a fixed set of ArchitectureAction objects.

    Typical usage in the environment:

        mapper = ArchitectureActionMapper.from_manager(manager, role_options=...)
        action_id = mapper.encode(action)
        action = mapper.decode(action_id)
    """

    def __init__(
        self,
        actions: List[ArchitectureAction],
        role_options: Optional[List[str]] = None,
    ) -> None:
        if not actions:
            raise ValueError("action set must be non-empty")

        # Deduplicate by serialization key while preserving deterministic order.
        order: List[ArchitectureAction] = []
        seen_keys: set = set()
        for action in sorted(actions, key=_action_sort_key):
            key = _action_serialization_key(action)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            order.append(action)

        if not order:
            raise ValueError("action set must be non-empty after deduplication")

        self._actions = order
        self._role_options = list(role_options) if role_options is not None else None
        self._encode: Dict[Tuple, int] = {
            _action_serialization_key(action): idx for idx, action in enumerate(order)
        }
        self._decode: Dict[int, ArchitectureAction] = {
            idx: action for idx, action in enumerate(order)
        }

    # ------------------------------------------------------------------
    # Construction from architecture manager
    # ------------------------------------------------------------------

    @classmethod
    def from_manager(
        cls,
        manager: ArchitectureManager,
        role_options: Optional[List[str]] = None,
    ) -> "ArchitectureActionMapper":
        """Build an action mapping from the current architecture + role vocabulary."""
        possible = manager.get_possible_actions(role_options=role_options)
        return cls(actions=possible, role_options=role_options)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, action: ArchitectureAction) -> int:
        """Return the deterministic integer ID for *action*."""
        key = _action_serialization_key(action)
        try:
            return self._encode[key]
        except KeyError:
            raise ValueError(f"action is not in this action mapping: {action.model_dump(exclude_none=True)}") from None

    def decode(self, action_id: int) -> ArchitectureAction:
        """Return the ArchitectureAction for *action_id*."""
        if action_id not in self._decode:
            raise ValueError(f"invalid action id: {action_id}")
        return self._decode[action_id]

    def is_valid_id(self, action_id: int) -> bool:
        """Return True if *action_id* maps to a known action."""
        return action_id in self._decode

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def action_count(self) -> int:
        """Number of actions in this mapping."""
        return len(self._actions)

    @property
    def actions(self) -> List[ArchitectureAction]:
        """Sorted list of actions in this mapping."""
        return list(self._actions)

    @property
    def role_options(self) -> Optional[List[str]]:
        """Role vocabulary used to build this mapping, if any."""
        return list(self._role_options) if self._role_options is not None else None

    # ------------------------------------------------------------------
    # Serialization safety
    # ------------------------------------------------------------------

    def encode_action_safely(self, action: ArchitectureAction) -> dict:
        """Return a JSON-friendly encoding of an action plus its ID.

        This is intended for info/logging, not for releasing secrets.
        ArchitectureAction fields are architecture-level identifiers/roles.
        """
        return {
            "action_id": self.encode(action),
            "action": action.serialize(),
        }


def _action_serialization_key(action: ArchitectureAction) -> Tuple:
    """Stable key for deduplicating and indexing actions in the same mapping.

    The key is a hashable tuple. Note that action.action_type is an
    ArchitectureActionType enum whose values are strings, so the enum member
    itself is usable as the first component of the key.
    """
    return (
        action.action_type,
        action.agent_id,
        action.source,
        action.target,
        action.new_role,
    )
