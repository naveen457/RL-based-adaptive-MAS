"""
Policy interface and baseline implementations for the RL controller.

This module defines:
* A policy interface that selects actions from observations and valid action IDs.
* A deterministic baseline policy suitable for testing.
* A random policy wrapper for stochastic exploration (without introducing
  complex RL algorithms).

Design notes
------------
* The policy interface is intentionally minimal and algorithm-agnostic.
* Policies do NOT directly modify the architecture. They only select action IDs.
* The interface is designed to be usable by both a future learned RL policy
  and a future Meta-RL policy without modification.
* Baseline policies are deterministic/reproducible and contain no learned
  parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from app.architecture.actions import ArchitectureAction


class BasePolicy(ABC):
    """Abstract policy interface for the RL controller.

    A policy receives:
    * The current observation (dict from ArchitectureStateEncoder.encode()).
    * The currently valid action IDs (list of ints).

    And returns:
    * A selected action ID (int).

    Policies do NOT modify the architecture. They only select actions.
    This keeps the policy decoupled from the environment and ready for
    future learned or Meta-RL implementations.
    """

    @abstractmethod
    def select_action(
        self,
        observation: Dict,
        valid_action_ids: List[int],
    ) -> int:
        """Select an action ID given the current observation and valid actions.

        Parameters
        ----------
        observation:
            Current architecture observation from the environment.
        valid_action_ids:
            List of action IDs that are currently valid in the environment.

        Returns
        -------
        action_id:
            The selected action ID.

        Raises
        ------
        ValueError:
            If valid_action_ids is empty.
        """
        ...


class DeterministicBaselinePolicy(BasePolicy):
    """A simple deterministic baseline policy for testing.

    This policy implements a fixed heuristic:
    * Prefer ADD_EDGE actions to increase connectivity.
    * If no ADD_EDGE actions are available, prefer ACTIVATE_AGENT.
    * Otherwise, select the first available action.

    This policy is deterministic and reproducible. It is intended for
    testing the controller/environment interaction, not for demonstration
    of learned behavior.

    This policy does NOT:
    * Learn from experience.
    * Use neural networks.
    * Implement Meta-RL.
    """

    def __init__(self) -> None:
        """Initialize the deterministic baseline policy."""
        pass

    def select_action(
        self,
        observation: Dict,
        valid_action_ids: List[int],
    ) -> int:
        """Select an action using the fixed heuristic.

        The heuristic prioritizes:
        1. ADD_EDGE actions (to increase connectivity).
        2. ACTIVATE_AGENT actions (to ensure agent availability).
        3. First available action as fallback.

        Parameters
        ----------
        observation:
            Current architecture observation.
        valid_action_ids:
            List of valid action IDs.

        Returns
        -------
        action_id:
            The selected action ID.

        Raises
        ------
        ValueError:
            If valid_action_ids is empty.
        """
        if not valid_action_ids:
            raise ValueError("no valid actions available")

        # Single action: return it directly.
        if len(valid_action_ids) == 1:
            return valid_action_ids[0]

        # Priority 1: ADD_EDGE actions.
        add_edge_ids = self._filter_action_ids_by_type(
            observation, valid_action_ids, "add_edge"
        )
        if add_edge_ids:
            # Select the first ADD_EDGE action deterministically.
            return add_edge_ids[0]

        # Priority 2: ACTIVATE_AGENT actions.
        activate_ids = self._filter_action_ids_by_type(
            observation, valid_action_ids, "activate_agent"
        )
        if activate_ids:
            return activate_ids[0]

        # Fallback: return the first available action.
        return valid_action_ids[0]

    def _filter_action_ids_by_type(
        self,
        observation: Dict,
        action_ids: List[int],
        action_type: str,
    ) -> List[int]:
        """Filter action IDs by action type string.

        This uses the action serialization in the observation's action space
        information if available, otherwise falls back to an empty list.

        Parameters
        ----------
        observation:
            Current observation.
        action_ids:
            List of action IDs to filter.
        action_type:
            Action type string to match (e.g., "add_edge").

        Returns
        -------
        filtered_ids:
            Action IDs matching the specified type, in deterministic order.
        """
        # The observation may contain action space metadata in info, but the
        # policy receives only the observation dict. We use a simple approach:
        # the environment can pass action type information via the observation
        # if needed. For the baseline, we assume the caller provides this
        # through the observation if desired.

        # For the baseline policy, we check if the observation contains
        # action metadata. If not, we return an empty list for type filtering.
        action_metadata = observation.get("action_metadata")
        if action_metadata is None:
            return []

        # action_metadata is expected to be a dict mapping action_id -> action_type
        filtered = [
            aid for aid in action_ids
            if action_metadata.get(aid, {}).get("action_type") == action_type
        ]
        return filtered


class RandomPolicy(BasePolicy):
    """A random policy for stochastic exploration.

    This policy selects actions uniformly at random from the valid action IDs.
    It is useful for testing and exploration, but does not implement any
    learned behavior.

    This policy depends on an external random number generator to keep it
    testable and reproducible.
    """

    def __init__(self, rng=None) -> None:
        """Initialize the random policy.

        Parameters
        ----------
        rng:
            A random number generator with a ``choice`` method (e.g.,
            ``random.Random`` or ``numpy.random.Generator``). If None,
            uses Python's ``random`` module.
        """
        if rng is None:
            import random
            self._rng = random
        else:
            self._rng = rng

    def select_action(
        self,
        observation: Dict,
        valid_action_ids: List[int],
    ) -> int:
        """Select a random action from the valid action IDs.

        Parameters
        ----------
        observation:
            Current architecture observation (unused in this policy).
        valid_action_ids:
            List of valid action IDs.

        Returns
        -------
        action_id:
            A randomly selected action ID.

        Raises
        ------
        ValueError:
            If valid_action_ids is empty.
        """
        if not valid_action_ids:
            raise ValueError("no valid actions available")

        return self._rng.choice(valid_action_ids)
