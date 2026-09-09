"""
Deterministic architecture state encoding for the RL environment.

This module converts a MASArchitecture into a fixed, machine-readable
representation suitable for an RL/Meta-RL controller to observe.

Design notes
------------
* The encoder is deterministic and reproducible. The same architecture
  always produces the same observation.
* No LLM calls are made. No embeddings are used. No arbitrary
  natural-language state strings are produced.
* The shape of the observation is fixed for a given architecture
  configuration (agent ordering and role vocabulary are derived from the
  architecture itself for the baseline).

Baseline representation
-----------------------
The baseline observation is built from:

1. Agent activity vector.
   For each defined agent (sorted by agent_id), emit a binary indicator:
     1 if the agent is active, else 0.

2. Role encoding vector.
   For each defined agent (sorted by agent_id), emit the agent's role as a
   plain string. This keeps the observation human-readable and deterministic
   without introducing an embedding.

3. Communication adjacency matrix.
   Rows and columns are sorted agent_ids. Entry (i, j) is 1 if there is a
   directed communication edge from agent i to agent j, else 0.

This is intentionally simple for Step 8. It can be extended later without
changing the environment contract.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from app.architecture.models import MASArchitecture


class ArchitectureStateEncoder:
    """Convert MASArchitecture into a deterministic RL-compatible observation.

    The observation is a plain Python structure. Consumers may convert it
    into a Gymnasium/NumPy/JAX representation as needed.
    """

    def __init__(self, architecture: MASArchitecture) -> None:
        self._architecture = architecture
        self._agent_ids: List[str] = sorted(architecture.agent_ids)
        self._role_map: Dict[str, str] = architecture.role_map

    # ------------------------------------------------------------------
    # Public encoding API
    # ------------------------------------------------------------------

    def encode(self) -> dict:
        """Return the current architecture as a deterministic observation.

        The returned dict is suitable for logging and for later conversion
        to an RL observation tensor (not provided in this step).
        """
        return {
            "agent_ids": self._agent_ids,
            "active_agent_indices": self._active_agent_indices(),
            "activity_vector": self._activity_vector(),
            "role_vector": self._role_vector(),
            "adjacency_matrix": self._adjacency_matrix(),
            "agent_count": self._architecture.agent_count,
            "active_agent_count": self._architecture.active_agent_count,
        }

    # ------------------------------------------------------------------
    # Deterministic helpers
    # ------------------------------------------------------------------

    def _active_agent_indices(self) -> List[int]:
        """Return sorted indices of active agents within the sorted agent_id list."""
        active_ids = self._architecture.active_agent_ids
        return [idx for idx, agent_id in enumerate(self._agent_ids) if agent_id in active_ids]

    def _activity_vector(self) -> List[int]:
        """Binary activity vector aligned with self._agent_ids."""
        active_ids = self._architecture.active_agent_ids
        return [1 if agent_id in active_ids else 0 for agent_id in self._agent_ids]

    def _role_vector(self) -> List[str]:
        """Role vector aligned with self._agent_ids.

        Roles are taken from the architecture's role_map and emitted in the
        same deterministic order as the agent id list.
        """
        return [self._role_map[agent_id] for agent_id in self._agent_ids]

    def _adjacency_matrix(self) -> List[List[int]]:
        """Directed adjacency matrix over sorted agent_ids.

        Entry (i, j) is 1 when there is a communication edge from agent i to
        agent j, else 0.
        """
        edge_set: Dict[Tuple[str, str], int] = {
            (edge.source, edge.target): 1 for edge in self._architecture.communication_edges
        }
        matrix: List[List[int]] = []
        for source in self._agent_ids:
            row: List[int] = []
            for target in self._agent_ids:
                row.append(edge_set.get((source, target), 0))
            matrix.append(row)
        return matrix
