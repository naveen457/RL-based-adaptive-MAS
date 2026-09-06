from __future__ import annotations

from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

class AgentDefinition(BaseModel):
    """Formal description of a single agent in the MAS.

    This is the unit of architectural representation. Later, a Meta-RL
    controller will observe and mutate a collection of these.
    """

    agent_id: str = Field(
        description="Unique identifier for the agent (e.g. 'planner', 'coder')."
    )
    role: str = Field(
        description="The agent's role in the system (e.g. 'planning', 'research')."
    )
    description: str = Field(
        description="Human-readable description of what the agent does."
    )
    capabilities: List[str] = Field(
        default_factory=list,
        description="Capabilities the agent provides (e.g. 'research', 'coding', 'verification').",
    )
    active: bool = Field(
        default=True,
        description="Whether the agent is currently active in the architecture.",
    )
    # Reserved for future use by Meta-RL.
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary architecture-level metadata (no secrets).",
    )

    @field_validator("agent_id")
    @classmethod
    def agent_id_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("agent_id must be a non-empty string")
        return v.strip()

    @field_validator("role")
    @classmethod
    def role_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("role must be a non-empty string")
        return v.strip()

    @field_validator("description")
    @classmethod
    def description_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("description must be a non-empty string")
        return v.strip()


# ---------------------------------------------------------------------------
# Communication edge
# ---------------------------------------------------------------------------

class CommunicationEdge(BaseModel):
    """A directed communication link from one agent to another.

    Represents the fact that the output of *source* can be used by *target*.
    This is the architectural communication topology, not the runtime
    LangGraph edges (which include join points and conditional routing).
    """

    source: str = Field(description="ID of the source agent.")
    target: str = Field(description="ID of the target agent.")

    @field_validator("source", "target")
    @classmethod
    def ids_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("CommunicationEdge source/target must be non-empty")
        return v.strip()

    def is_self_loop(self) -> bool:
        """Return True if source and target are the same agent."""
        return self.source == self.target


# ---------------------------------------------------------------------------
# Full MAS architecture
# ---------------------------------------------------------------------------

class MASArchitecture(BaseModel):
    """Machine-readable representation of the entire multi-agent architecture.

    This is the object that later becomes the basis for the RL/Meta-RL state.
    For now it is only created, inspected, validated, and serialized.
    """

    architecture_id: str = Field(
        default="static-mas-v1",
        description="Identifier for this architecture configuration/version.",
    )
    agents: List[AgentDefinition] = Field(
        default_factory=list,
        description="All agents defined in this architecture.",
    )
    communication_edges: List[CommunicationEdge] = Field(
        default_factory=list,
        description="Directed communication links between agents.",
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Architecture-level metadata (version, description, etc.).",
    )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def agent_ids(self) -> Set[str]:
        """Set of all agent IDs in this architecture."""
        return {a.agent_id for a in self.agents}

    @property
    def active_agent_ids(self) -> Set[str]:
        """Set of IDs of agents that are currently active."""
        return {a.agent_id for a in self.agents if a.active}

    @property
    def active_agents(self) -> List[AgentDefinition]:
        """List of agents that are currently active."""
        return [a for a in self.agents if a.active]

    @property
    def agent_count(self) -> int:
        """Total number of agents defined."""
        return len(self.agents)

    @property
    def active_agent_count(self) -> int:
        """Number of currently active agents."""
        return len(self.active_agents)

    @property
    def role_map(self) -> Dict[str, str]:
        """Mapping from agent_id -> role."""
        return {a.agent_id: a.role for a in self.agents}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate the architecture and return a list of error messages.

        Returns an empty list when the architecture is valid.
        """
        errors: List[str] = []

        agent_ids = self.agent_ids

        # 1. Agent IDs must be unique (Pydantic already enforces list
        #    uniqueness for the model instances, but we check IDs explicitly).
        seen: Set[str] = set()
        for a in self.agents:
            if a.agent_id in seen:
                errors.append(f"Duplicate agent_id: {a.agent_id}")
            seen.add(a.agent_id)

        # 2. Every communication edge must reference existing agents.
        for e in self.communication_edges:
            if e.source not in agent_ids:
                errors.append(
                    f"CommunicationEdge source '{e.source}' does not reference "
                    f"any defined agent"
                )
            if e.target not in agent_ids:
                errors.append(
                    f"CommunicationEdge target '{e.target}' does not reference "
                    f"any defined agent"
                )

        # 3. Self-communication is disallowed unless the edge explicitly
        #    opts into it (we treat it as an error for the static baseline).
        for e in self.communication_edges:
            if e.is_self_loop():
                errors.append(
                    f"Self-loop communication edge not allowed: "
                    f"{e.source} -> {e.target}"
                )

        # 4. Core agents must be present.
        required_core = {"planner", "researcher", "coder", "critic", "finalizer"}
        missing_core = required_core - agent_ids
        if missing_core:
            errors.append(
                f"Missing required core agents: {sorted(missing_core)}"
            )

        # 5. Active status must be valid booleans (already guaranteed by
        #    Pydantic, but we document the rule here).
        #    (No additional check needed — Pydantic guarantees bool.)

        return errors

    @property
    def is_valid(self) -> bool:
        """Return True when validate() produces no errors."""
        return len(self.validate()) == 0

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def serialize(self) -> Dict:
        """Serialize the architecture to a plain dict suitable for logging,
        storage, or later conversion to an RL state representation.

        No secrets are included.
        """
        return {
            "architecture_id": self.architecture_id,
            "agent_count": self.agent_count,
            "active_agent_count": self.active_agent_count,
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "role": a.role,
                    "description": a.description,
                    "capabilities": a.capabilities,
                    "active": a.active,
                    "metadata": a.metadata,
                }
                for a in self.agents
            ],
            "communication_edges": [
                {
                    "source": e.source,
                    "target": e.target,
                }
                for e in self.communication_edges
            ],
            "metadata": self.metadata,
        }
