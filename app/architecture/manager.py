from __future__ import annotations

from typing import List

from app.architecture.models import (
    AgentDefinition,
    CommunicationEdge,
    MASArchitecture,
)


# ---------------------------------------------------------------------------
# Default static MAS topology (matches the Step 4 LangGraph workflow)
# ---------------------------------------------------------------------------

# The Step 4 LangGraph workflow has these agent nodes:
#   planner, researcher, coder, critic, finalizer
# plus a no-op join_after_plan node that is only a graph routing artifact,
# not an agent. The architecture representation describes agents, not
# LangGraph plumbing nodes, so join_after_plan is excluded.

# Actual agent-to-agent communication derived from the Step 4 workflow:
#   planner -> researcher   (when requires_research)
#   planner -> coder        (when requires_coding)
#   planner -> finalizer    (when neither research nor coding is needed,
#                            the planner's plan goes through join point to
#                            critic/finalizer; architecturally planner feeds
#                            the critic/finalizer pipeline)
#   researcher -> critic     (after research, proceed to critic)
#   coder -> critic          (after coding, proceed to critic)
#   critic -> finalizer      (after verification, proceed to finalizer)
#
# Note: researcher and coder do NOT communicate directly with each other in
# the current static baseline. They both feed into the critic via the join
# point.

_DEFAULT_AGENTS: List[AgentDefinition] = [
    AgentDefinition(
        agent_id="planner",
        role="planning",
        description=(
            "Decomposes the user task into a structured plan, determining which "
            "capabilities are needed and which downstream agents should run."
        ),
        capabilities=["planning", "task_analysis", "routing"],
    ),
    AgentDefinition(
        agent_id="researcher",
        role="research",
        description=(
            "Researches the given task or question and returns a structured "
            "report with findings, evidence, and uncertainties."
        ),
        capabilities=["research", "information_synthesis"],
    ),
    AgentDefinition(
        agent_id="coder",
        role="implementation",
        description=(
            "Writes code to solve a programming task and returns a structured "
            "result with implementation, explanation, and testing notes."
        ),
        capabilities=["coding", "code_explanation", "testing"],
    ),
    AgentDefinition(
        agent_id="critic",
        role="verification",
        description=(
            "Critically reviews intermediate outputs for correctness, "
            "completeness, logic, and quality, and provides issues and "
            "corrections."
        ),
        capabilities=["verification", "critique", "quality_assessment"],
    ),
    AgentDefinition(
        agent_id="finalizer",
        role="synthesis",
        description=(
            "Synthesizes the intermediate agent outputs into a clear final "
            "answer to the original task."
        ),
        capabilities=["synthesis", "final_answer_generation"],
    ),
]

_DEFAULT_EDGES: List[CommunicationEdge] = [
    # Planner dispatches to downstream specialist agents.
    CommunicationEdge(source="planner", target="researcher"),
    CommunicationEdge(source="planner", target="coder"),
    # Specialist agents feed into the critic for verification.
    CommunicationEdge(source="researcher", target="critic"),
    CommunicationEdge(source="coder", target="critic"),
    # Critic feeds into the finalizer for synthesis.
    CommunicationEdge(source="critic", target="finalizer"),
    # Planner can also feed directly into the finalizer pipeline when no
    # research or coding is required (architectural representation of the
    # planner -> join -> critic/finalizer path when no specialists run).
    CommunicationEdge(source="planner", target="finalizer"),
]


# ---------------------------------------------------------------------------
# Architecture manager
# ---------------------------------------------------------------------------

class ArchitectureManager:
    """Manages the (static, current) MAS architecture representation.

    Responsibilities:
      - Create and hold the architecture model.
      - Validate the architecture.
      - Query agents, edges, and active agents.
      - Serialize the architecture for logging / later RL state conversion.

    This manager does NOT execute the LangGraph workflow and does NOT
    mutate the architecture dynamically. Those belong to later steps.
    """

    def __init__(self, architecture: MASArchitecture) -> None:
        self._architecture = architecture

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create_default_architecture(cls) -> ArchitectureManager:
        """Create an ArchitectureManager for the current static MAS.

        The default architecture represents the Step 4 static baseline:
          planner, researcher, coder, critic, finalizer
        with the communication topology derived from the actual LangGraph
        workflow in app/graph/workflow.py.
        """
        architecture = MASArchitecture(
            architecture_id="static-mas-v1",
            agents=_DEFAULT_AGENTS,
            communication_edges=_DEFAULT_EDGES,
            metadata={
                "description": "Static baseline MAS for adaptive-mas research.",
                "version": "1.0",
                "source": "app/architecture/manager.py",
            },
        )
        return cls(architecture=architecture)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_architecture(self) -> MASArchitecture:
        """Return the current architecture model."""
        return self._architecture

    def get_agent(self, agent_id: str) -> AgentDefinition:
        """Return the agent definition for *agent_id*.

        Raises:
            KeyError: If no agent with *agent_id* exists.
        """
        for a in self._architecture.agents:
            if a.agent_id == agent_id:
                return a
        raise KeyError(f"No agent with id '{agent_id}'")

    def get_active_agents(self) -> List[AgentDefinition]:
        """Return the list of agents that are currently active."""
        return self._architecture.active_agents

    def get_active_agent_ids(self) -> List[str]:
        """Return the list of currently active agent IDs."""
        return list(self._architecture.active_agent_ids)

    def get_communication_edges(self) -> List[CommunicationEdge]:
        """Return the list of communication edges in the architecture."""
        return list(self._architecture.communication_edges)

    def get_role(self, agent_id: str) -> str:
        """Return the role of the agent with *agent_id*."""
        return self.get_agent(agent_id).role

    def get_capabilities(self, agent_id: str) -> List[str]:
        """Return the capabilities of the agent with *agent_id*."""
        return list(self.get_agent(agent_id).capabilities)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate the current architecture.

        Returns an empty list when valid.
        """
        return self._architecture.validate()

    @property
    def is_valid(self) -> bool:
        """Return True when the current architecture is valid."""
        return self._architecture.is_valid

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self) -> dict:
        """Serialize the architecture to a plain dict (no secrets)."""
        return self._architecture.serialize()

    def to_architecture_model(self) -> MASArchitecture:
        """Return a copy of the underlying MASArchitecture."""
        return self._architecture.model_copy()
