import pytest

from app.architecture.manager import ArchitectureManager
from app.architecture.models import (
    AgentDefinition,
    CommunicationEdge,
    MASArchitecture,
)


# ---------------------------------------------------------------------------
# AgentDefinition tests
# ---------------------------------------------------------------------------

def test_agent_definition_minimal() -> None:
    a = AgentDefinition(
        agent_id="planner",
        role="planning",
        description="Plans tasks.",
    )
    assert a.agent_id == "planner"
    assert a.role == "planning"
    assert a.description == "Plans tasks."
    assert a.capabilities == []
    assert a.active is True
    assert a.metadata == {}


def test_agent_definition_with_fields() -> None:
    a = AgentDefinition(
        agent_id="coder",
        role="implementation",
        description="Writes code.",
        capabilities=["coding", "testing"],
        active=True,
        metadata={"version": "1"},
    )
    assert a.capabilities == ["coding", "testing"]
    assert a.metadata == {"version": "1"}


def test_agent_definition_validation_empty_id() -> None:
    with pytest.raises(ValueError, match="agent_id"):
        AgentDefinition(
            agent_id="",
            role="r",
            description="d",
        )


def test_agent_definition_validation_empty_role() -> None:
    with pytest.raises(ValueError, match="role"):
        AgentDefinition(
            agent_id="x",
            role="",
            description="d",
        )


def test_agent_definition_validation_whitespace_id() -> None:
    a = AgentDefinition(
        agent_id="  planner  ",
        role="planning",
        description="Plans.",
    )
    assert a.agent_id == "planner"


# ---------------------------------------------------------------------------
# CommunicationEdge tests
# ---------------------------------------------------------------------------

def test_communication_edge_basic() -> None:
    e = CommunicationEdge(source="planner", target="researcher")
    assert e.source == "planner"
    assert e.target == "researcher"
    assert not e.is_self_loop()


def test_communication_edge_self_loop() -> None:
    e = CommunicationEdge(source="planner", target="planner")
    assert e.is_self_loop()


def test_communication_edge_validation_empty() -> None:
    with pytest.raises(ValueError, match="source"):
        CommunicationEdge(source="", target="x")
    with pytest.raises(ValueError, match="target"):
        CommunicationEdge(source="x", target="")


# ---------------------------------------------------------------------------
# MASArchitecture tests
# ---------------------------------------------------------------------------

def test_mas_architecture_empty() -> None:
    arch = MASArchitecture()
    assert arch.agents == []
    assert arch.communication_edges == []
    assert arch.agent_count == 0
    assert arch.active_agent_count == 0
    assert arch.is_valid is False  # missing core agents


def test_mas_architecture_agent_count() -> None:
    agents = [
        AgentDefinition(agent_id=f"agent-{i}", role="r", description="d")
        for i in range(3)
    ]
    arch = MASArchitecture(agents=agents)
    assert arch.agent_count == 3


def test_mas_architecture_active_agents() -> None:
    agents = [
        AgentDefinition(agent_id="a", role="r", description="d", active=True),
        AgentDefinition(agent_id="b", role="r", description="d", active=False),
    ]
    arch = MASArchitecture(agents=agents)
    assert arch.active_agent_count == 1
    assert [a.agent_id for a in arch.active_agents] == ["a"]


def test_mas_architecture_agent_ids() -> None:
    agents = [
        AgentDefinition(agent_id="x", role="r", description="d"),
        AgentDefinition(agent_id="y", role="r", description="d"),
    ]
    arch = MASArchitecture(agents=agents)
    assert arch.agent_ids == {"x", "y"}


def test_mas_architecture_role_map() -> None:
    agents = [
        AgentDefinition(agent_id="planner", role="planning", description="d"),
        AgentDefinition(agent_id="coder", role="implementation", description="d"),
    ]
    arch = MASArchitecture(agents=agents)
    assert arch.role_map == {"planner": "planning", "coder": "implementation"}


# ---------------------------------------------------------------------------
# Architecture validation tests
# ---------------------------------------------------------------------------

def test_valid_architecture_is_valid() -> None:
    arch = MASArchitecture(
        agents=[
            AgentDefinition(agent_id="planner", role="planning", description="d"),
            AgentDefinition(agent_id="researcher", role="research", description="d"),
            AgentDefinition(agent_id="coder", role="implementation", description="d"),
            AgentDefinition(agent_id="critic", role="verification", description="d"),
            AgentDefinition(agent_id="finalizer", role="synthesis", description="d"),
        ],
        communication_edges=[
            CommunicationEdge(source="planner", target="researcher"),
            CommunicationEdge(source="planner", target="coder"),
            CommunicationEdge(source="researcher", target="critic"),
            CommunicationEdge(source="coder", target="critic"),
            CommunicationEdge(source="critic", target="finalizer"),
        ],
    )
    assert arch.is_valid is True
    assert arch.validate() == []


def test_missing_core_agent_invalid() -> None:
    arch = MASArchitecture(
        agents=[
            AgentDefinition(agent_id="planner", role="planning", description="d"),
            AgentDefinition(agent_id="coder", role="implementation", description="d"),
        ],
        communication_edges=[],
    )
    errors = arch.validate()
    assert any("Missing required core agents" in e for e in errors)
    assert arch.is_valid is False


def test_duplicate_agent_id_invalid() -> None:
    agents = [
        AgentDefinition(agent_id="planner", role="planning", description="d"),
        AgentDefinition(agent_id="planner", role="planning", description="d2"),
        AgentDefinition(agent_id="coder", role="implementation", description="d"),
        AgentDefinition(agent_id="researcher", role="research", description="d"),
        AgentDefinition(agent_id="critic", role="verification", description="d"),
        AgentDefinition(agent_id="finalizer", role="synthesis", description="d"),
    ]
    arch = MASArchitecture(agents=agents, communication_edges=[])
    assert any("Duplicate agent_id" in e for e in arch.validate())
    assert arch.is_valid is False


def test_edge_missing_source_invalid() -> None:
    arch = MASArchitecture(
        agents=[
            AgentDefinition(agent_id="planner", role="planning", description="d"),
            AgentDefinition(agent_id="coder", role="implementation", description="d"),
            AgentDefinition(agent_id="researcher", role="research", description="d"),
            AgentDefinition(agent_id="critic", role="verification", description="d"),
            AgentDefinition(agent_id="finalizer", role="synthesis", description="d"),
        ],
        communication_edges=[
            CommunicationEdge(source="planner", target="coder"),
            CommunicationEdge(source="ghost", target="critic"),
        ],
    )
    errors = arch.validate()
    assert any("ghost" in e for e in errors)
    assert arch.is_valid is False


def test_self_loop_invalid() -> None:
    arch = MASArchitecture(
        agents=[
            AgentDefinition(agent_id="planner", role="planning", description="d"),
            AgentDefinition(agent_id="coder", role="implementation", description="d"),
            AgentDefinition(agent_id="researcher", role="research", description="d"),
            AgentDefinition(agent_id="critic", role="verification", description="d"),
            AgentDefinition(agent_id="finalizer", role="synthesis", description="d"),
        ],
        communication_edges=[
            CommunicationEdge(source="planner", target="planner"),
        ],
    )
    errors = arch.validate()
    assert any("Self-loop" in e for e in errors)
    assert arch.is_valid is False


# ---------------------------------------------------------------------------
# ArchitectureManager tests
# ---------------------------------------------------------------------------

def test_default_architecture_has_all_five_agents() -> None:
    mgr = ArchitectureManager.create_default_architecture()
    ids = {a.agent_id for a in mgr.get_architecture().agents}
    expected = {"planner", "researcher", "coder", "critic", "finalizer"}
    assert ids == expected


def test_default_architecture_roles_correct() -> None:
    mgr = ArchitectureManager.create_default_architecture()
    role_map = mgr.get_architecture().role_map
    assert role_map["planner"] == "planning"
    assert role_map["researcher"] == "research"
    assert role_map["coder"] == "implementation"
    assert role_map["critic"] == "verification"
    assert role_map["finalizer"] == "synthesis"


def test_default_architecture_has_five_edges() -> None:
    mgr = ArchitectureManager.create_default_architecture()
    edges = mgr.get_communication_edges()
    assert len(edges) == 6  # planner->researcher, planner->coder,
    # researcher->critic, coder->critic, critic->finalizer, planner->finalizer


def test_default_architecture_edges_match_expected_topology() -> None:
    mgr = ArchitectureManager.create_default_architecture()
    edges = {(e.source, e.target) for e in mgr.get_communication_edges()}
    expected = {
        ("planner", "researcher"),
        ("planner", "coder"),
        ("researcher", "critic"),
        ("coder", "critic"),
        ("critic", "finalizer"),
        ("planner", "finalizer"),
    }
    assert edges == expected


def test_default_architecture_is_valid() -> None:
    mgr = ArchitectureManager.create_default_architecture()
    assert mgr.is_valid is True
    assert mgr.validate() == []


def test_get_agent_by_id() -> None:
    mgr = ArchitectureManager.create_default_architecture()
    a = mgr.get_agent("planner")
    assert a.agent_id == "planner"
    assert a.role == "planning"


def test_get_agent_unknown_raises() -> None:
    mgr = ArchitectureManager.create_default_architecture()
    with pytest.raises(KeyError, match="No agent with id"):
        mgr.get_agent("nonexistent")


def test_get_active_agents_all_active() -> None:
    mgr = ArchitectureManager.create_default_architecture()
    active = mgr.get_active_agents()
    assert len(active) == 5
    assert all(a.active for a in active)


def test_get_active_agent_ids() -> None:
    mgr = ArchitectureManager.create_default_architecture()
    ids = mgr.get_active_agent_ids()
    assert set(ids) == {"planner", "researcher", "coder", "critic", "finalizer"}


def test_get_role() -> None:
    mgr = ArchitectureManager.create_default_architecture()
    assert mgr.get_role("planner") == "planning"
    assert mgr.get_role("coder") == "implementation"


def test_get_capabilities() -> None:
    mgr = ArchitectureManager.create_default_architecture()
    caps = mgr.get_capabilities("coder")
    assert "coding" in caps
    assert "testing" in caps


def test_serialize_architecture() -> None:
    mgr = ArchitectureManager.create_default_architecture()
    data = mgr.serialize()
    assert isinstance(data, dict)
    assert data["architecture_id"] == "static-mas-v1"
    assert data["agent_count"] == 5
    assert data["active_agent_count"] == 5
    assert len(data["agents"]) == 5
    assert len(data["communication_edges"]) == 6
    # No secrets.
    blob = str(data)
    assert "sk-or-" not in blob
    assert "sk-proj-" not in blob
    # Each agent has the required fields.
    for agent in data["agents"]:
        for key in ("agent_id", "role", "description", "capabilities", "active", "metadata"):
            assert key in agent
    # Each edge has source/target.
    for edge in data["communication_edges"]:
        assert "source" in edge
        assert "target" in edge


def test_serialize_does_not_include_credentials() -> None:
    mgr = ArchitectureManager.create_default_architecture()
    data = mgr.serialize()
    blob = str(data)
    assert "sk-or-" not in blob
    assert "sk-proj-" not in blob
    assert "api_key" not in blob.lower()
