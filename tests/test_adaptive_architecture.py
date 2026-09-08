import copy

import pytest

from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.adaptive import AdaptiveArchitecture, TransitionRecord
from app.architecture.manager import ArchitectureManager
from app.architecture.models import MASArchitecture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_manager() -> ArchitectureManager:
    return ArchitectureManager.create_default_architecture()


def _default_adaptive() -> AdaptiveArchitecture:
    return AdaptiveArchitecture(manager=_default_manager())


# ---------------------------------------------------------------------------
# 1. Initialization from default architecture
# ---------------------------------------------------------------------------


def test_initialization_from_default_architecture() -> None:
    aa = _default_adaptive()
    arch = aa.current_architecture()
    assert arch.architecture_id == "static-mas-v1"
    assert {a.agent_id for a in arch.agents} == {
        "planner",
        "researcher",
        "coder",
        "critic",
        "finalizer",
    }
    assert aa.version() == 0
    assert aa.history() == []


# ---------------------------------------------------------------------------
# 2. Current architecture retrieval
# ---------------------------------------------------------------------------


def test_current_architecture_retrieval() -> None:
    aa = _default_adaptive()
    arch = aa.get_current_architecture()
    assert arch.is_valid is True
    assert arch.agent_count == 5
    assert arch.active_agent_count == 5


def test_current_architecture_is_independent_copy() -> None:
    aa = _default_adaptive()
    a1 = aa.current_architecture()
    a2 = aa.current_architecture()
    assert a1 is not a2
    assert a1.model_dump() == a2.model_dump()


# ---------------------------------------------------------------------------
# 3. Successful ADD_EDGE transition
# ---------------------------------------------------------------------------


def test_successful_add_edge_transition() -> None:
    aa = _default_adaptive()
    initial_edges = {(e.source, e.target) for e in aa.current_architecture().communication_edges}
    # planner -> critic does not exist in the default architecture.
    action = ArchitectureAction(
        action_type=ActionType.ADD_EDGE,
        source="planner",
        target="critic",
    )
    new_arch = aa.step(action)

    assert new_arch.is_valid is True
    edges = {(e.source, e.target) for e in new_arch.communication_edges}
    assert ("planner", "critic") in edges
    assert edges == initial_edges | {("planner", "critic")}


def test_add_edge_increases_version() -> None:
    aa = _default_adaptive()
    aa.step(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    assert aa.version() == 1


# ---------------------------------------------------------------------------
# 4. Successful REMOVE_EDGE transition
# ---------------------------------------------------------------------------


def test_successful_remove_edge_transition() -> None:
    aa = _default_adaptive()
    # Remove planner -> finalizer (exists in default).
    action = ArchitectureAction(
        action_type=ActionType.REMOVE_EDGE,
        source="planner",
        target="finalizer",
    )
    new_arch = aa.step(action)

    assert new_arch.is_valid is True
    edges = {(e.source, e.target) for e in new_arch.communication_edges}
    assert ("planner", "finalizer") not in edges


def test_remove_edge_that_does_not_exist_is_rejected() -> None:
    aa = _default_adaptive()
    action = ArchitectureAction(
        action_type=ActionType.REMOVE_EDGE,
        source="planner",
        target="ghost",
    )
    with pytest.raises(ValueError, match="Transition rejected"):
        aa.step(action)


# ---------------------------------------------------------------------------
# 5. Successful CHANGE_ROLE transition
# ---------------------------------------------------------------------------


def test_successful_change_role_transition() -> None:
    aa = _default_adaptive()
    action = ArchitectureAction(
        action_type=ActionType.CHANGE_ROLE,
        agent_id="coder",
        new_role="analysis",
    )
    new_arch = aa.step(action)

    assert new_arch.is_valid is True
    coder = next(a for a in new_arch.agents if a.agent_id == "coder")
    assert coder.role == "analysis"


def test_change_role_to_same_role_is_allowed() -> None:
    aa = _default_adaptive()
    action = ArchitectureAction(
        action_type=ActionType.CHANGE_ROLE,
        agent_id="coder",
        new_role="implementation",
    )
    # Changing to the same role is allowed (it's a no-op but valid).
    new_arch = aa.step(action)
    assert new_arch.is_valid is True
    coder = next(a for a in new_arch.agents if a.agent_id == "coder")
    assert coder.role == "implementation"


# ---------------------------------------------------------------------------
# 6. Successful ACTIVATE_AGENT transition
# ---------------------------------------------------------------------------


def test_successful_activate_agent_transition() -> None:
    aa = _default_adaptive()
    # First deactivate to create an inactive agent.
    aa.step(
        ArchitectureAction(
            action_type=ActionType.DEACTIVATE_AGENT,
            agent_id="finalizer",
        )
    )
    # Then reactivate.
    action = ArchitectureAction(
        action_type=ActionType.ACTIVATE_AGENT,
        agent_id="finalizer",
    )
    new_arch = aa.step(action)

    assert new_arch.is_valid is True
    finalizer = next(a for a in new_arch.agents if a.agent_id == "finalizer")
    assert finalizer.active is True


def test_activate_already_active_agent_is_rejected() -> None:
    aa = _default_adaptive()
    action = ArchitectureAction(
        action_type=ActionType.ACTIVATE_AGENT,
        agent_id="planner",
    )
    with pytest.raises(ValueError, match="Transition rejected"):
        aa.step(action)


# ---------------------------------------------------------------------------
# 7. Successful DEACTIVATE_AGENT transition
# ---------------------------------------------------------------------------


def test_successful_deactivate_agent_transition() -> None:
    aa = _default_adaptive()
    action = ArchitectureAction(
        action_type=ActionType.DEACTIVATE_AGENT,
        agent_id="finalizer",
    )
    new_arch = aa.step(action)

    assert new_arch.is_valid is True
    finalizer = next(a for a in new_arch.agents if a.agent_id == "finalizer")
    assert finalizer.active is False


def test_deactivate_last_active_agent_is_rejected() -> None:
    aa = _default_adaptive()
    # Deactivate all but one agent, then try to deactivate the last one.
    for agent_id in ["researcher", "coder", "critic", "finalizer"]:
        aa.step(
            ArchitectureAction(
                action_type=ActionType.DEACTIVATE_AGENT,
                agent_id=agent_id,
            )
        )
    # Now only planner is active. Deactivating planner must be rejected.
    action = ArchitectureAction(
        action_type=ActionType.DEACTIVATE_AGENT,
        agent_id="planner",
    )
    with pytest.raises(ValueError, match="Transition rejected"):
        aa.step(action)


# ---------------------------------------------------------------------------
# 8. Multiple sequential transitions
# ---------------------------------------------------------------------------


def test_multiple_sequential_transitions() -> None:
    aa = _default_adaptive()

    # A0 -> A1: ADD_EDGE planner -> critic
    a1 = aa.step(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    assert ("planner", "critic") in {
        (e.source, e.target) for e in a1.communication_edges
    }

    # A1 -> A2: CHANGE_ROLE coder -> analysis
    a2 = aa.step(
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        )
    )
    coder = next(a for a in a2.agents if a.agent_id == "coder")
    assert coder.role == "analysis"

    # A2 -> A3: REMOVE_EDGE coder -> critic
    a3 = aa.step(
        ArchitectureAction(
            action_type=ActionType.REMOVE_EDGE,
            source="coder",
            target="critic",
        )
    )
    edges = {(e.source, e.target) for e in a3.communication_edges}
    assert ("coder", "critic") not in edges

    assert aa.version() == 3
    assert len(aa.history()) == 3


# ---------------------------------------------------------------------------
# 9. Architecture version changes after successful transition
# ---------------------------------------------------------------------------


def test_version_increments_after_each_successful_transition() -> None:
    aa = _default_adaptive()
    assert aa.version() == 0

    aa.step(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    assert aa.version() == 1

    aa.step(
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        )
    )
    assert aa.version() == 2

    aa.step(
        ArchitectureAction(
            action_type=ActionType.REMOVE_EDGE,
            source="coder",
            target="critic",
        )
    )
    assert aa.version() == 3


# ---------------------------------------------------------------------------
# 10. Transition history is recorded
# ---------------------------------------------------------------------------


def test_transition_history_is_recorded() -> None:
    aa = _default_adaptive()
    action = ArchitectureAction(
        action_type=ActionType.ADD_EDGE,
        source="planner",
        target="critic",
    )
    aa.step(action)

    history = aa.history()
    assert len(history) == 1
    record = history[0]
    assert isinstance(record, TransitionRecord)
    assert record.step == 1
    assert record.action is action
    assert record.previous_architecture.is_valid is True
    assert record.resulting_architecture.is_valid is True
    assert ("planner", "critic") in {
        (e.source, e.target) for e in record.resulting_architecture.communication_edges
    }


def test_history_records_multiple_transitions() -> None:
    aa = _default_adaptive()
    aa.step(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    aa.step(
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        )
    )
    history = aa.history()
    assert len(history) == 2
    assert history[0].step == 1
    assert history[1].step == 2


# ---------------------------------------------------------------------------
# 11. Invalid action is rejected
# ---------------------------------------------------------------------------


def test_invalid_action_is_rejected() -> None:
    aa = _default_adaptive()
    # Add an edge that already exists.
    action = ArchitectureAction(
        action_type=ActionType.ADD_EDGE,
        source="planner",
        target="researcher",
    )
    with pytest.raises(ValueError, match="Transition rejected"):
        aa.step(action)


def test_invalid_action_type_raises_type_error() -> None:
    aa = _default_adaptive()
    with pytest.raises(TypeError, match="action must be an ArchitectureAction"):
        aa.step("not an action")


# ---------------------------------------------------------------------------
# 12. Failed action does not change architecture
# ---------------------------------------------------------------------------


def test_failed_action_does_not_change_architecture() -> None:
    aa = _default_adaptive()
    initial_arch = aa.current_architecture()
    initial_edges = {(e.source, e.target) for e in initial_arch.communication_edges}

    # Attempt to add a duplicate edge (should fail).
    action = ArchitectureAction(
        action_type=ActionType.ADD_EDGE,
        source="planner",
        target="researcher",
    )
    with pytest.raises(ValueError, match="Transition rejected"):
        aa.step(action)

    # Architecture should be unchanged.
    current_arch = aa.current_architecture()
    current_edges = {(e.source, e.target) for e in current_arch.communication_edges}
    assert current_edges == initial_edges
    assert current_arch.architecture_id == initial_arch.architecture_id


def test_failed_action_does_not_change_architecture_deep_copy() -> None:
    aa = _default_adaptive()
    initial_arch = aa.current_architecture()

    action = ArchitectureAction(
        action_type=ActionType.ADD_EDGE,
        source="planner",
        target="researcher",
    )
    with pytest.raises(ValueError):
        aa.step(action)

    current_arch = aa.current_architecture()
    assert current_arch.model_dump() == initial_arch.model_dump()


# ---------------------------------------------------------------------------
# 13. Failed action does not change history
# ---------------------------------------------------------------------------


def test_failed_action_does_not_change_history() -> None:
    aa = _default_adaptive()
    initial_history_len = len(aa.history())

    action = ArchitectureAction(
        action_type=ActionType.ADD_EDGE,
        source="planner",
        target="researcher",
    )
    with pytest.raises(ValueError):
        aa.step(action)

    assert len(aa.history()) == initial_history_len


# ---------------------------------------------------------------------------
# 14. Failed action does not change version
# ---------------------------------------------------------------------------


def test_failed_action_does_not_change_version() -> None:
    aa = _default_adaptive()
    initial_version = aa.version()

    action = ArchitectureAction(
        action_type=ActionType.ADD_EDGE,
        source="planner",
        target="researcher",
    )
    with pytest.raises(ValueError):
        aa.step(action)

    assert aa.version() == initial_version


# ---------------------------------------------------------------------------
# 15. Reset restores initial architecture
# ---------------------------------------------------------------------------


def test_reset_restores_initial_architecture() -> None:
    aa = _default_adaptive()

    # Apply some transitions.
    aa.step(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    aa.step(
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        )
    )

    # Reset.
    reset_arch = aa.reset()

    # Should be back to initial state.
    assert reset_arch.architecture_id == "static-mas-v1"
    assert reset_arch.agent_count == 5
    assert {a.agent_id for a in reset_arch.agents} == {
        "planner",
        "researcher",
        "coder",
        "critic",
        "finalizer",
    }
    coder = next(a for a in reset_arch.agents if a.agent_id == "coder")
    assert coder.role == "implementation"
    edges = {(e.source, e.target) for e in reset_arch.communication_edges}
    assert ("planner", "critic") not in edges


def test_reset_clears_history_and_version() -> None:
    aa = _default_adaptive()
    aa.step(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    aa.step(
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        )
    )
    aa.reset()

    assert aa.version() == 0
    assert aa.history() == []


def test_reset_enables_replay() -> None:
    aa = _default_adaptive()
    aa.step(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    aa.reset()

    # Can apply the same action again after reset.
    new_arch = aa.step(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    assert ("planner", "critic") in {
        (e.source, e.target) for e in new_arch.communication_edges
    }


# ---------------------------------------------------------------------------
# 16. Snapshot contains expected architecture information
# ---------------------------------------------------------------------------


def test_snapshot_contains_expected_architecture_information() -> None:
    aa = _default_adaptive()
    snapshot = aa.snapshot()

    assert isinstance(snapshot, dict)
    assert snapshot["architecture_id"] == "static-mas-v1"
    assert snapshot["agent_count"] == 5
    assert snapshot["active_agent_count"] == 5
    assert len(snapshot["agents"]) == 5
    assert len(snapshot["communication_edges"]) == 6

    # Each agent has the required fields.
    for agent in snapshot["agents"]:
        for key in ("agent_id", "role", "description", "capabilities", "active", "metadata"):
            assert key in agent

    # Each edge has source/target.
    for edge in snapshot["communication_edges"]:
        assert "source" in edge
        assert "target" in edge


def test_snapshot_reflects_transitions() -> None:
    aa = _default_adaptive()
    initial_snapshot = aa.snapshot()

    aa.step(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    updated_snapshot = aa.snapshot()

    assert updated_snapshot["communication_edges"] != initial_snapshot["communication_edges"]
    assert len(updated_snapshot["communication_edges"]) == len(initial_snapshot["communication_edges"]) + 1


# ---------------------------------------------------------------------------
# 17. Snapshot contains no credentials
# ---------------------------------------------------------------------------


def test_snapshot_contains_no_credentials() -> None:
    aa = _default_adaptive()
    snapshot = aa.snapshot()
    blob = str(snapshot)

    # No API keys or credentials.
    assert "sk-or-" not in blob
    assert "sk-proj-" not in blob
    assert "api_key" not in blob.lower()
    assert "password" not in blob.lower()
    assert "secret" not in blob.lower()


def test_snapshot_after_transitions_contains_no_credentials() -> None:
    aa = _default_adaptive()
    aa.step(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    aa.step(
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        )
    )
    snapshot = aa.snapshot()
    blob = str(snapshot)

    assert "sk-or-" not in blob
    assert "sk-proj-" not in blob
    assert "api_key" not in blob.lower()


# ---------------------------------------------------------------------------
# 18. Every successful architecture remains valid
# ---------------------------------------------------------------------------


def test_successful_transition_produces_valid_architecture() -> None:
    aa = _default_adaptive()

    actions = [
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        ),
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        ),
        ArchitectureAction(
            action_type=ActionType.REMOVE_EDGE,
            source="coder",
            target="critic",
        ),
    ]

    for action in actions:
        new_arch = aa.step(action)
        assert new_arch.is_valid is True, f"Architecture became invalid after {action}"


def test_all_architecture_versions_in_history_are_valid() -> None:
    aa = _default_adaptive()
    aa.step(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    aa.step(
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        )
    )
    aa.step(
        ArchitectureAction(
            action_type=ActionType.REMOVE_EDGE,
            source="coder",
            target="critic",
        )
    )

    for record in aa.history():
        assert record.previous_architecture.is_valid is True
        assert record.resulting_architecture.is_valid is True


# ---------------------------------------------------------------------------
# Example scenario: A0 -> A1 -> A2 transitions
# ---------------------------------------------------------------------------


def test_example_scenario_a0_to_a1_to_a2() -> None:
    """Deterministic scenario from Step 7 example.

    Initial: planner, researcher, coder, critic, finalizer
    Action 1: ADD_EDGE planner -> critic
    Action 2: CHANGE_ROLE coder -> analysis
    Action 3: REMOVE_EDGE coder -> critic
    """
    aa = _default_adaptive()
    a0 = aa.current_architecture()

    # Action 1: ADD_EDGE planner -> critic
    a1 = aa.step(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    assert ("planner", "critic") in {
        (e.source, e.target) for e in a1.communication_edges
    }
    assert a1.is_valid is True

    # Action 2: CHANGE_ROLE coder -> analysis
    a2 = aa.step(
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        )
    )
    coder = next(a for a in a2.agents if a.agent_id == "coder")
    assert coder.role == "analysis"
    assert a2.is_valid is True

    # Action 3: REMOVE_EDGE coder -> critic
    a3 = aa.step(
        ArchitectureAction(
            action_type=ActionType.REMOVE_EDGE,
            source="coder",
            target="critic",
        )
    )
    edges = {(e.source, e.target) for e in a3.communication_edges}
    assert ("coder", "critic") not in edges
    assert a3.is_valid is True

    # Verify history contains all three transitions.
    history = aa.history()
    assert len(history) == 3
    assert history[0].step == 1
    assert history[1].step == 2
    assert history[2].step == 3


def test_architecture_at_step_lookup() -> None:
    aa = _default_adaptive()

    # Apply two transitions.
    aa.step(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    aa.step(
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        )
    )

    # Step 0 is the initial architecture.
    arch0 = aa.architecture_at_step(0)
    assert arch0 is not None
    assert arch0.architecture_id == "static-mas-v1"
    coder0 = next(a for a in arch0.agents if a.agent_id == "coder")
    assert coder0.role == "implementation"

    # Step 1 is after the first transition.
    arch1 = aa.architecture_at_step(1)
    assert arch1 is not None
    edges1 = {(e.source, e.target) for e in arch1.communication_edges}
    assert ("planner", "critic") in edges1

    # Step 2 is after the second transition.
    arch2 = aa.architecture_at_step(2)
    assert arch2 is not None
    coder2 = next(a for a in arch2.agents if a.agent_id == "coder")
    assert coder2.role == "analysis"

    # Out of range returns None.
    assert aa.architecture_at_step(99) is None
    assert aa.architecture_at_step(-1) is None


def test_previous_architecture_lookup() -> None:
    aa = _default_adaptive()

    aa.step(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )

    prev = aa.previous_architecture(1)
    assert prev is not None
    assert prev.architecture_id == "static-mas-v1"
    edges = {(e.source, e.target) for e in prev.communication_edges}
    assert ("planner", "critic") not in edges


# ---------------------------------------------------------------------------
# API aliases
# ---------------------------------------------------------------------------


def test_step_and_apply_action_are_equivalent() -> None:
    aa1 = _default_adaptive()
    aa2 = _default_adaptive()

    action = ArchitectureAction(
        action_type=ActionType.ADD_EDGE,
        source="planner",
        target="critic",
    )

    result1 = aa1.step(action)
    result2 = aa2.apply_action(action)

    assert result1.model_dump() == result2.model_dump()


def test_get_snapshot_and_get_history_aliases() -> None:
    aa = _default_adaptive()
    assert aa.get_snapshot() == aa.snapshot()
    assert aa.get_history() == aa.history()
    assert aa.get_current_architecture().model_dump() == aa.current_architecture().model_dump()
