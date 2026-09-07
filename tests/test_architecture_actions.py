import pytest
from pydantic import ValidationError

from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.manager import ArchitectureManager


def test_each_action_type_can_be_constructed() -> None:
    actions = [
        ArchitectureAction(
            action_type=ActionType.ACTIVATE_AGENT, agent_id="researcher"
        ),
        ArchitectureAction(
            action_type=ActionType.DEACTIVATE_AGENT, agent_id="researcher"
        ),
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE, source="planner", target="critic"
        ),
        ArchitectureAction(
            action_type=ActionType.REMOVE_EDGE, source="coder", target="critic"
        ),
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        ),
    ]

    assert [action.action_type for action in actions] == list(ActionType)


@pytest.mark.parametrize(
    "payload",
    [
        {"action_type": ActionType.ACTIVATE_AGENT},
        {"action_type": ActionType.DEACTIVATE_AGENT},
        {"action_type": ActionType.ADD_EDGE, "source": "planner"},
        {"action_type": ActionType.REMOVE_EDGE, "target": "critic"},
        {"action_type": ActionType.CHANGE_ROLE, "agent_id": "coder"},
        {
            "action_type": ActionType.CHANGE_ROLE,
            "agent_id": "coder",
            "new_role": "   ",
        },
    ],
)
def test_missing_required_action_fields_are_rejected(payload: dict) -> None:
    with pytest.raises(ValidationError):
        ArchitectureAction(**payload)


def test_action_fields_are_trimmed_and_serialized() -> None:
    action = ArchitectureAction(
        action_type="change_role",
        agent_id=" coder ",
        new_role=" analysis ",
    )

    assert action.agent_id == "coder"
    assert action.new_role == "analysis"
    assert action.serialize() == {
        "action_type": "change_role",
        "agent_id": "coder",
        "new_role": "analysis",
    }


def test_unknown_agent_is_rejected() -> None:
    manager = ArchitectureManager.create_default_architecture()
    original = manager.serialize()

    with pytest.raises(ValueError, match="Unknown agent"):
        manager.apply_action(
            ArchitectureAction(
                action_type=ActionType.ACTIVATE_AGENT,
                agent_id="unknown",
            )
        )

    assert manager.serialize() == original


def test_self_loop_is_rejected() -> None:
    manager = ArchitectureManager.create_default_architecture()

    with pytest.raises(ValueError, match="self-loop"):
        manager.apply_action(
            ArchitectureAction(
                action_type=ActionType.ADD_EDGE,
                source="planner",
                target="planner",
            )
        )


def test_duplicate_edge_addition_is_rejected() -> None:
    manager = ArchitectureManager.create_default_architecture()

    with pytest.raises(ValueError, match="duplicate edge"):
        manager.apply_action(
            ArchitectureAction(
                action_type=ActionType.ADD_EDGE,
                source="planner",
                target="researcher",
            )
        )


def test_removing_nonexistent_edge_is_rejected() -> None:
    manager = ArchitectureManager.create_default_architecture()

    with pytest.raises(ValueError, match="nonexistent edge"):
        manager.apply_action(
            ArchitectureAction(
                action_type=ActionType.REMOVE_EDGE,
                source="researcher",
                target="coder",
            )
        )


def test_valid_agent_deactivation_and_activation_work() -> None:
    manager = ArchitectureManager.create_default_architecture()

    deactivated = manager.apply_action(
        ArchitectureAction(
            action_type=ActionType.DEACTIVATE_AGENT,
            agent_id="researcher",
        )
    )
    assert deactivated.active_agent_ids == {
        "planner",
        "coder",
        "critic",
        "finalizer",
    }

    activated = manager.apply_action(
        ArchitectureAction(
            action_type=ActionType.ACTIVATE_AGENT,
            agent_id="researcher",
        )
    )
    assert activated.active_agent_ids == {
        "planner",
        "researcher",
        "coder",
        "critic",
        "finalizer",
    }


def test_valid_edge_addition_and_removal_work() -> None:
    manager = ArchitectureManager.create_default_architecture()

    added = manager.apply_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="researcher",
            target="coder",
        )
    )
    assert ("researcher", "coder") in {
        (edge.source, edge.target) for edge in added.communication_edges
    }

    removed = manager.apply_action(
        ArchitectureAction(
            action_type=ActionType.REMOVE_EDGE,
            source="researcher",
            target="coder",
        )
    )
    assert ("researcher", "coder") not in {
        (edge.source, edge.target) for edge in removed.communication_edges
    }


def test_valid_role_change_works() -> None:
    manager = ArchitectureManager.create_default_architecture()

    updated = manager.apply_action(
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="analysis",
        )
    )

    assert updated.role_map["coder"] == "analysis"
    assert updated.is_valid


def test_failed_action_does_not_modify_original_architecture() -> None:
    manager = ArchitectureManager.create_default_architecture()
    original = manager.to_architecture_model()

    with pytest.raises(ValueError):
        manager.apply_action(
            ArchitectureAction(
                action_type=ActionType.ADD_EDGE,
                source="planner",
                target="planner",
            )
        )

    assert manager.get_architecture() == original


def test_resulting_architecture_is_validated() -> None:
    manager = ArchitectureManager.create_default_architecture()
    updated = manager.apply_action(
        ArchitectureAction(
            action_type=ActionType.DEACTIVATE_AGENT,
            agent_id="planner",
        )
    )

    assert updated.is_valid
    assert manager.is_valid


def test_last_active_agent_cannot_be_deactivated() -> None:
    manager = ArchitectureManager.create_default_architecture()
    for agent_id in ("researcher", "coder", "critic", "finalizer"):
        manager.apply_action(
            ArchitectureAction(
                action_type=ActionType.DEACTIVATE_AGENT,
                agent_id=agent_id,
            )
        )

    original = manager.serialize()
    with pytest.raises(ValueError, match="last active agent"):
        manager.apply_action(
            ArchitectureAction(
                action_type=ActionType.DEACTIVATE_AGENT,
                agent_id="planner",
            )
        )

    assert manager.serialize() == original


def test_possible_actions_are_deterministic_and_include_role_options() -> None:
    manager = ArchitectureManager.create_default_architecture()
    first = [action.serialize() for action in manager.get_possible_actions(["analysis"])]
    second = [action.serialize() for action in manager.get_possible_actions(["analysis"])]

    assert first == second
    assert {
        "action_type": "add_edge",
        "source": "researcher",
        "target": "coder",
    } in first
    assert {
        "action_type": "change_role",
        "agent_id": "coder",
        "new_role": "analysis",
    } in first
    assert {
        "action_type": "remove_edge",
        "source": "coder",
        "target": "critic",
    } in first
