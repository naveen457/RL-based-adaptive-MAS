from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ActionType(str, Enum):
    """Architecture mutations supported by the Step 6 action space."""

    ACTIVATE_AGENT = "activate_agent"
    DEACTIVATE_AGENT = "deactivate_agent"
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    CHANGE_ROLE = "change_role"


# Descriptive alias for callers that prefer the more explicit name.
ArchitectureActionType = ActionType


class ArchitectureAction(BaseModel):
    """A validated, architecture-level action request.

    This model validates the fields needed to describe an action. Whether the
    referenced agents and edges exist is intentionally left to
    ``ArchitectureManager``, which has access to a concrete architecture.
    """

    model_config = ConfigDict(extra="forbid")

    action_type: ActionType = Field(description="Type of architecture mutation.")
    agent_id: Optional[str] = Field(
        default=None,
        description="Agent affected by an activate, deactivate, or role-change action.",
    )
    source: Optional[str] = Field(
        default=None,
        description="Source agent for an edge action.",
    )
    target: Optional[str] = Field(
        default=None,
        description="Target agent for an edge action.",
    )
    new_role: Optional[str] = Field(
        default=None,
        description="Replacement role for a change-role action.",
    )

    @field_validator("agent_id", "source", "target", "new_role")
    @classmethod
    def optional_strings_must_be_nonempty(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("action fields must be non-empty when provided")
        return value

    @model_validator(mode="after")
    def validate_required_fields(self) -> ArchitectureAction:
        required_fields = {
            ActionType.ACTIVATE_AGENT: ("agent_id",),
            ActionType.DEACTIVATE_AGENT: ("agent_id",),
            ActionType.ADD_EDGE: ("source", "target"),
            ActionType.REMOVE_EDGE: ("source", "target"),
            ActionType.CHANGE_ROLE: ("agent_id", "new_role"),
        }

        missing = [
            field
            for field in required_fields[self.action_type]
            if getattr(self, field) is None
        ]
        if missing:
            fields = ", ".join(missing)
            raise ValueError(
                f"{self.action_type.value} action requires: {fields}"
            )
        return self

    def serialize(self) -> dict[str, str]:
        """Return a JSON-friendly representation with omitted optional fields."""
        return self.model_dump(mode="json", exclude_none=True)
