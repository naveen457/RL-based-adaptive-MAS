"""
Adaptive integration layer between the RL-driven adaptive architecture and the
existing LangGraph MAS workflow.

This module provides AdaptiveWorkflowAdapter, a controlled bridge that:

* Represents the current adaptive architecture independently from the protected
  LangGraph workflow implementation.
* Accepts validated ArchitectureAction objects produced by the existing
  architecture/RL system.
* Applies architecture actions through AdaptiveArchitecture.
* Validates the resulting architecture.
* Produces a structured LangGraph-compatible architecture/configuration
  description from the resulting architecture.
* Clearly separates:
  - architecture state
  - architecture action
  - resulting architecture
  - executable workflow configuration/execution plan
* Provides compatibility checks so callers can determine whether a given
  architecture can be represented/executed by the existing MAS.

Design notes
------------
* This adapter does NOT rebuild or modify the compiled LangGraph graph
  dynamically. It returns a structured description/execution plan that can
  inform downstream execution while keeping app/graph/workflow.py protected.
* The existing build_workflow/run_workflow remain the baseline executable
  workflow. This adapter describes how a candidate architecture maps to the
  existing MAS topology.
* The adapter is deterministic and serializable.
* No LLM calls, no API calls, no agent execution, no dynamic graph compilation.

Scope for this step
-------------------
* Return a structured execution plan/configuration rather than compiling a new
  LangGraph graph.
* Identify unsupported architectures explicitly.
* Preserve the existing workflow as the baseline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.architecture.actions import ArchitectureAction
from app.architecture.adaptive import AdaptiveArchitecture
from app.architecture.manager import ArchitectureManager
from app.architecture.models import (
    AgentDefinition,
    CommunicationEdge,
    MASArchitecture,
)
from app.graph.workflow import build_workflow, run_workflow
from app.graph.state import MASState


# ---------------------------------------------------------------------------
# Workflow configuration / execution plan models
# ---------------------------------------------------------------------------


class WorkflowCompatibilityReport(BaseModel):
    """Compatibility assessment for converting an architecture to a workflow
    configuration that the existing MAS can represent/execute.

    Attributes
    ----------
    compatible : bool
        Whether the architecture is compatible with the existing MAS baseline.
    reason : str
        Human-readable reason for the compatibility status.
    missing_required_agents : List[str]
        Required core agents that are missing from the architecture.
    inactive_required_agents : List[str]
        Required core agents that are present but inactive.
    unsupported_elements : List[str]
        Other unsupported or risky elements identified by the adapter.
    """

    compatible: bool = Field(description="Whether the architecture is compatible")
    reason: str = Field(description="Human-readable reason")
    missing_required_agents: List[str] = Field(
        default_factory=list, description="Missing required core agents"
    )
    inactive_required_agents: List[str] = Field(
        default_factory=list, description="Required agents that are inactive"
    )
    unsupported_elements: List[str] = Field(
        default_factory=list, description="Unsupported or risky elements"
    )


class WorkflowExecutionPlan(BaseModel):
    """A structured description of how a candidate architecture maps to the
    existing MAS workflow.

    This is an execution plan/configuration description, not a dynamically
    compiled LangGraph graph.

    Attributes
    ----------
    architecture_id : str
        Architecture identifier from the source architecture.
    architecture_version : int
        Architecture version after the latest adaptation step.
    plan_description : str
        Human-readable description of the execution plan.
    active_agents : List[str]
        Ordered list of active agent IDs that should participate in execution.
    expected_edges : List[Dict[str, str]]
        Directed edges (source/target) present in the architecture.
    compatibility : WorkflowCompatibilityReport
        Compatibility assessment for the architecture.
    execution_mode : str
        Mode describing how execution should proceed:
        - "baseline" : use existing workflow as-is with current architecture
        - "adapted" : adapted architecture within supported boundaries
        - "unsupported" : architecture cannot be executed by existing MAS
    notes : str
        Additional notes about the plan.
    """

    architecture_id: str = Field(description="Architecture identifier")
    architecture_version: int = Field(description="Architecture version")
    plan_description: str = Field(description="Human-readable execution plan")
    active_agents: List[str] = Field(
        default_factory=list, description="Active agent IDs for execution"
    )
    expected_edges: List[Dict[str, str]] = Field(
        default_factory=list, description="Directed edges in the architecture"
    )
    compatibility: WorkflowCompatibilityReport = Field(
        description="Compatibility assessment"
    )
    execution_mode: str = Field(
        default="baseline",
        description="Execution mode: baseline, adapted, or unsupported",
    )
    notes: str = Field(
        default="",
        description="Additional notes about the plan",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        return self.model_dump(mode="json", exclude_none=True)

    def serialize(self) -> Dict[str, Any]:
        """Alias for to_dict for consistency."""
        return self.to_dict()


# ---------------------------------------------------------------------------
# AdaptiveWorkflowAdapter
# ---------------------------------------------------------------------------


class AdaptiveWorkflowAdapter:
    """Bridge between RL-driven adaptive architecture decisions and the existing
    LangGraph MAS.

    Responsibilities:
    * Hold an AdaptiveArchitecture instance.
    * Accept and apply validated ArchitectureAction objects safely.
    * Validate resulting architectures.
    * Produce a LangGraph-compatible workflow configuration/execution plan from
      the resulting architecture.
    * Provide compatibility checks without rebuilding the compiled graph.

    This adapter does NOT modify app/graph/workflow.py. It returns structured
    descriptions that can inform execution while keeping the existing workflow
    intact.
    """

    # Required core agents that the existing baseline MAS expects.
    REQUIRED_CORE_AGENTS = frozenset({
        "planner",
        "researcher",
        "coder",
        "critic",
        "finalizer",
    })

    def __init__(
        self,
        adaptive: Optional[AdaptiveArchitecture] = None,
        manager: Optional[ArchitectureManager] = None,
    ) -> None:
        """Initialize the adapter.

        Parameters
        ----------
        adaptive : Optional[AdaptiveArchitecture]
            Existing adaptive architecture wrapper. If provided, manager must
            not be provided.
        manager : Optional[ArchitectureManager]
            Architecture manager to wrap. If provided, adaptive must not be
            provided. If neither is provided, the default architecture is used.
        """
        if adaptive is not None and manager is not None:
            raise ValueError(
                "Provide either adaptive or manager, not both"
            )

        if adaptive is not None:
            self._adaptive = adaptive
        elif manager is not None:
            self._adaptive = AdaptiveArchitecture(manager)
        else:
            self._adaptive = AdaptiveArchitecture(
                ArchitectureManager.create_default_architecture()
            )

        # Keep a reference architecture for compatibility snapshots when needed.
        self._baseline_workflow_description = self._describe_baseline_workflow()

    # ------------------------------------------------------------------
    # Current architecture
    # ------------------------------------------------------------------

    def current_architecture(self) -> MASArchitecture:
        """Return the current architecture from the adaptive layer."""
        return self._adaptive.current_architecture()

    def get_current_architecture(self) -> MASArchitecture:
        """Public accessor for the current architecture."""
        return self.current_architecture()

    def current_version(self) -> int:
        """Return the current architecture version."""
        return self._adaptive.version()

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def apply_action(self, action: ArchitectureAction) -> MASArchitecture:
        """Apply a validated architecture action and return the resulting
        architecture.

        Parameters
        ----------
        action : ArchitectureAction
            The action to apply.

        Returns
        -------
        MASArchitecture
            The resulting architecture after the action.

        Raises
        ------
        TypeError
            If action is not an ArchitectureAction.
        ValueError
            If the action is rejected or produces an invalid architecture.
        """
        return self._adaptive.apply_action(action)

    def try_apply_action(
        self, action: ArchitectureAction
    ) -> Tuple[bool, Optional[MASArchitecture], str]:
        """Safely attempt to apply an action without raising.

        Returns (success, resulting_architecture_or_None, error_message).

        Parameters
        ----------
        action : ArchitectureAction
            The action to attempt.

        Returns
        -------
        Tuple[bool, Optional[MASArchitecture], str]
            Success flag, resulting architecture if successful, and error
            message if unsuccessful.
        """
        try:
            result = self.apply_action(action)
            return True, result, ""
        except (TypeError, ValueError) as exc:
            return False, None, str(exc)

    # ------------------------------------------------------------------
    # Reset / rollback
    # ------------------------------------------------------------------

    def reset(self) -> MASArchitecture:
        """Reset to the initial/baseline architecture."""
        return self._adaptive.reset()

    def rollback_to_version(self, version: int) -> MASArchitecture:
        """Rollback to a previously recorded architecture version.

        This is implemented by replaying the initial architecture and
        re-applying transitions up to the requested version. If the
        requested version is out of range, the current architecture is
        returned unchanged and an error message is returned in the second
        element when using the safe variant.

        Parameters
        ----------
        version : int
            Target architecture version (0 = initial).

        Returns
        -------
        MASArchitecture
            Architecture at the requested version.
        """
        if version < 0:
            raise ValueError("version must be >= 0")
        if version == 0:
            return self.reset()
        if version > self._adaptive.version():
            raise ValueError(
                f"Cannot rollback to future version {version}; "
                f"current version is {self._adaptive.version()}"
            )

        # Rebuild from initial and replay transitions up to target version.
        self._adaptive.reset()
        history = self._adaptive.history()
        for record in history:
            if record.step > version:
                break
            # Re-apply the recorded action to reach the target version.
            self._adaptive.apply_action(record.action)

        return self._adaptive.current_architecture()

    # ------------------------------------------------------------------
    # Workflow configuration / execution plan
    # ------------------------------------------------------------------

    def produce_workflow_config(
        self, architecture: Optional[MASArchitecture] = None
    ) -> WorkflowExecutionPlan:
        """Produce a LangGraph-compatible workflow configuration/execution plan
        from the given architecture.

        This does NOT rebuild the compiled LangGraph graph. It returns a
        structured description that the existing MAS can use to interpret the
        architecture.

        Parameters
        ----------
        architecture : Optional[MASArchitecture]
            Architecture to describe. If None, uses the current architecture.

        Returns
        -------
        WorkflowExecutionPlan
            Structured execution plan/configuration.
        """
        arch = architecture if architecture is not None else self.current_architecture()
        compatibility = self._assess_compatibility(arch)

        active_agent_ids = sorted(arch.active_agent_ids)
        edges = [
            {"source": e.source, "target": e.target}
            for e in sorted(
                arch.communication_edges,
                key=lambda e: (e.source, e.target),
            )
        ]

        if not compatibility.compatible:
            execution_mode = "unsupported"
            description = (
                f"Architecture '{arch.architecture_id}' is not compatible with "
                f"the existing MAS baseline. {compatibility.reason}"
            )
            notes = "Use reset() to restore the baseline architecture before execution."
        elif arch.architecture_id == "static-mas-v1" and arch.agent_ids == self.REQUIRED_CORE_AGENTS and arch.active_agent_ids == sorted(self.REQUIRED_CORE_AGENTS) and not compatibility.unsupported_elements:
            execution_mode = "baseline"
            description = (
                f"Baseline MAS workflow configuration for architecture "
                f"'{arch.architecture_id}'. Use existing workflow as-is."
            )
            notes = "Architecture matches the protected baseline topology."
        else:
            execution_mode = "adapted"
            description = (
                f"Adapted MAS workflow configuration for architecture "
                f"'{arch.architecture_id}'. Execution plan reflects current "
                f"active agents and communication topology within supported boundaries."
            )
            notes = (
                "Adapted architecture is represented as a configuration/execution "
                "plan; the existing workflow remains the executable baseline."
            )

        return WorkflowExecutionPlan(
            architecture_id=arch.architecture_id,
            architecture_version=self._adaptive.version(),
            plan_description=description,
            active_agents=active_agent_ids,
            expected_edges=edges,
            compatibility=compatibility,
            execution_mode=execution_mode,
            
            notes=notes,
        )

    def produce_execution_plan_from_action(
        self, action: ArchitectureAction
    ) -> Tuple[bool, Optional[WorkflowExecutionPlan], Optional[str]]:
        """Apply an action, validate the result, and produce an execution plan.

        This is the primary controlled integration entry point for RL-driven
        action application.

        Parameters
        ----------
        action : ArchitectureAction
            The action to apply.

        Returns
        -------
        Tuple[bool, Optional[WorkflowExecutionPlan], Optional[str]]
            Success flag, execution plan if successful, and error message if
            unsuccessful.
        """
        success, architecture, error = self.try_apply_action(action)
        if not success:
            return False, None, error

        if not architecture.is_valid:
            return False, None, "Resulting architecture is invalid"

        plan = self.produce_workflow_config(architecture)
        return True, plan, ""

    # ------------------------------------------------------------------
    # Compatibility checks
    # ------------------------------------------------------------------

    def is_compatible(self, architecture: Optional[MASArchitecture] = None) -> bool:
        """Return True if the architecture is compatible with the existing MAS."""
        return self._assess_compatibility(
            architecture if architecture is not None else self.current_architecture()
        ).compatible

    def get_compatibility_report(
        self, architecture: Optional[MASArchitecture] = None
    ) -> WorkflowCompatibilityReport:
        """Return a detailed compatibility report for the architecture."""
        return self._assess_compatibility(
            architecture if architecture is not None else self.current_architecture()
        )

    def can_execute_baseline_workflow(self) -> bool:
        """Return True if the current architecture can be executed by the
        existing baseline workflow without unsupported changes."""
        return self.is_compatible()

    # ------------------------------------------------------------------
    # Baseline workflow access
    # ------------------------------------------------------------------

    def get_baseline_workflow(self):
        """Return the existing baseline compiled LangGraph workflow.

        This preserves the protected workflow as the executable baseline.
        """
        return build_workflow()

    def run_baseline_workflow(self, task: str) -> MASState:
        """Run the existing baseline workflow on a task.

        This is provided for integration smoke tests to verify the existing
        workflow remains unaffected.
        """
        return run_workflow(task)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a safe snapshot of the current adaptive state."""
        arch = self.current_architecture()
        return {
            "architecture_id": arch.architecture_id,
            "architecture_version": self.current_version(),
            "active_agents": sorted(arch.active_agent_ids),
            "agent_count": arch.agent_count,
            "active_agent_count": arch.active_agent_count,
            "edges": [
                {"source": e.source, "target": e.target}
                for e in arch.communication_edges
            ],
            "is_valid": arch.is_valid,
            "compatible_with_baseline": self.is_compatible(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation of the adapter state."""
        return self.snapshot()

    def serialize(self) -> Dict[str, Any]:
        """Alias for to_dict for consistency."""
        return self.to_dict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assess_compatibility(
        self, architecture: MASArchitecture
    ) -> WorkflowCompatibilityReport:
        """Assess whether an architecture is compatible with the existing MAS."""
        agent_ids = architecture.agent_ids
        active_ids = architecture.active_agent_ids

        missing = sorted(self.REQUIRED_CORE_AGENTS - agent_ids)
        inactive = sorted(
            self.REQUIRED_CORE_AGENTS & (agent_ids - active_ids)
        )

        unsupported: List[str] = []

        # Detect unexpected agents not present in the baseline.
        unexpected = sorted(agent_ids - self.REQUIRED_CORE_AGENTS)
        if unexpected:
            unsupported.append(
                f"Unexpected agents present: {', '.join(unexpected)}"
            )

        # Detect missing edges for required core communication paths.
        existing_edges = {
            (e.source, e.target) for e in architecture.communication_edges
        }
        required_paths = [
            ("planner", "researcher"),
            ("planner", "coder"),
            ("researcher", "critic"),
            ("coder", "critic"),
            ("critic", "finalizer"),
        ]
        missing_edges = [
            f"{s} -> {t}" for s, t in required_paths if (s, t) not in existing_edges
        ]
        if missing_edges:
            unsupported.append(f"Missing required edges: {', '.join(missing_edges)}")

        # Self-loops are disallowed in the baseline.
        self_loops = [
            f"{e.source} -> {e.target}"
            for e in architecture.communication_edges
            if e.source == e.target
        ]
        if self_loops:
            unsupported.append(f"Self-loop edges present: {', '.join(self_loops)}")

        if missing:
            reason = (
                f"Missing required core agents: {', '.join(missing)}. "
                f"The existing MAS requires planner, researcher, coder, critic, "
                f"and finalizer."
            )
            compatible = False
        elif inactive:
            reason = (
                f"Required core agents are inactive: {', '.join(inactive)}. "
                f"The existing MAS requires all core agents to be active."
            )
            compatible = False
        elif unsupported:
            reason = "Unsupported elements detected: " + "; ".join(unsupported)
            compatible = False
        else:
            reason = "Architecture is compatible with the existing MAS baseline."
            compatible = True

        return WorkflowCompatibilityReport(
            compatible=compatible,
            reason=reason,
            missing_required_agents=missing,
            inactive_required_agents=inactive,
            unsupported_elements=unsupported,
        )

    def _describe_baseline_workflow(self) -> Dict[str, Any]:
        """Return a structured description of the baseline workflow for
        reference/comparison purposes.

        This is used internally for compatibility context and does not modify
        the workflow.
        """
        return {
            "workflow_source": "app/graph/workflow.py",
            "build_function": "build_workflow",
            "run_function": "run_workflow",
            "required_core_agents": sorted(self.REQUIRED_CORE_AGENTS),
            "baseline_architecture_id": "static-mas-v1",
            "description": (
                "Existing LangGraph MAS workflow remains the protected baseline. "
                "AdaptiveWorkflowAdapter produces configuration/execution plans "
                "without rebuilding the compiled graph."
            ),
        }
