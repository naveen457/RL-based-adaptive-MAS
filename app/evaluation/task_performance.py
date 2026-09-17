"""
Deterministic task-performance evaluation for the adaptive MAS research project.

This module provides TaskPerformanceEvaluator, a deterministic architecture-level
task-performance evaluation abstraction that measures how well a MAS architecture
matches a specific task's requirements.

This is NOT actual LLM task execution. The scores are architecture-task compatibility
proxies based on:

* Which required capabilities are represented by active agents' roles
* Whether the architecture has reasonable communication connectivity
* How many required capabilities are covered vs missing

The evaluator:

* Accepts MetaTask and MASArchitecture
* Returns deterministic TaskPerformanceResult
* Uses scores in the range 0.0-1.0
* Does NOT call LLMs, execute agents, or make HTTP requests
* Is fully offline and reproducible

Design notes
------------
* This module extends the evaluation layer without replacing the structural evaluator.
* The existing Step 9/10 structural metrics remain available.
* Combined evaluation makes it possible to distinguish:
  - Architecture structurally valid
  - Architecture suitable for the specific task
* This is a deterministic architecture-task compatibility proxy for task performance.
  It is NOT actual LLM task execution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from app.architecture.models import MASArchitecture
from app.rl.meta_task import MetaTask


# ============================================================================
# Role-to-Capability Mapping
# ============================================================================

# This mapping defines which capabilities each agent role provides.
# It is based on the actual agent roles in the default architecture:
# - planner: planning
# - researcher: research
# - coder: implementation (coding)
# - critic: verification
# - finalizer: synthesis

# Each role provides a set of capabilities.
# This mapping is used to determine capability coverage for a task.

ROLE_CAPABILITY_MAP: Dict[str, Set[str]] = {
    "planning": {"planning", "analysis"},
    "research": {"research", "information_synthesis"},
    "implementation": {"coding", "implementation"},
    "verification": {"verification", "testing", "analysis"},
    "synthesis": {"synthesis", "summarization"},
    "tool_execution": {"tool_use", "web_search", "external_api"},
}


def get_capabilities_for_role(role: str) -> Set[str]:
    """
    Return the set of capabilities that an agent with the given role provides.

    Parameters
    ----------
    role : str
        The agent role name.

    Returns
    -------
    Set[str]
        The set of capabilities provided by this role.
    """
    return ROLE_CAPABILITY_MAP.get(role, set())


def get_roles_for_capability(capability: str) -> Set[str]:
    """
    Return the set of roles that can provide a given capability.

    Parameters
    ----------
    capability : str
        The capability name.

    Returns
    -------
    Set[str]
        The set of roles that provide this capability.
    """
    roles = set()
    for role, capabilities in ROLE_CAPABILITY_MAP.items():
        if capability in capabilities:
            roles.add(role)
    return roles


# ============================================================================
# Result Models
# ============================================================================


class TaskPerformanceResult(BaseModel):
    """
    Deterministic task-performance evaluation result.

    This result represents an architecture-task compatibility proxy, NOT
    actual LLM task execution. The scores indicate how well the architecture
    matches the task requirements based on role/capability mapping.

    All scores are in the range [0.0, 1.0].

    Attributes
    ----------
    task_id : str
        The MetaTask.task_id.
    architecture_id : str
        The MASArchitecture.architecture_id.
    task_success_score : float
        Overall task success proxy score in [0.0, 1.0].
    capability_coverage_score : float
        Fraction of required capabilities covered by active agents in [0.0, 1.0].
    task_alignment_score : float
        How well the architecture aligns with the task category in [0.0, 1.0].
    missing_capabilities : List[str]
        Required capabilities not covered by any active agent.
    covered_capabilities : List[str]
        Required capabilities covered by at least one active agent.
    active_agent_count : int
        Number of currently active agents.
    communication_edge_count : int
        Number of directed communication edges.
    notes : str
        Human-readable notes about the evaluation.
    """

    task_id: str = Field(description="MetaTask.task_id")
    architecture_id: str = Field(description="MASArchitecture.architecture_id")
    task_success_score: float = Field(
        description="Overall task success proxy score in [0.0, 1.0]"
    )
    capability_coverage_score: float = Field(
        description="Fraction of required capabilities covered in [0.0, 1.0]"
    )
    task_alignment_score: float = Field(
        description="Task-architecture alignment score in [0.0, 1.0]"
    )
    missing_capabilities: List[str] = Field(
        default_factory=list,
        description="Required capabilities not covered by active agents",
    )
    covered_capabilities: List[str] = Field(
        default_factory=list,
        description="Required capabilities covered by active agents",
    )
    active_agent_count: int = Field(
        description="Number of currently active agents"
    )
    communication_edge_count: int = Field(
        description="Number of directed communication edges"
    )
    notes: str = Field(
        default="",
        description="Human-readable notes for the evaluation",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        return self.model_dump(mode="json", exclude_none=True)

    def serialize(self) -> Dict[str, Any]:
        """Alias for to_dict for consistency with other result models."""
        return self.to_dict()


class CombinedEvaluationResult(BaseModel):
    """
    Combined structural and task-performance evaluation result.

    This result combines:
    - Structural metrics from ArchitectureEvaluator (Step 9/10)
    - Task-performance metrics from TaskPerformanceEvaluator (Step 20)

    This makes it possible to distinguish:
    - Architecture structurally valid
    - Architecture suitable for the specific task

    Attributes
    ----------
    architecture_id : str
        The architecture identifier.
    architecture_version : Optional[int]
        Architecture version when available.
    validity_score : float
        Structural validity score in [0, 1].
    efficiency_score : float
        Structural efficiency proxy in [0, 1].
    communication_cost : int
        Number of directed communication edges.
    active_agent_count : int
        Number of active agents.
    overall_structural_score : float
        Combined structural score in [0, 1].
    task_id : Optional[str]
        The task ID if task-performance was evaluated.
    task_success_score : Optional[float]
        Task success proxy score in [0, 1], or None if not evaluated.
    capability_coverage_score : Optional[float]
        Capability coverage score in [0, 1], or None if not evaluated.
    task_alignment_score : Optional[float]
        Task alignment score in [0, 1], or None if not evaluated.
    missing_capabilities : List[str]
        Missing capabilities, if task-performance was evaluated.
    covered_capabilities : List[str]
        Covered capabilities, if task-performance was evaluated.
    notes : str
        Human-readable notes.
    """

    architecture_id: str = Field(description="Architecture identifier")
    architecture_version: Optional[int] = Field(
        default=None, description="Architecture version when available"
    )
    validity_score: float = Field(description="Structural validity score in [0, 1]")
    efficiency_score: float = Field(description="Structural efficiency proxy in [0, 1]")
    communication_cost: int = Field(description="Number of directed communication edges")
    active_agent_count: int = Field(description="Number of active agents")
    overall_structural_score: float = Field(
        description="Combined structural score in [0, 1]"
    )
    task_id: Optional[str] = Field(
        default=None, description="Task ID if task-performance was evaluated"
    )
    task_success_score: Optional[float] = Field(
        default=None,
        description="Task success proxy score in [0, 1], or None if not evaluated",
    )
    capability_coverage_score: Optional[float] = Field(
        default=None,
        description="Capability coverage score in [0, 1], or None if not evaluated",
    )
    task_alignment_score: Optional[float] = Field(
        default=None,
        description="Task alignment score in [0, 1], or None if not evaluated",
    )
    missing_capabilities: List[str] = Field(
        default_factory=list,
        description="Missing capabilities, if task-performance was evaluated",
    )
    covered_capabilities: List[str] = Field(
        default_factory=list,
        description="Covered capabilities, if task-performance was evaluated",
    )
    notes: str = Field(
        default="",
        description="Human-readable notes for the evaluation",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        return self.model_dump(mode="json", exclude_none=True)

    def serialize(self) -> Dict[str, Any]:
        """Alias for to_dict for consistency."""
        return self.to_dict()


# ============================================================================
# TaskPerformanceEvaluator
# ============================================================================


class TaskPerformanceEvaluator:
    """
    Deterministic task-performance evaluator for MAS architectures.

    This evaluator computes how well a MAS architecture matches a specific
    task's requirements based on:

    1. Capability coverage: Which required capabilities are provided by
       active agents' roles.
    2. Task alignment: How well the architecture's roles align with the
       task category.
    3. Communication structure: Whether the architecture has reasonable
       connectivity for the task.

    The task_success_score is a deterministic architecture-task compatibility
    proxy. It is NOT actual LLM task execution.

    The evaluator is fully offline and deterministic. Given the same
    MetaTask and MASArchitecture, it always produces the same result.

    Attributes
    ----------
    None (stateless evaluator)
    """

    def __init__(self) -> None:
        """Initialize the evaluator."""
        pass

    def evaluate(
        self,
        task: MetaTask,
        architecture: MASArchitecture,
        *,
        architecture_version: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> TaskPerformanceResult:
        """
        Evaluate how well the architecture matches the task requirements.

        Parameters
        ----------
        task : MetaTask
            The task to evaluate against.
        architecture : MASArchitecture
            The architecture to evaluate.
        architecture_version : Optional[int]
            Optional architecture version.
        notes : Optional[str]
            Optional human-readable notes.

        Returns
        -------
        TaskPerformanceResult
            The task-performance evaluation result.
        """
        # Get required capabilities from the task
        required_capabilities = set(task.required_capabilities)

        # Get active agents and their roles
        active_agent_ids = set(architecture.active_agent_ids)
        role_map = architecture.role_map

        # Get capabilities provided by active agents
        covered_capabilities: Set[str] = set()
        agent_roles: Dict[str, str] = {}

        for agent_id in active_agent_ids:
            role = role_map.get(agent_id, "")
            if role:
                agent_roles[agent_id] = role
                capabilities = get_capabilities_for_role(role)
                covered_capabilities.update(capabilities)

        # Calculate capability coverage
        missing_capabilities = required_capabilities - covered_capabilities
        covered_list = sorted(list(required_capabilities & covered_capabilities))
        missing_list = sorted(list(missing_capabilities))

        if len(required_capabilities) > 0:
            capability_coverage_score = len(required_capabilities & covered_capabilities) / len(
                required_capabilities
            )
        else:
            # No requirements = full coverage
            capability_coverage_score = 1.0

        # Calculate task alignment score
        task_alignment_score = self._calculate_task_alignment(
            task_category=task.task_category,
            required_capabilities=required_capabilities,
            covered_capabilities=covered_capabilities,
            agent_roles=agent_roles,
            communication_edge_count=len(architecture.communication_edges),
            active_agent_count=architecture.active_agent_count,
        )

        # Calculate task success score (combined proxy)
        task_success_score = self._calculate_task_success_score(
            capability_coverage_score=capability_coverage_score,
            task_alignment_score=task_alignment_score,
            missing_capabilities=missing_capabilities,
            active_agent_count=architecture.active_agent_count,
            required_capability_count=len(required_capabilities),
        )

        # Build notes
        note_parts = []
        if missing_capabilities:
            note_parts.append(f"Missing capabilities: {', '.join(sorted(missing_capabilities))}")
        if capability_coverage_score < 1.0:
            note_parts.append(
                f"Capability coverage: {capability_coverage_score:.2%}"
            )

        default_notes = (
            "Deterministic architecture-task compatibility proxy for task performance. "
            "This is NOT actual LLM task execution."
        )
        if note_parts:
            default_notes += " " + " ".join(note_parts)

        return TaskPerformanceResult(
            task_id=task.task_id,
            architecture_id=architecture.architecture_id,
            task_success_score=round(task_success_score, 6),
            capability_coverage_score=round(capability_coverage_score, 6),
            task_alignment_score=round(task_alignment_score, 6),
            missing_capabilities=missing_list,
            covered_capabilities=covered_list,
            active_agent_count=architecture.active_agent_count,
            communication_edge_count=len(architecture.communication_edges),
            notes=notes or default_notes,
        )

    def _calculate_task_alignment(
        self,
        task_category: str,
        required_capabilities: Set[str],
        covered_capabilities: Set[str],
        agent_roles: Dict[str, str],
        communication_edge_count: int,
        active_agent_count: int,
    ) -> float:
        """
        Calculate how well the architecture aligns with the task category.

        Parameters
        ----------
        task_category : str
            The task category.
        required_capabilities : Set[str]
            The required capabilities.
        covered_capabilities : Set[str]
            The capabilities covered by active agents.
        agent_roles : Dict[str, str]
            Mapping of agent ID to role for active agents.
        communication_edge_count : int
            Number of communication edges.
        active_agent_count : int
            Number of active agents.

        Returns
        -------
        float
            Task alignment score in [0.0, 1.0].
        """
        if not required_capabilities:
            return 1.0

        # Base alignment from capability coverage
        capability_ratio = len(covered_capabilities & required_capabilities) / len(
            required_capabilities
        )

        # Role diversity bonus: more diverse roles = better alignment
        unique_roles = len(set(agent_roles.values()))
        role_diversity_factor = min(1.0, unique_roles / 3.0)  # Normalize to 3+ roles = 1.0

        # Communication structure factor
        # More edges can indicate better coordination, but not always necessary
        if active_agent_count > 1:
            # For multi-agent systems, some communication is expected
            expected_min_edges = active_agent_count - 1  # At least a tree structure
            if communication_edge_count >= expected_min_edges:
                comm_factor = 1.0
            else:
                comm_factor = communication_edge_count / max(1, expected_min_edges)
        else:
            # Single agent doesn't need communication
            comm_factor = 1.0

        # Combine factors
        alignment = 0.5 * capability_ratio + 0.3 * role_diversity_factor + 0.2 * comm_factor

        return min(1.0, max(0.0, alignment))

    def _calculate_task_success_score(
        self,
        capability_coverage_score: float,
        task_alignment_score: float,
        missing_capabilities: Set[str],
        active_agent_count: int,
        required_capability_count: int,
    ) -> float:
        """
        Calculate the overall task success proxy score.

        Parameters
        ----------
        capability_coverage_score : float
            Capability coverage score in [0.0, 1.0].
        task_alignment_score : float
            Task alignment score in [0.0, 1.0].
        missing_capabilities : Set[str]
            Set of missing capabilities.
        active_agent_count : int
            Number of active agents.
        required_capability_count : int
            Number of required capabilities.

        Returns
        -------
        float
            Task success proxy score in [0.0, 1.0].
        """
        # If no requirements, full success
        if required_capability_count == 0:
            return 1.0

        # If all capabilities are missing, very low score
        if len(missing_capabilities) == required_capability_count:
            return 0.0

        # Weighted combination of scores
        # Capability coverage is most important
        # Task alignment provides additional signal
        # Penalty for missing critical capabilities

        base_score = 0.6 * capability_coverage_score + 0.4 * task_alignment_score

        # Penalty for missing capabilities (proportional to how many are missing)
        if missing_capabilities:
            missing_ratio = len(missing_capabilities) / required_capability_count
            penalty = 0.1 * missing_ratio  # Up to 10% penalty
            base_score = max(0.0, base_score - penalty)

        # Bonus for having more active agents (more capacity to handle tasks)
        if active_agent_count >= 3:
            base_score = min(1.0, base_score + 0.05)  # Up to 5% bonus

        return min(1.0, max(0.0, base_score))


# ============================================================================
# Combined Evaluation Helpers
# ============================================================================


def create_combined_evaluation(
    structural_result: Any,
    task_performance_result: Optional[TaskPerformanceResult] = None,
) -> CombinedEvaluationResult:
    """
    Create a combined evaluation result from structural and task-performance results.

    Parameters
    ----------
    structural_result : EvaluationResult
        The structural evaluation result from ArchitectureEvaluator.
    task_performance_result : Optional[TaskPerformanceResult]
        Optional task-performance result from TaskPerformanceEvaluator.

    Returns
    -------
    CombinedEvaluationResult
        The combined evaluation result.
    """
    from app.evaluation.evaluator import EvaluationResult

    if not isinstance(structural_result, EvaluationResult):
        raise TypeError("structural_result must be an EvaluationResult")

    result = CombinedEvaluationResult(
        architecture_id=structural_result.architecture_id,
        architecture_version=structural_result.architecture_version,
        validity_score=structural_result.validity_score,
        efficiency_score=structural_result.efficiency_score,
        communication_cost=structural_result.communication_cost,
        active_agent_count=structural_result.active_agent_count,
        overall_structural_score=structural_result.overall_score,
        notes="Combined structural + task-performance evaluation",
    )

    if task_performance_result is not None:
        result.task_id = task_performance_result.task_id
        result.task_success_score = task_performance_result.task_success_score
        result.capability_coverage_score = task_performance_result.capability_coverage_score
        result.task_alignment_score = task_performance_result.task_alignment_score
        result.missing_capabilities = task_performance_result.missing_capabilities
        result.covered_capabilities = task_performance_result.covered_capabilities
        result.notes = (
            f"Combined structural + task-performance evaluation. "
            f"Task: {task_performance_result.task_id}. "
            f"Task success proxy: {task_performance_result.task_success_score:.4f}. "
            f"(This is a compatibility proxy, NOT actual LLM task execution.)"
        )

    return result


# ============================================================================
# Extended Reward Calculator
# ============================================================================


class TaskPerformanceRewardCalculator:
    """
    Extended reward calculator that can incorporate task-performance signals.

    This calculator extends the existing RewardCalculator from Step 9/10
    to optionally include task-performance components in the reward.

    The reward formula (when task performance is provided):

        structural_reward = current.overall_score - previous.overall_score
        task_performance_reward = task_performance_delta (optional)
        combined_reward = structural_reward + task_performance_weight * task_performance_reward

    When task performance is NOT provided, the calculator falls back to the
    original Step 9/10 behavior (structural reward only).

    This maintains backward compatibility: callers that do not provide
    task-performance information get the same reward as before.

    Attributes
    ----------
    task_performance_weight : float
        Weight for task-performance component in combined reward.
        Default 0.0 (no task-performance contribution, backward compatible).
    invalid_penalty : float
        Penalty for invalid transitions.
    """

    def __init__(
        self,
        invalid_penalty: Optional[float] = None,
        task_performance_weight: float = 0.0,
    ) -> None:
        """
        Initialize the reward calculator.

        Parameters
        ----------
        invalid_penalty : Optional[float]
            Penalty for invalid transitions. Default -1.0.
        task_performance_weight : float
            Weight for task-performance component. Default 0.0 (backward compatible).
        """
        from app.evaluation.reward import INVALID_TRANSITION_PENALTY

        self._invalid_penalty = (
            invalid_penalty if invalid_penalty is not None else INVALID_TRANSITION_PENALTY
        )
        self.task_performance_weight = task_performance_weight

    def calculate(
        self,
        *,
        previous_result: Any,
        current_result: Any,
        valid_transition: bool,
        previous_task_performance: Optional[TaskPerformanceResult] = None,
        current_task_performance: Optional[TaskPerformanceResult] = None,
        notes: Optional[str] = None,
    ) -> float:
        """
        Calculate reward from evaluation results.

        Parameters
        ----------
        previous_result : EvaluationResult
            Structural evaluation before the transition.
        current_result : EvaluationResult
            Structural evaluation after the transition.
        valid_transition : bool
            Whether the transition was valid.
        previous_task_performance : Optional[TaskPerformanceResult]
            Task performance before the transition.
        current_task_performance : Optional[TaskPerformanceResult]
            Task performance after the transition.
        notes : Optional[str]
            Optional notes.

        Returns
        -------
        float
            The calculated reward.
        """
        from app.evaluation.evaluator import EvaluationResult

        if not valid_transition:
            return self._invalid_penalty

        # Structural reward (always included)
        if not isinstance(previous_result, EvaluationResult) or not isinstance(
            current_result, EvaluationResult
        ):
            raise TypeError("Results must be EvaluationResult objects")

        structural_delta = (
            current_result.overall_score - previous_result.overall_score
        )

        # Task performance reward (optional)
        task_performance_delta = 0.0
        if (
            previous_task_performance is not None
            and current_task_performance is not None
        ):
            # Calculate delta in task success score
            task_performance_delta = (
                current_task_performance.task_success_score
                - previous_task_performance.task_success_score
            )

        # Combined reward
        combined_reward = structural_delta + self.task_performance_weight * task_performance_delta

        return round(combined_reward, 6)

    def calculate_with_context(
        self,
        *,
        previous_result: Any,
        current_result: Any,
        valid_transition: bool,
        previous_task_performance: Optional[TaskPerformanceResult] = None,
        current_task_performance: Optional[TaskPerformanceResult] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate reward and return a structured context dict.

        Parameters
        ----------
        previous_result : EvaluationResult
            Structural evaluation before the transition.
        current_result : EvaluationResult
            Structural evaluation after the transition.
        valid_transition : bool
            Whether the transition was valid.
        previous_task_performance : Optional[TaskPerformanceResult]
            Task performance before the transition.
        current_task_performance : Optional[TaskPerformanceResult]
            Task performance after the transition.
        notes : Optional[str]
            Optional notes.

        Returns
        -------
        Dict[str, Any]
            Dictionary with reward and context information.
        """
        reward = self.calculate(
            previous_result=previous_result,
            current_result=current_result,
            valid_transition=valid_transition,
            previous_task_performance=previous_task_performance,
            current_task_performance=current_task_performance,
            notes=notes,
        )

        from app.evaluation.evaluator import EvaluationResult

        context: Dict[str, Any] = {
            "reward": reward,
            "valid_transition": valid_transition,
            "previous_score": previous_result.overall_score,
            "current_score": current_result.overall_score,
            "reward_delta": round(
                current_result.overall_score - previous_result.overall_score, 6
            ),
            "notes": notes or "",
        }

        if (
            previous_task_performance is not None
            and current_task_performance is not None
        ):
            context["previous_task_success_score"] = (
                previous_task_performance.task_success_score
            )
            context["current_task_success_score"] = (
                current_task_performance.task_success_score
            )
            context["task_performance_delta"] = round(
                current_task_performance.task_success_score
                - previous_task_performance.task_success_score,
                6,
            )
            context["task_performance_weight"] = self.task_performance_weight

        return context
