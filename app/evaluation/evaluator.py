"""
Deterministic architecture evaluation utilities.

This module provides structural metrics for a MASArchitecture and an
ArchitectureEvaluator that turns an architecture into an EvaluationResult.

Design notes
------------
* All metrics are computed only from architecture data.
* No LLM calls are made.
* No task execution is performed. Task performance is explicitly marked
  as unavailable in the baseline.
* Metrics are deterministic and reproducible.
* Structural metrics are proxies only. They do NOT claim to measure real
  end-to-end task quality.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, computed_field

from app.architecture.models import MASArchitecture


class EvaluationResult(BaseModel):
    """Deterministic evaluation result for one architecture.

    This result intentionally separates structural metrics from actual
    task-performance measurement. In the baseline, task performance is not
    yet measurable, so it is represented as None.
    """

    architecture_id: str = Field(description="Architecture identifier.")
    architecture_version: Optional[int] = Field(
        default=None,
        description="Architecture version when available, otherwise None.",
    )
    validity_score: float = Field(
        description="Structural validity score in [0, 1].",
    )
    task_success_score: Optional[float] = Field(
        default=None,
        description="Actual task-success measurement. Unavailable in the baseline.",
    )
    efficiency_score: float = Field(
        description="Structural efficiency proxy in [0, 1].",
    )
    communication_cost: int = Field(
        description="Number of directed communication edges.",
    )
    active_agent_count: int = Field(
        description="Number of currently active agents.",
    )
    edge_count: int = Field(
        description="Alias for communication_cost.",
    )
    overall_score: float = Field(
        description="Combined structural score in [0, 1].",
    )
    notes: str = Field(
        default="",
        description="Human-readable notes for the evaluation.",
    )

    @computed_field
    def edge_count_computed(self) -> int:
        """Explicit edge count alias for serialization clarity."""
        return self.edge_count


# ---------------------------------------------------------------------------
# Baseline structural metric functions
# ---------------------------------------------------------------------------


def calculate_validity_score(architecture: MASArchitecture) -> float:
    """Return 1.0 if the architecture is valid, else 0.0.

    This is a structural validity proxy, not a task-quality measure.
    """
    return 1.0 if architecture.is_valid else 0.0


def calculate_communication_cost(architecture: MASArchitecture) -> int:
    """Return the number of directed communication edges.

    This is a crude structural cost proxy. More edges does not mean a better
    architecture.
    """
    return len(architecture.communication_edges)


def calculate_efficiency_score(architecture: MASArchitecture) -> float:
    """Return a deterministic structural efficiency proxy in [0, 1].

    Baseline interpretation:
      * Fewer active agents is structurally cheaper.
      * Lower communication cost is structurally cheaper.
      * Both are normalized and combined with a simple weighted average.

    This is NOT a validated measure of actual task-solving efficiency.
    It exists only so the RL loop has a deterministic scalar to work with.
    """
    active = max(1, architecture.active_agent_count)
    edges = max(1, architecture.communication_edges.__len__() if hasattr(architecture.communication_edges, "__len__") else 0)

    # Normalize each term into [0, 1] where 1.0 is structurally cheapest.
    # These divisor caps are baselines, not research claims.
    active_norm = 1.0 - min(1.0, (active - 1.0) / 4.0)
    edge_norm = 1.0 - min(1.0, (edges - 1.0) / 6.0)

    # Simple weighted combination.
    return round(0.5 * active_norm + 0.5 * edge_norm, 6)


def calculate_overall_score(result: EvaluationResult) -> float:
    """Combine structural metrics into a single [0, 1] score.

    Baseline weighting:
      * validity      0.4
      * efficiency    0.3
      * (negative) communication cost 0.3

    Task success is intentionally excluded because it is unavailable.
    """
    comm_norm = 1.0 - min(1.0, (result.communication_cost - 1.0) / 6.0)
    return round(
        0.4 * result.validity_score
        + 0.3 * result.efficiency_score
        + 0.3 * comm_norm,
        6,
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ArchitectureEvaluator:
    """Deterministic evaluator for MAS architectures.

    Responsibilities:
      - Evaluate an architecture into an EvaluationResult.
      - Compute structural metrics only.
      - Never call an LLM.
      - Make task-performance unavailability explicit.
    """

    def __init__(
        self,
        *,
        default_version: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> None:
        self._default_version = default_version
        self._notes = notes or "Baseline structural evaluation; task performance unavailable."

    def evaluate(
        self,
        architecture: MASArchitecture,
        *,
        architecture_version: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate *architecture* and return an EvaluationResult."""
        version = (
            architecture_version
            if architecture_version is not None
            else self._default_version
        )
        validity = calculate_validity_score(architecture)
        comm_cost = calculate_communication_cost(architecture)
        efficiency = calculate_efficiency_score(architecture)
        active_count = architecture.active_agent_count

        result = EvaluationResult(
            architecture_id=architecture.architecture_id,
            architecture_version=version,
            validity_score=validity,
            task_success_score=None,
            efficiency_score=efficiency,
            communication_cost=comm_cost,
            edge_count=comm_cost,
            active_agent_count=active_count,
            overall_score=0.0,
            notes=notes or self._notes,
        )
        # Compute overall after fields are set.
        object.__setattr__(result, "overall_score", calculate_overall_score(result))
        return result
