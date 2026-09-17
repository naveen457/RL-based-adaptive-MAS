"""
Baseline reward calculator built on EvaluationResult.

This module turns changes in architecture evaluation into a reward signal.

Design notes
------------
* Reward is based on evaluation delta, not on whether the action was accepted.
* Valid but useless changes can get small/zero reward.
* Valid beneficial structural changes can get positive reward.
* Valid but costly structural changes can get lower reward.
* Invalid transitions receive a negative penalty.

Baseline formula
----------------
For a successful transition:
    reward = current.overall_score - previous.overall_score

For an invalid transition:
    reward = INVALID_TRANSITION_PENALTY

Normalization
-------------
overall_score is already bounded in [0, 1], so the resulting reward is
bounded in [-1, 1] for valid transitions.

This is explicitly a BASELINE reward and will later be replaced or evaluated
as part of the research.
"""

from __future__ import annotations

from typing import Optional

from app.evaluation.evaluator import EvaluationResult

INVALID_TRANSITION_PENALTY = -1.0


class RewardCalculator:
    """Calculate baseline architecture rewards from evaluation results."""

    def __init__(
        self,
        invalid_penalty: Optional[float] = None,
    ) -> None:
        self._invalid_penalty = (
            invalid_penalty if invalid_penalty is not None else INVALID_TRANSITION_PENALTY
        )

    def calculate(
        self,
        *,
        previous_result: EvaluationResult,
        current_result: EvaluationResult,
        valid_transition: bool,
        notes: Optional[str] = None,
    ) -> float:
        """Calculate reward from previous and current evaluation results.

        Parameters
        ----------
        previous_result:
            Evaluation of the architecture before the transition.
        current_result:
            Evaluation of the architecture after the transition.
        valid_transition:
            True when the transition was accepted and produced a valid
            architecture, False for invalid/rejected transitions.
        notes:
            Optional human-readable note returned alongside the reward via the
            result dict.
        """
        if not valid_transition:
            return self._invalid_penalty

        delta = current_result.overall_score - previous_result.overall_score
        return round(delta, 6)

    def calculate_with_context(
        self,
        *,
        previous_result: EvaluationResult,
        current_result: EvaluationResult,
        valid_transition: bool,
        notes: Optional[str] = None,
    ) -> dict:
        """Calculate reward and return a small structured context dict."""
        reward = self.calculate(
            previous_result=previous_result,
            current_result=current_result,
            valid_transition=valid_transition,
            notes=notes,
        )
        return {
            "reward": reward,
            "valid_transition": valid_transition,
            "previous_score": previous_result.overall_score,
            "current_score": current_result.overall_score,
            "reward_delta": round(current_result.overall_score - previous_result.overall_score, 6),
            "notes": notes or "",
        }


# ---------------------------------------------------------------------------
# Multi-Objective Theoretical Reward Calculator
# ---------------------------------------------------------------------------

class MultiObjectiveRewardCalculator:
    """Calculates closed-loop RL rewards using multi-objective theoretical evaluation.
    
    Rewards:
      - Alignment & Quality improvement (positive delta)
      - Simplification bonus: pruning redundant nodes while preserving full coverage
    Penalties:
      - Bloat penalty: adding unnecessary nodes or redundant edges
      - Capability drop penalty: deactivating agents needed by the task
      - Invalid transition penalty
    """

    def __init__(
        self,
        *,
        simplification_bonus: float = 0.20,
        bloat_penalty: float = 0.15,
        coverage_drop_penalty: float = 0.35,
        invalid_penalty: float = INVALID_TRANSITION_PENALTY,
    ) -> None:
        self.simplification_bonus = simplification_bonus
        self.bloat_penalty = bloat_penalty
        self.coverage_drop_penalty = coverage_drop_penalty
        self.invalid_penalty = invalid_penalty

    def calculate(
        self,
        *,
        previous_result: Any,
        current_result: Any,
        valid_transition: bool,
        notes: Optional[str] = None,
    ) -> float:
        """Calculate scalar reward, providing polymorphic compatibility with RewardCalculator."""
        res = self.calculate_from_evaluations(
            previous_eval=previous_result,
            current_eval=current_result,
            valid_transition=valid_transition,
            notes=notes,
        )
        return float(res["reward"])

    def calculate_from_evaluations(
        self,
        *,
        previous_eval: Any,  # ComprehensiveArchitectureEvaluation or EvaluationResult
        current_eval: Any,   # ComprehensiveArchitectureEvaluation or EvaluationResult
        valid_transition: bool,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute multi-objective reward from evaluation instances."""
        if not valid_transition:
            return {
                "reward": self.invalid_penalty,
                "valid_transition": False,
                "components": {"invalid_penalty": self.invalid_penalty},
                "notes": notes or "Invalid transition",
            }

        prev_net = (
            previous_eval.pareto.net_utility
            if hasattr(previous_eval, "pareto")
            else getattr(previous_eval, "overall_score", 0.0)
        )
        curr_net = (
            current_eval.pareto.net_utility
            if hasattr(current_eval, "pareto")
            else getattr(current_eval, "overall_score", 0.0)
        )
        utility_delta = curr_net - prev_net

        prev_coverage = (
            previous_eval.alignment.coverage_score
            if hasattr(previous_eval, "alignment")
            else 1.0
        )
        curr_coverage = (
            current_eval.alignment.coverage_score
            if hasattr(current_eval, "alignment")
            else 1.0
        )

        prev_surplus = (
            previous_eval.alignment.surplus_agents
            if hasattr(previous_eval, "alignment")
            else []
        )
        curr_surplus = (
            current_eval.alignment.surplus_agents
            if hasattr(current_eval, "alignment")
            else []
        )

        # Check simplification bonus: fewer agents, but coverage remains complete (>= 1.0)
        simplification = 0.0
        if (
            current_eval.active_agent_count < previous_eval.active_agent_count
            and curr_coverage >= 1.0
        ):
            simplification = self.simplification_bonus

        # Check bloat penalty: more surplus agents or more agents without utility gain
        bloat = 0.0
        if (
            len(curr_surplus) > len(prev_surplus)
            or (current_eval.active_agent_count > previous_eval.active_agent_count and curr_net <= prev_net)
        ):
            bloat = -self.bloat_penalty

        # Check capability drop penalty
        coverage_drop = 0.0
        if curr_coverage < prev_coverage:
            coverage_drop = -self.coverage_drop_penalty

        total_reward = round(utility_delta + simplification + bloat + coverage_drop, 6)

        prev_class = getattr(getattr(previous_eval, "pareto", None), "classification", "unknown")
        curr_class = getattr(getattr(current_eval, "pareto", None), "classification", "unknown")

        return {
            "reward": total_reward,
            "valid_transition": True,
            "components": {
                "utility_delta": round(utility_delta, 6),
                "simplification_bonus": simplification,
                "bloat_penalty": bloat,
                "coverage_drop_penalty": coverage_drop,
            },
            "previous_utility": prev_net,
            "current_utility": curr_net,
            "previous_classification": prev_class,
            "current_classification": curr_class,
            "notes": notes or "",
        }

