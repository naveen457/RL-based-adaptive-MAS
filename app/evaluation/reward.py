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
