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


# ---------------------------------------------------------------------------
# Decoupled Dual-Objective Reward Calculator (Architecture vs. Response)
# ---------------------------------------------------------------------------

class DualObjectiveRewardCalculator:
    """Decoupled reward calculator balancing Architecture Efficiency (R_arch) with Response Quality (R_resp).
    
    Prevents reward hacking where strong LLMs output correct answers despite bloated topologies
    or redundant tools, while ensuring topologies missing necessary tools/nodes are heavily penalized.
    """

    def __init__(
        self,
        *,
        missing_capability_penalty: float = 0.40,
        unnecessary_tool_penalty: float = 0.35,
        surplus_agent_penalty: float = 0.25,
        parsimony_bonus: float = 0.30,
        human_approval_reward: float = 0.50,
        human_rejection_penalty: float = 0.60,
    ) -> None:
        self.missing_capability_penalty = missing_capability_penalty
        self.unnecessary_tool_penalty = unnecessary_tool_penalty
        self.surplus_agent_penalty = surplus_agent_penalty
        self.parsimony_bonus = parsimony_bonus
        self.human_approval_reward = human_approval_reward
        self.human_rejection_penalty = human_rejection_penalty

    def calculate(
        self,
        *,
        required_capabilities: list[str],
        active_agents: list[str],
        invoked_agents: list[str],
        tools_executed: list[str],
        coverage_score: float,
        missing_capabilities: list[str],
        surplus_agents: list[str],
        user_feedback: Optional[str] = None,
        final_response: Any = None,
    ) -> Dict[str, Any]:
        """Compute decoupled architecture and response rewards.
        
        Returns:
            Dict with total_reward, r_arch, r_resp, component breakdowns, and explanations.
        """
        # 1. Architecture Reward (R_arch)
        # Missing capabilities penalty: task needed a capability that graph failed to provide
        missing_pen = -self.missing_capability_penalty * len(missing_capabilities) if missing_capabilities else 0.0

        # Unnecessary tools penalty: tool was run whose capability was not required
        tool_to_cap = {
            "web_search": "web_search",
            "calculator": "math",
            "get_current_date": "web_search",
            "code_interpreter": "coding",
            "retriever": "research",
        }
        unnecessary_tools = []
        for t in tools_executed:
            cap = tool_to_cap.get(t, t)
            if cap not in required_capabilities:
                unnecessary_tools.append(t)
        tool_pen = -self.unnecessary_tool_penalty * len(unnecessary_tools)

        # Surplus agent penalty: nodes active that were not required
        surplus_pen = -self.surplus_agent_penalty * len(surplus_agents) if surplus_agents else 0.0

        # Parsimony bonus: minimal sufficient topology with 0 waste
        parsimony = 0.0
        if len(missing_capabilities) == 0 and len(surplus_agents) == 0 and len(unnecessary_tools) == 0:
            parsimony = self.parsimony_bonus

        # 2. Response Quality Reward (R_resp) and Human-Directed Feedback
        human_choice = str(user_feedback or "").strip().lower()
        if human_choice in {"y", "perfect", "p"}:
            r_resp = self.human_approval_reward
            resp_reason = "Human confirmed PERFECT execution & answer (+0.50)"
            if len(missing_capabilities) == 0 and len(surplus_agents) == 0 and len(unnecessary_tools) == 0:
                parsimony += 0.15  # Extra precision bonus for verified minimal topology
        elif human_choice in {"over", "o"}:
            r_resp = 0.10  # Answer was usable, but topology was bloated
            surplus_pen -= 0.40  # Direct architectural penalty for over-provisioning
            resp_reason = "Human flagged OVER-USED topology: unnecessary tools or redundant nodes (-0.40)"
        elif human_choice in {"under", "u"}:
            r_resp = -0.30  # Answer was deficient due to missing nodes
            missing_pen -= 0.50  # Direct architectural penalty for under-provisioning
            resp_reason = "Human flagged UNDER-USED topology: missing essential nodes or tools (-0.50)"
        elif human_choice in {"n", "bad", "no"}:
            r_resp = -self.human_rejection_penalty
            resp_reason = "Human rejected task answer (-0.60)"
        else:
            # Automated verification without human bias
            if missing_capabilities:
                r_resp = -0.30
                resp_reason = "Automated evaluation: necessary capabilities missing (-0.30)"
            elif final_response is None or (isinstance(final_response, dict) and not final_response.get("final_answer")):
                r_resp = -0.40
                resp_reason = "Automated evaluation: empty or invalid final answer (-0.40)"
            else:
                r_resp = 0.20
                resp_reason = "Automated evaluation: complete execution and capability satisfaction (+0.20)"

        r_arch = round(coverage_score + missing_pen + tool_pen + surplus_pen + parsimony, 4)
        total_reward = round(r_arch + r_resp, 4)

        return {
            "total_reward": total_reward,
            "r_arch": r_arch,
            "r_resp": r_resp,
            "components": {
                "coverage_score": round(coverage_score, 4),
                "missing_capability_penalty": round(missing_pen, 4),
                "unnecessary_tool_penalty": round(tool_pen, 4),
                "surplus_agent_penalty": round(surplus_pen, 4),
                "parsimony_bonus": round(parsimony, 4),
            },
            "unnecessary_tools": unnecessary_tools,
            "missing_capabilities": missing_capabilities,
            "surplus_agents": surplus_agents,
            "response_reason": resp_reason,
        }


