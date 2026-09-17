"""Token, Cost, and Architecture Adaptation Profiler.

Tracks token consumption, latency, and cost reduction across adaptation iterations
as an architecture evolves toward its minimal sufficient topology.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.architecture.models import MASArchitecture
from app.evaluation.theoretical_evaluator import (
    TheoreticalArchitectureEvaluator,
)


class CostTracker:
    """Profiles token usage and dollar costs for MAS architectures."""

    # Default pricing: $0.0015 per 1,000 tokens (approx. typical open-weights / NIM pricing)
    COST_PER_1K_TOKENS: float = 0.0015
    AVG_TOKENS_PER_AGENT_INVOCATION: int = 650

    @classmethod
    def estimate_tokens(cls, active_agent_count: int, critical_path_length: int) -> int:
        """Estimate token expenditure based on active agents and sequential depth."""
        base_tokens = active_agent_count * cls.AVG_TOKENS_PER_AGENT_INVOCATION
        # Sequential context accumulation adds overhead along the critical path
        context_overhead = max(0, critical_path_length - 1) * 150
        return base_tokens + context_overhead

    @classmethod
    def estimate_cost(cls, tokens: int) -> float:
        """Calculate estimated USD cost."""
        return round((tokens / 1000.0) * cls.COST_PER_1K_TOKENS, 5)

    @classmethod
    def profile_architecture(
        cls,
        architecture: MASArchitecture,
        required_capabilities: Optional[List[str]] = None,
        evaluator: Optional[TheoreticalArchitectureEvaluator] = None,
    ) -> Dict[str, Any]:
        """Profile token usage, cost, latency, and quality for a single architecture run."""
        ev = evaluator or TheoreticalArchitectureEvaluator()
        evaluation = ev.evaluate(architecture, required_capabilities=required_capabilities)
        tokens = cls.estimate_tokens(
            active_agent_count=evaluation.active_agent_count,
            critical_path_length=evaluation.topology.critical_path_length,
        )
        cost = cls.estimate_cost(tokens)
        return {
            "architecture_id": architecture.architecture_id,
            "active_agents": sorted(list(architecture.active_agent_ids)),
            "active_agent_count": evaluation.active_agent_count,
            "critical_path_length": evaluation.topology.critical_path_length,
            "estimated_tokens": tokens,
            "estimated_cost_usd": cost,
            "classification": evaluation.pareto.classification,
            "coverage_score": evaluation.alignment.coverage_score,
            "surplus_agents": evaluation.alignment.surplus_agents,
            "has_cycles": evaluation.topology.has_cycles,
            "detected_cycles": evaluation.topology.detected_cycles,
            "net_utility": evaluation.pareto.net_utility,
            "overall_score": evaluation.overall_score,
        }

