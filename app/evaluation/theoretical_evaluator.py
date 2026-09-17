"""Theoretical Architecture Evaluation Engine.

Provides multi-dimensional architecture evaluation grounded in:
1. Semantic & Capability Alignment (Coverage vs. Surplus Waste)
2. Topological & Graph Theory Properties (Critical Path, Density, Cycles/Feedback Loops)
3. Multi-Objective Pareto Utility & Cost-Performance Modeling
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from app.architecture.models import MASArchitecture
from app.evaluation.task_performance import ROLE_CAPABILITY_MAP, get_capabilities_for_role


# ---------------------------------------------------------------------------
# Metric Models
# ---------------------------------------------------------------------------

class AlignmentMetrics(BaseModel):
    """Semantic alignment between task capability needs and active agents."""

    coverage_score: float = Field(
        description="Ratio of required capabilities covered by active agents in [0, 1]."
    )
    covered_capabilities: List[str] = Field(default_factory=list)
    missing_capabilities: List[str] = Field(default_factory=list)
    surplus_agents: List[str] = Field(
        default_factory=list,
        description="Active agents whose capabilities are completely unused by the task."
    )
    redundancy_ratio: float = Field(
        description="Proportion of active agents that are surplus/redundant in [0, 1]."
    )
    alignment_score: float = Field(
        description="Composite score balancing coverage against redundant bloat in [0, 1]."
    )


class TopologicalMetrics(BaseModel):
    """Graph theoretical metrics for communication topology."""

    critical_path_length: int = Field(
        description="Length (number of nodes) of the longest simple path from entry to exit."
    )
    critical_path_nodes: List[str] = Field(default_factory=list)
    graph_density: float = Field(
        description="Ratio of actual directed communication edges to possible edges."
    )
    has_cycles: bool = Field(
        description="Whether the topology contains directed feedback loops/cycles."
    )
    detected_cycles: List[List[str]] = Field(
        default_factory=list,
        description="Lists of nodes forming feedback loops (e.g., ['coder', 'critic', 'coder'])."
    )
    orphan_nodes: List[str] = Field(
        default_factory=list,
        description="Active nodes with in-degree 0 (excluding entry point)."
    )
    dead_end_nodes: List[str] = Field(
        default_factory=list,
        description="Active nodes with out-degree 0 (excluding terminal exit point)."
    )
    is_valid_dag: bool = Field(
        description="True if topology is acyclic and free of orphans/dead-ends."
    )
    topological_efficiency: float = Field(
        description="Overall structural efficiency score in [0, 1]."
    )


class ParetoUtilityResult(BaseModel):
    """Multi-objective cost-performance trade-off evaluation."""

    quality_score: float = Field(description="Estimated or empirical task outcome quality in [0, 1].")
    cost_penalty: float = Field(description="Penalty proportional to token usage / agent count.")
    latency_penalty: float = Field(description="Penalty proportional to critical path length.")
    bloat_penalty: float = Field(description="Penalty for unnecessary structural complexity.")
    net_utility: float = Field(description="Net architecture utility score.")
    classification: str = Field(
        description="Classification: 'optimal', 'over_engineered', 'under_engineered', or 'malformed'."
    )


class ComprehensiveArchitectureEvaluation(BaseModel):
    """Unified evaluation combining semantic, topological, and Pareto metrics."""

    architecture_id: str
    architecture_version: Optional[int] = None
    validity_score: float = 1.0
    task_success_score: Optional[float] = None
    efficiency_score: float = 0.0
    communication_cost: int = 0
    active_agent_count: int
    edge_count: int
    alignment: AlignmentMetrics
    topology: TopologicalMetrics
    pareto: ParetoUtilityResult
    overall_score: float

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


# ---------------------------------------------------------------------------
# Graph Theory Algorithms
# ---------------------------------------------------------------------------

def detect_cycles(nodes: List[str], edges: List[Tuple[str, str]]) -> List[List[str]]:
    """Detect all elementary directed cycles in the graph using DFS."""
    adj: Dict[str, List[str]] = collections.defaultdict(list)
    for src, dst in edges:
        if src in nodes and dst in nodes:
            adj[src].append(dst)

    cycles: List[List[str]] = []
    visited: Set[str] = set()
    rec_stack: List[str] = []

    def dfs(curr: str) -> None:
        visited.add(curr)
        rec_stack.append(curr)

        for neighbor in adj[curr]:
            if neighbor in rec_stack:
                # Cycle found
                idx = rec_stack.index(neighbor)
                cycle_path = rec_stack[idx:] + [neighbor]
                # Normalize cycle order for deduplication
                if len(cycle_path) > 2:
                    cycles.append(cycle_path)
            elif neighbor not in visited:
                dfs(neighbor)

        rec_stack.pop()

    for node in nodes:
        if node not in visited:
            dfs(node)

    return cycles


def compute_longest_path(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    entry: str = "planner",
    exit_node: str = "finalizer",
) -> Tuple[int, List[str]]:
    """Compute the longest simple path from entry to exit (critical path)."""
    if entry not in nodes or exit_node not in nodes:
        return 0, []

    adj: Dict[str, List[str]] = collections.defaultdict(list)
    for src, dst in edges:
        if src in nodes and dst in nodes and src != dst:
            adj[src].append(dst)

    longest_path: List[str] = []

    def dfs(curr: str, current_path: List[str], visited: Set[str]) -> None:
        nonlocal longest_path
        if curr == exit_node:
            if len(current_path) > len(longest_path):
                longest_path = list(current_path)
            return

        for nxt in adj[curr]:
            if nxt not in visited:
                visited.add(nxt)
                current_path.append(nxt)
                dfs(nxt, current_path, visited)
                current_path.pop()
                visited.remove(nxt)

    dfs(entry, [entry], {entry})
    return len(longest_path), longest_path


# ---------------------------------------------------------------------------
# Evaluator Class
# ---------------------------------------------------------------------------

class TheoreticalArchitectureEvaluator:
    """Theoretical Architecture Evaluator evaluating Alignment, Graph Topology,
    and Multi-Objective Pareto Utility.
    """

    def __init__(
        self,
        *,
        lambda_cost: float = 0.15,
        lambda_latency: float = 0.15,
        lambda_bloat: float = 0.10,
    ) -> None:
        self.lambda_cost = lambda_cost
        self.lambda_latency = lambda_latency
        self.lambda_bloat = lambda_bloat

    def evaluate(
        self,
        architecture: MASArchitecture,
        required_capabilities: Optional[List[str]] = None,
        task_quality: Optional[float] = None,
        tokens_consumed: Optional[int] = None,
        entry_node: str = "planner",
        exit_node: str = "finalizer",
    ) -> ComprehensiveArchitectureEvaluation:
        """Run full multi-dimensional evaluation on a MASArchitecture."""
        req_caps = set(required_capabilities or [])
        active_agents = sorted(list(architecture.active_agent_ids))
        edges = [(e.source, e.target) for e in architecture.communication_edges]
        active_edges = [(s, t) for s, t in edges if s in active_agents and t in active_agents]

        # --------------------------------------------------------------
        # 1. Semantic & Capability Alignment
        # --------------------------------------------------------------
        agent_role_map = architecture.role_map
        provided_caps: Set[str] = set()
        surplus_agents: List[str] = []

        # Role capabilities
        agent_caps_map: Dict[str, Set[str]] = {}
        for agent_id in active_agents:
            role = agent_role_map.get(agent_id, "")
            caps = get_capabilities_for_role(role)
            # Also include any capabilities declared on the agent definition
            for agent_def in architecture.agents:
                if agent_def.agent_id == agent_id:
                    caps = caps.union(set(agent_def.capabilities))
            agent_caps_map[agent_id] = caps
            provided_caps.update(caps)

            # Check if agent is surplus (core entry/exit planner/finalizer not marked as surplus)
            if agent_id not in {entry_node, exit_node}:
                if req_caps and not (caps & req_caps):
                    surplus_agents.append(agent_id)
                elif not req_caps and agent_id in {"coder", "researcher", "critic", "tool_executor"}:
                    surplus_agents.append(agent_id)

        covered = sorted(list(req_caps & provided_caps)) if req_caps else []
        missing = sorted(list(req_caps - provided_caps)) if req_caps else []
        coverage_score = round(len(covered) / len(req_caps), 4) if req_caps else 1.0

        redundancy_ratio = round(len(surplus_agents) / max(1, len(active_agents)), 4)
        alignment_score = round(max(0.0, coverage_score - (0.3 * redundancy_ratio)), 4)

        alignment = AlignmentMetrics(
            coverage_score=coverage_score,
            covered_capabilities=covered,
            missing_capabilities=missing,
            surplus_agents=surplus_agents,
            redundancy_ratio=redundancy_ratio,
            alignment_score=alignment_score,
        )

        # --------------------------------------------------------------
        # 2. Graph & Topological Metrics
        # --------------------------------------------------------------
        cycles = detect_cycles(active_agents, active_edges)
        has_cycles = len(cycles) > 0

        # Critical path length
        crit_len, crit_nodes = compute_longest_path(
            active_agents, active_edges, entry=entry_node, exit_node=exit_node
        )

        # Graph density
        n = len(active_agents)
        possible_edges = n * (n - 1) if n > 1 else 1
        graph_density = round(len(active_edges) / possible_edges, 4)

        # In-degree and out-degree
        in_deg: Dict[str, int] = collections.defaultdict(int)
        out_deg: Dict[str, int] = collections.defaultdict(int)
        for s, t in active_edges:
            out_deg[s] += 1
            in_deg[t] += 1

        orphans = [node for node in active_agents if in_deg[node] == 0 and node != entry_node]
        dead_ends = [node for node in active_agents if out_deg[node] == 0 and node != exit_node]
        is_valid_dag = (not has_cycles) and (len(orphans) == 0) and (len(dead_ends) == 0) and (crit_len > 0)

        # Efficiency calculation
        compactness = 1.0 - min(1.0, max(0, crit_len - 2) / 6.0)
        density_penalty = min(1.0, graph_density * 1.5)
        topological_efficiency = round(
            max(0.0, 0.5 * compactness + 0.3 * (1.0 - density_penalty) + (0.2 if is_valid_dag else 0.0)),
            4
        )

        topology = TopologicalMetrics(
            critical_path_length=crit_len,
            critical_path_nodes=crit_nodes,
            graph_density=graph_density,
            has_cycles=has_cycles,
            detected_cycles=cycles,
            orphan_nodes=orphans,
            dead_end_nodes=dead_ends,
            is_valid_dag=is_valid_dag,
            topological_efficiency=topological_efficiency,
        )

        # --------------------------------------------------------------
        # 3. Multi-Objective Pareto Utility & Classification
        # --------------------------------------------------------------
        q_score = task_quality if task_quality is not None else coverage_score

        # Cost penalty (tokens if given, otherwise active agent count proxy)
        if tokens_consumed is not None:
            norm_cost = min(1.0, tokens_consumed / 4000.0)
        else:
            norm_cost = min(1.0, (len(active_agents) - 1) / 5.0)
        cost_penalty = round(self.lambda_cost * norm_cost, 4)

        # Latency penalty (based on critical path)
        norm_latency = min(1.0, max(0, crit_len - 1) / 5.0)
        latency_penalty = round(self.lambda_latency * norm_latency, 4)

        # Structural bloat penalty (edges + surplus agents)
        norm_bloat = min(1.0, (len(active_edges) + len(surplus_agents) * 2) / 10.0)
        bloat_penalty = round(self.lambda_bloat * norm_bloat, 4)

        net_utility = round(max(0.0, q_score - cost_penalty - latency_penalty - bloat_penalty), 4)

        # Classification
        if len(orphans) > 0 or len(dead_ends) > 0 or crit_len == 0:
            classification = "malformed"
        elif coverage_score < 1.0:
            classification = "under_engineered"
        elif redundancy_ratio > 0.35 or len(surplus_agents) >= 2:
            classification = "over_engineered"
        else:
            classification = "optimal"

        pareto = ParetoUtilityResult(
            quality_score=q_score,
            cost_penalty=cost_penalty,
            latency_penalty=latency_penalty,
            bloat_penalty=bloat_penalty,
            net_utility=net_utility,
            classification=classification,
        )

        # Combined overall score
        overall = round((0.4 * alignment_score + 0.3 * topological_efficiency + 0.3 * net_utility), 4)

        return ComprehensiveArchitectureEvaluation(
            architecture_id=architecture.architecture_id,
            architecture_version=getattr(architecture, "version", None),
            validity_score=1.0 if (len(orphans) == 0 and len(dead_ends) == 0 and crit_len > 0) else 0.0,
            task_success_score=task_quality,
            efficiency_score=topological_efficiency,
            communication_cost=len(active_edges),
            active_agent_count=len(active_agents),
            edge_count=len(active_edges),
            alignment=alignment,
            topology=topology,
            pareto=pareto,
            overall_score=overall,
        )
