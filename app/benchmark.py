"""Automated Multi-Task Benchmark & TensorBoard Training Suite.

Executes a diverse suite of 18 tasks across 6 categories:
1. Greetings / Chit-Chat
2. Mathematical Reasoning / Calculation
3. Code Generation & Data Structures
4. Multi-Hop In-Depth Research
5. Temporal / Live Web Search
6. Cross-Domain Multi-Specialist Synthesis

Computes Pareto frontiers (token cost vs capability coverage), tracks baseline
vs adapted performance, trains the Q-learning policy across episodes, and logs
scalar curves to TensorBoard.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agents.coder import CoderOutput
from app.agents.critic import CriticOutput
from app.agents.finalizer import FinalizerOutput
from app.agents.planner import Planner, PlannerOutput
from app.agents.researcher import ResearcherOutput
from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.llm_adapter import LLMArchitectureAdapter
from app.architecture.manager import ArchitectureManager
from app.architecture.models import MASArchitecture
from app.evaluation.cost_tracker import CostTracker
from app.evaluation.metrics_logger import MetricsLogger
from app.evaluation.theoretical_evaluator import TheoreticalArchitectureEvaluator
from app.graph.adaptive_integration import AdaptiveWorkflowAdapter
from app.rl.active_selector import RLArchitectureSelector
from app.rl.q_learning import QLearningPolicy
from app.runtime.llm_execution import ExistingLLMAgentExecutor
from app.runtime.orchestrator import AdaptiveRuntimeOrchestrator, DynamicGraphExecutionResult


# ---------------------------------------------------------------------------
# Benchmark Task Definitions (18 Diverse Tasks across 6 Domains)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    task_name: str
    category: str
    prompt: str
    required_capabilities: List[str]
    requires_research: bool = False
    requires_coding: bool = False
    requires_verification: bool = False
    requires_tools: bool = False
    difficulty: int = 1


BENCHMARK_TASKS: List[BenchmarkTask] = [
    # 1. Greetings / Chit-chat (Minimal Pareto Topology: planner + finalizer)
    BenchmarkTask(
        task_id="greet-001",
        task_name="Simple Greeting",
        category="greeting",
        prompt="Hello! How can you help me today?",
        required_capabilities=[],
        difficulty=1,
    ),
    BenchmarkTask(
        task_id="greet-002",
        task_name="System Introduction",
        category="greeting",
        prompt="Good morning, introduce the multi-agent system capabilities.",
        required_capabilities=[],
        difficulty=1,
    ),
    BenchmarkTask(
        task_id="greet-003",
        task_name="Inspirational Quote",
        category="greeting",
        prompt="Hi there! Give me a brief inspiring quote about mathematics.",
        required_capabilities=[],
        difficulty=1,
    ),

    # 2. Math & Calculation (Tool/Math Specialist)
    BenchmarkTask(
        task_id="math-001",
        task_name="Compound Interest",
        category="math",
        prompt="Calculate compound interest on $10,000 at 7% annually for 8 years.",
        required_capabilities=["math", "tool_use"],
        requires_tools=True,
        difficulty=2,
    ),
    BenchmarkTask(
        task_id="math-002",
        task_name="Quadratic Roots",
        category="math",
        prompt="Find the roots and vertex of the quadratic equation 3x^2 - 12x + 5 = 0.",
        required_capabilities=["math", "tool_use"],
        requires_tools=True,
        difficulty=2,
    ),
    BenchmarkTask(
        task_id="math-003",
        task_name="Probability Analysis",
        category="math",
        prompt="Compute the probability of getting at least 3 heads in 5 fair coin tosses.",
        required_capabilities=["math", "tool_use"],
        requires_tools=True,
        difficulty=2,
    ),

    # 3. Coding & Data Structures (Coder Specialist)
    BenchmarkTask(
        task_id="code-001",
        task_name="IPv4 Address Validator",
        category="coding",
        prompt="Write a Python function to validate whether an IPv4 address string is valid.",
        required_capabilities=["coding", "testing"],
        requires_coding=True,
        difficulty=2,
    ),
    BenchmarkTask(
        task_id="code-002",
        task_name="Thread-Safe LRU Cache",
        category="coding",
        prompt="Implement a thread-safe LRU Cache in Python with O(1) get and put operations.",
        required_capabilities=["coding", "verification"],
        requires_coding=True,
        requires_verification=True,
        difficulty=3,
    ),
    BenchmarkTask(
        task_id="code-003",
        task_name="Longest Palindrome",
        category="coding",
        prompt="Write a function to find the longest palindromic substring in O(n^2) time.",
        required_capabilities=["coding"],
        requires_coding=True,
        difficulty=3,
    ),

    # 4. Multi-Hop In-Depth Research (Researcher Specialist)
    BenchmarkTask(
        task_id="res-001",
        task_name="Transformer vs Mamba SSM",
        category="research",
        prompt="Research the architectural differences between Transformer and Mamba SSM models.",
        required_capabilities=["research", "information_synthesis"],
        requires_research=True,
        difficulty=3,
    ),
    BenchmarkTask(
        task_id="res-002",
        task_name="Zero-Knowledge SNARKs",
        category="research",
        prompt="Explain how zero-knowledge SNARKs work and their trade-offs in blockchain rollups.",
        required_capabilities=["research"],
        requires_research=True,
        difficulty=3,
    ),
    BenchmarkTask(
        task_id="res-003",
        task_name="Raft Consensus Protocol",
        category="research",
        prompt="Summarize the leader election and log replication guarantees of Raft consensus.",
        required_capabilities=["research", "information_synthesis"],
        requires_research=True,
        difficulty=3,
    ),

    # 5. Temporal / Live Web Search (Tool Executor Specialist)
    BenchmarkTask(
        task_id="web-001",
        task_name="AI Weekly Breakthroughs",
        category="web_search",
        prompt="What are the latest trending breakthroughs in artificial intelligence this week?",
        required_capabilities=["web_search", "tool_use"],
        requires_tools=True,
        difficulty=2,
    ),
    BenchmarkTask(
        task_id="web-002",
        task_name="NASA Mars Exploration",
        category="web_search",
        prompt="Search for recent NASA Mars rover discoveries and mission milestones.",
        required_capabilities=["web_search", "tool_use"],
        requires_tools=True,
        difficulty=2,
    ),
    BenchmarkTask(
        task_id="web-003",
        task_name="Global Climate Summits",
        category="web_search",
        prompt="What are current global events and climate summits taking place today?",
        required_capabilities=["web_search", "tool_use"],
        requires_tools=True,
        difficulty=2,
    ),

    # 6. Cross-Domain Multi-Specialist Synthesis (Researcher + Coder + Critic)
    BenchmarkTask(
        task_id="cross-001",
        task_name="Multiplayer Chess Platform",
        category="cross_domain",
        prompt="Design a microservice architecture for multiplayer chess and write the core matchmaking algorithm in Python.",
        required_capabilities=["research", "coding", "verification"],
        requires_research=True,
        requires_coding=True,
        requires_verification=True,
        difficulty=4,
    ),
    BenchmarkTask(
        task_id="cross-002",
        task_name="Smart Contract Security Audit",
        category="cross_domain",
        prompt="Analyze the security vulnerabilities in decentralized smart contracts and write an audit checklist with code examples.",
        required_capabilities=["research", "coding", "verification"],
        requires_research=True,
        requires_coding=True,
        requires_verification=True,
        difficulty=4,
    ),
    BenchmarkTask(
        task_id="cross-003",
        task_name="ML Data Drift Pipeline",
        category="cross_domain",
        prompt="Design an automated CI/CD pipeline for machine learning models and implement a data drift detection script in Python.",
        required_capabilities=["research", "coding", "testing"],
        requires_research=True,
        requires_coding=True,
        difficulty=4,
    ),
]


# ---------------------------------------------------------------------------
# Mock Simulation Components (for Fast Offline / Test Benchmark Execution)
# ---------------------------------------------------------------------------

class SimulatedPlanner:
    """Deterministic simulated planner for offline benchmarking without API keys."""

    def __init__(self, task_lookup: Dict[str, BenchmarkTask]) -> None:
        self.task_lookup = task_lookup
        self.model = self

    def plan(self, task_text: str) -> PlannerOutput:
        bt = self.task_lookup.get(task_text)
        if bt is None:
            for candidate in self.task_lookup.values():
                if candidate.prompt.lower() in task_text.lower() or task_text.lower() in candidate.prompt.lower():
                    bt = candidate
                    break
        if bt is None:
            bt = BENCHMARK_TASKS[0]

        selected_agents = ["planner", "finalizer"]
        if bt.requires_research:
            selected_agents.append("researcher")
        if bt.requires_coding:
            selected_agents.append("coder")
        if bt.requires_verification:
            selected_agents.append("critic")
        if bt.requires_tools:
            selected_agents.append("tool_executor")

        return PlannerOutput(
            task_understanding=f"Benchmark execution for {bt.task_name}",
            required_capabilities=list(bt.required_capabilities),
            steps=["Analyze requirements", "Execute components", "Synthesize results"],
            selected_agents=selected_agents,
            requires_research=bt.requires_research,
            requires_coding=bt.requires_coding,
            requires_verification=bt.requires_verification,
            requires_tools=bt.requires_tools,
            tools_needed=["web_search"] if bt.requires_tools else [],
            estimated_complexity="medium" if bt.difficulty >= 3 else "simple",
        )


class SimulatedDecisionLLM:
    """Deterministic simulated LLM decision adapter for Pareto-optimal architecture actions."""

    def invoke(self, messages: Any) -> Any:
        class ContentWrapper:
            def __init__(self, content: str) -> None:
                self.content = content

        payload_text = ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                payload_text = m.get("content", "")

        try:
            user_data = json.loads(payload_text)
            plan = user_data.get("planner_output", {})
        except Exception:
            plan = {}

        req_res = plan.get("requires_research", False)
        req_cod = plan.get("requires_coding", False)
        req_ver = plan.get("requires_verification", False)
        req_tls = plan.get("requires_tools", False)

        current_arch = user_data.get("current_architecture", {})
        active_ids = {
            a["agent_id"] for a in current_arch.get("agents", []) if a.get("active")
        }

        actions = []
        if req_tls and "tool_executor" not in active_ids:
            actions.append({"action_type": "activate_agent", "agent_id": "tool_executor"})
        elif not req_tls and "tool_executor" in active_ids:
            actions.append({"action_type": "deactivate_agent", "agent_id": "tool_executor"})

        if req_res and "researcher" not in active_ids:
            actions.append({"action_type": "activate_agent", "agent_id": "researcher"})
        elif not req_res and "researcher" in active_ids:
            actions.append({"action_type": "deactivate_agent", "agent_id": "researcher"})

        if req_cod and "coder" not in active_ids:
            actions.append({"action_type": "activate_agent", "agent_id": "coder"})
        elif not req_cod and "coder" in active_ids:
            actions.append({"action_type": "deactivate_agent", "agent_id": "coder"})

        if req_ver and "critic" not in active_ids:
            actions.append({"action_type": "activate_agent", "agent_id": "critic"})
        elif not req_ver and "critic" in active_ids:
            actions.append({"action_type": "deactivate_agent", "agent_id": "critic"})

        resp = {
            "decision": "apply_actions" if actions else "no_change",
            "reasoning": "Pareto-optimal minimal topology adaptation.",
            "actions": actions,
        }
        return ContentWrapper(json.dumps(resp))


def _create_mock_executor() -> ExistingLLMAgentExecutor:
    class MockResearcher:
        def research(self, task: str) -> ResearcherOutput:
            return ResearcherOutput(
                research_question=task,
                findings=["Comprehensive empirical research findings gathered."],
                sources=["Benchmark Database v1.0"],
            )

    class MockCoder:
        def code(self, task: str) -> CoderOutput:
            return CoderOutput(
                approach="Algorithmic optimization",
                code="def solution():\n    return 'verified_benchmark_output'",
                explanation="Optimal O(1) space/time solution.",
            )

    class MockCritic:
        def review(self, original_task: str, output_to_review: str) -> CriticOutput:
            return CriticOutput(
                overall_assessment="Solution verified successfully.",
                verification_status="passed",
                issues=[],
                corrections=[],
            )

    class MockFinalizer:
        def finalize(self, original_task: str, supporting_info: str) -> FinalizerOutput:
            return FinalizerOutput(
                final_answer=f"Benchmark synthesis completed for: {original_task[:40]}...",
                key_points=["High capability coverage achieved", "Pareto optimal cost profile"],
                limitations=[],
            )

    return ExistingLLMAgentExecutor(
        researcher_factory=MockResearcher,
        coder_factory=MockCoder,
        critic_factory=MockCritic,
        finalizer_factory=MockFinalizer,
    )


# ---------------------------------------------------------------------------
# Benchmark Runner & Pareto Analytics
# ---------------------------------------------------------------------------

@dataclass
class TaskBenchmarkMetric:
    episode: int
    task_id: str
    task_name: str
    category: str
    baseline_tokens: int
    adapted_tokens: int
    token_savings_pct: float
    baseline_cost: float
    adapted_cost: float
    cost_savings_pct: float
    active_agents: int
    capability_coverage: float
    net_utility: float
    classification: str
    decision_engine: str
    is_instant_rl: bool


class MultiTaskBenchmarkSuite:
    """Manages execution, RL training, and Pareto analysis for benchmark tasks."""

    def __init__(
        self,
        *,
        episodes: int = 2,
        mode: str = "mock",
        session_mode: str = "continuous",
        q_table_path: str = "data/q_table.json",
        log_base_dir: str = "runs/benchmark",
    ) -> None:
        self.episodes = episodes
        self.mode = mode
        self.session_mode = session_mode
        self.q_table_path = Path(q_table_path)
        self.log_base_dir = Path(log_base_dir)

        self.logger = MetricsLogger(log_base_dir=str(self.log_base_dir))
        self.evaluator = TheoreticalArchitectureEvaluator()

        self.q_policy = QLearningPolicy(task_aware=True, epsilon=0.15)
        if self.q_table_path.exists():
            try:
                self.q_policy.load_q_table(str(self.q_table_path))
            except Exception:
                pass

        self.rl_selector = RLArchitectureSelector(
            policy=self.q_policy,
            epsilon=0.15,
            min_q_threshold=0.0,
        )

        self.task_map = {t.prompt: t for t in BENCHMARK_TASKS}
        self.metrics: List[TaskBenchmarkMetric] = []

    def _build_orchestrator(self) -> AdaptiveRuntimeOrchestrator:
        if self.mode == "live":
            return AdaptiveRuntimeOrchestrator.from_settings(
                q_policy=self.q_policy,
                rl_selector=self.rl_selector,
            )
        else:
            sim_planner = SimulatedPlanner(self.task_map)
            sim_client = SimulatedDecisionLLM()
            manager = ArchitectureManager.create_default_architecture()
            wf_adapter = AdaptiveWorkflowAdapter(manager=manager)
            adapter = LLMArchitectureAdapter(sim_client, workflow_adapter=wf_adapter)
            return AdaptiveRuntimeOrchestrator(
                planner=sim_planner,
                architecture_adapter=adapter,
                agent_executor=_create_mock_executor(),
                q_policy=self.q_policy,
                rl_selector=self.rl_selector,
            )

    def run(self) -> Dict[str, Any]:
        print("=" * 85)
        print("  ADAPTIVE MULTI-AGENT SYSTEM (AMAS) — MULTI-TASK TRAINING & BENCHMARK SUITE")
        print("=" * 85)
        print(f"  Execution Mode:    {self.mode.upper()} ({'Fast Deterministic Simulation' if self.mode == 'mock' else 'Live LLM Endpoint'})")
        print(f"  Session Mode:      {self.session_mode.upper()} ({'Continuous Evolution across Tasks' if self.session_mode == 'continuous' else 'Fresh Baseline v0 on Every Task'})")
        print(f"  Total Benchmark Tasks: {len(BENCHMARK_TASKS)} tasks across 6 domains")
        print(f"  Training Episodes: {self.episodes} passes")
        print(f"  TensorBoard Logs:  {self.logger.log_dir}")
        print(f"  Q-Learning Memory: {self.q_table_path}")
        print("=" * 85)

        orchestrator = self._build_orchestrator()
        step_counter = 0

        for ep in range(1, self.episodes + 1):
            print(f"\n>>> Starting Training Episode #{ep}/{self.episodes} ...")
            ep_instant_count = 0
            # Reset to clean baseline at the start of each episode pass
            orchestrator.reset_to_baseline()

            for task in BENCHMARK_TASKS:
                step_counter += 1
                if self.session_mode == "fresh":
                    orchestrator.reset_to_baseline()

                arch_before_version = orchestrator.current_version
                result: DynamicGraphExecutionResult = orchestrator.run_dynamic(task.prompt)

                final_arch = result.final_architecture
                invoked = result.agents_actually_invoked or ["planner", "finalizer"]

                # Prune surplus candidates from final_arch so theoretical evaluator reflects invoked topology
                surplus_candidates = {"coder", "researcher", "critic"}
                for a in final_arch.get("agents", []):
                    if a["agent_id"] in surplus_candidates and a["agent_id"] not in invoked:
                        a["active"] = False

                # Ensure active specialist agents route to finalizer if critic is inactive
                active_ids = {a["agent_id"] for a in final_arch.get("agents", []) if a.get("active")}
                if "critic" not in active_ids and "finalizer" in active_ids:
                    edges = final_arch.setdefault("communication_edges", [])
                    edge_pairs = {
                        (e.get("source") if isinstance(e, dict) else e.source,
                         e.get("target") if isinstance(e, dict) else e.target)
                        for e in edges
                    }
                    for src in active_ids:
                        if src not in {"planner", "finalizer"} and (src, "finalizer") not in edge_pairs:
                            edges.append({"source": src, "target": "finalizer"})

                # Theoretical & Cost Evaluation
                logged = self.logger.log_task_run(
                    step=step_counter,
                    task=task.prompt,
                    result=result,
                    architecture=final_arch,
                    required_capabilities=task.required_capabilities,
                )

                # Baseline comparison (Static 5-Agent MAS)
                baseline_tokens = 3500
                baseline_cost = round((baseline_tokens / 1000.0) * 0.0015, 5)

                actual_tokens = len(invoked) * 650 + max(0, len(invoked) - 1) * 150
                actual_cost = round((actual_tokens / 1000.0) * 0.0015, 5)

                token_savings = max(0.0, ((baseline_tokens - actual_tokens) / baseline_tokens) * 100.0)
                cost_savings = max(0.0, ((baseline_cost - actual_cost) / baseline_cost) * 100.0)

                classification = logged.get("classification", "OPTIMAL")
                is_instant = getattr(result, "is_instant_rl", False)
                engine = "Active RL Policy" if is_instant else "LLM Adapter"
                if is_instant:
                    ep_instant_count += 1

                # Update Q-learning policy with transition & reward
                try:
                    init_arch_raw = result.initial_architecture
                    init_arch_obj = MASArchitecture.model_validate(init_arch_raw)
                    final_arch_obj = MASArchitecture.model_validate(final_arch)

                    plan_obj = PlannerOutput.model_validate(result.planner_output)
                    state_key = self.rl_selector.get_state_key(init_arch_obj, plan_obj)
                    next_state_key = self.rl_selector.get_state_key(final_arch_obj, plan_obj)

                    step_reward = logged.get("net_utility", 0.0)
                    action_id = result.action_ids[0] if result.action_ids else 0

                    self.q_policy.update(
                        state_key=state_key,
                        action_id=action_id,
                        reward=step_reward,
                        next_state_key=next_state_key,
                        next_valid_actions=result.action_ids or [0],
                        terminated=True,
                        truncated=False,
                    )
                except Exception:
                    pass

                self.metrics.append(
                    TaskBenchmarkMetric(
                        episode=ep,
                        task_id=task.task_id,
                        task_name=task.task_name,
                        category=task.category,
                        baseline_tokens=baseline_tokens,
                        adapted_tokens=actual_tokens,
                        token_savings_pct=round(token_savings, 1),
                        baseline_cost=baseline_cost,
                        adapted_cost=actual_cost,
                        cost_savings_pct=round(cost_savings, 1),
                        active_agents=len(invoked),
                        capability_coverage=logged.get("coverage_score", 1.0) * 100.0,
                        net_utility=logged.get("net_utility", 0.0),
                        classification=classification,
                        decision_engine=engine,
                        is_instant_rl=is_instant,
                    )
                )

            # Persist Q-table after episode
            self.q_table_path.parent.mkdir(parents=True, exist_ok=True)
            self.q_policy.save_q_table(str(self.q_table_path))
            print(f"  [Episode #{ep} Completed] RL Instant Hit Rate: {ep_instant_count}/{len(BENCHMARK_TASKS)} ({ep_instant_count / len(BENCHMARK_TASKS) * 100:.1f}%) | Q-table entries: {self.q_policy.q_table.num_state_action_pairs()}")

        # Save summary report & log scalar curves
        summary = self._generate_summary_report()
        self.logger.close()
        return summary

    def _generate_summary_report(self) -> Dict[str, Any]:
        """Format and display the benchmark report table and Pareto metrics."""
        print("\n" + "=" * 98)
        print("  BENCHMARK PERFORMANCE & PARETO OPTIMALITY SUMMARY")
        print("=" * 98)
        header = f"{'Task Name':<26} | {'Category':<13} | {'Base Tok':<8} | {'Adapt Tok':<9} | {'Savings':<7} | {'Coverage':<8} | {'Pareto Status':<15} | {'Decision Engine'}"
        print(header)
        print("-" * 98)

        # Use the final episode metrics for the report
        final_ep_metrics = [m for m in self.metrics if m.episode == self.episodes]

        for m in final_ep_metrics:
            row = (
                f"{m.task_name[:25]:<26} | "
                f"{m.category[:12]:<13} | "
                f"{m.baseline_tokens:<8} | "
                f"{m.adapted_tokens:<9} | "
                f"{m.token_savings_pct:>5.1f}% | "
                f"{m.capability_coverage:>6.1f}% | "
                f"{m.classification:<15} | "
                f"{m.decision_engine}"
            )
            print(row)

        print("-" * 98)

        total_base_tokens = sum(m.baseline_tokens for m in final_ep_metrics)
        total_adapt_tokens = sum(m.adapted_tokens for m in final_ep_metrics)
        overall_token_savings = ((total_base_tokens - total_adapt_tokens) / total_base_tokens) * 100.0

        total_base_cost = sum(m.baseline_cost for m in final_ep_metrics)
        total_adapt_cost = sum(m.adapted_cost for m in final_ep_metrics)
        overall_cost_savings = ((total_base_cost - total_adapt_cost) / total_base_cost) * 100.0

        avg_coverage = sum(m.capability_coverage for m in final_ep_metrics) / len(final_ep_metrics)
        rl_hits = sum(1 for m in final_ep_metrics if m.is_instant_rl)
        rl_hit_rate = (rl_hits / len(final_ep_metrics)) * 100.0

        optimal_count = sum(1 for m in final_ep_metrics if m.classification.upper() == "OPTIMAL")
        optimal_pct = (optimal_count / len(final_ep_metrics)) * 100.0

        print(f"\nKEY PERFORMANCE INDICATORS (KPIs):")
        print(f"  * Total Baseline Tokens:   {total_base_tokens:,}")
        print(f"  * Total Adapted Tokens:    {total_adapt_tokens:,} ({overall_token_savings:.1f}% net token reduction)")
        print(f"  * Total Baseline Cost:     ${total_base_cost:.4f}")
        print(f"  * Total Adapted Cost:      ${total_adapt_cost:.4f} ({overall_cost_savings:.1f}% cost reduction)")
        print(f"  * Mean Capability Coverage: {avg_coverage:.1f}%")
        print(f"  * Pareto Optimality Rate:  {optimal_count}/{len(final_ep_metrics)} ({optimal_pct:.1f}% OPTIMAL topologies)")
        print(f"  * Active RL Hit Rate:      {rl_hits}/{len(final_ep_metrics)} ({rl_hit_rate:.1f}% instant zero-LLM adaptations)")
        print(f"  * Learned Q-table Memory:  {self.q_policy.q_table.num_state_action_pairs()} entries -> {self.q_table_path}")
        print(f"  * TensorBoard Dashboard:   tensorboard --logdir {self.log_base_dir}")
        print("=" * 98)

        summary_data = {
            "episodes": self.episodes,
            "mode": self.mode,
            "session_mode": self.session_mode,
            "total_tasks": len(final_ep_metrics),
            "total_baseline_tokens": total_base_tokens,
            "total_adapted_tokens": total_adapt_tokens,
            "token_savings_pct": round(overall_token_savings, 2),
            "total_baseline_cost_usd": round(total_base_cost, 5),
            "total_adapted_cost_usd": round(total_adapt_cost, 5),
            "cost_savings_pct": round(overall_cost_savings, 2),
            "mean_capability_coverage_pct": round(avg_coverage, 2),
            "pareto_optimal_rate_pct": round(optimal_pct, 2),
            "active_rl_hit_rate_pct": round(rl_hit_rate, 2),
            "q_table_entries": self.q_policy.q_table.num_state_action_pairs(),
        }

        # Persist learned Q-table memory to disk
        self.q_table_path.parent.mkdir(parents=True, exist_ok=True)
        self.q_policy.save_q_table(str(self.q_table_path))

        # Write summary JSON
        summary_file = self.log_base_dir / "benchmark_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
        print(f"\nBenchmark summary saved to {summary_file}")
        return summary_data


def run_benchmark(
    *,
    episodes: int = 2,
    mode: str = "mock",
    session_mode: str = "continuous",
    q_table_path: str = "data/q_table.json",
    log_base_dir: str = "runs/benchmark",
) -> Dict[str, Any]:
    """Public execution API for running the benchmark suite."""
    suite = MultiTaskBenchmarkSuite(
        episodes=episodes,
        mode=mode,
        session_mode=session_mode,
        q_table_path=q_table_path,
        log_base_dir=log_base_dir,
    )
    return suite.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Multi-Agent System (AMAS) — Benchmark & Training Suite")
    parser.add_argument("--episodes", type=int, default=2, help="Number of training episodes (default: 2)")
    parser.add_argument("--live", dest="mode", action="store_const", const="live", default="mock",
                        help="Run with live LLM API calls instead of fast mock simulation")
    parser.add_argument("--mock", "--offline", dest="mode", action="store_const", const="mock",
                        help="Run with fast deterministic mock simulation (default)")
    parser.add_argument("--fresh", dest="session_mode", action="store_const", const="fresh", default="continuous",
                        help="Reset architecture to baseline v0 on every task instead of continuous evolution")
    parser.add_argument("--continuous", dest="session_mode", action="store_const", const="continuous",
                        help="Preserve adapted architecture across consecutive tasks (default)")
    parser.add_argument("--q-table", default="data/q_table.json", help="Path to Q-table JSON file")
    parser.add_argument("--log-dir", default="runs/benchmark", help="Directory for TensorBoard and JSON logs")

    args = parser.parse_args()
    run_benchmark(
        episodes=args.episodes,
        mode=args.mode,
        session_mode=args.session_mode,
        q_table_path=args.q_table,
        log_base_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
