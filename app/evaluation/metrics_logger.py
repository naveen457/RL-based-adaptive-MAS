"""TensorBoard & Structured Metrics Logger for Adaptive MAS.

Persists all task runs, architecture adaptations, RL actions, and cost metrics
to disk in TensorBoard event format and structured JSONL format.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.architecture.models import MASArchitecture
from app.evaluation.cost_tracker import CostTracker
from app.evaluation.theoretical_evaluator import TheoreticalArchitectureEvaluator
from app.graph.visualizer import render_langgraph_ascii, render_mermaid_graph


class MetricsLogger:
    """Logs adaptive MAS metrics to TensorBoard and persistent JSONL records."""

    def __init__(
        self,
        log_base_dir: str = "runs",
        session_id: Optional[str] = None,
    ) -> None:
        self.log_base_dir = Path(log_base_dir)
        self.session_id = session_id or datetime.datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = self.log_base_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.session_dir / "metrics.jsonl"
        self.global_jsonl_path = self.log_base_dir / "all_runs.jsonl"

        self._writer: Any = None
        self._init_tensorboard()

        self.evaluator = TheoreticalArchitectureEvaluator()

        from app.memory.mongo_client import get_mongo_db
        self.db = get_mongo_db(required=True)
        self.runs_collection = self.db["runs"] if self.db is not None else None

    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard SummaryWriter safely."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(self.session_dir))
        except Exception:
            self._writer = None

    @property
    def log_dir(self) -> str:
        return str(self.session_dir)

    def log_task_run(
        self,
        step: int,
        task: str,
        result: Any,
        architecture: MASArchitecture | Dict[str, Any],
        required_capabilities: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        reward_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record and log all metrics for one task execution."""
        # Normalize architecture to MASArchitecture if needed
        if isinstance(architecture, dict):
            arch_obj = MASArchitecture.model_validate(architecture)
        else:
            arch_obj = architecture

        # Theoretical and Cost Evaluation
        evaluation = self.evaluator.evaluate(
            arch_obj,
            required_capabilities=required_capabilities,
        )
        cost_profile = CostTracker.profile_architecture(
            arch_obj,
            required_capabilities=required_capabilities,
            evaluator=self.evaluator,
        )

        # Extract result metadata
        if hasattr(result, "serialize"):
            res_dict = result.serialize()
        elif isinstance(result, dict):
            res_dict = result
        else:
            res_dict = {}

        accepted_actions = res_dict.get("accepted_actions", [])
        rejected_actions = res_dict.get("rejected_actions", [])
        invoked_agents = res_dict.get("agents_actually_invoked", [])
        arch_version = res_dict.get("architecture_version", 0)
        final_response = res_dict.get("final_response", "")

        # 1. Log to TensorBoard (if writer initialized)
        if self._writer is not None:
            # Numerical Metrics
            self._writer.add_scalar("Metrics/Tokens", cost_profile["estimated_tokens"], step)
            self._writer.add_scalar("Metrics/Cost_USD", cost_profile["estimated_cost_usd"], step)
            self._writer.add_scalar("Metrics/ActiveAgentsCount", evaluation.active_agent_count, step)
            self._writer.add_scalar("Metrics/CriticalPathLength", evaluation.topology.critical_path_length, step)
            self._writer.add_scalar("Metrics/NetUtility", evaluation.pareto.net_utility, step)
            self._writer.add_scalar("Metrics/CapabilityCoverage", evaluation.alignment.coverage_score, step)
            self._writer.add_scalar("Metrics/OverallScore", evaluation.overall_score, step)
            self._writer.add_scalar("Metrics/ArchitectureVersion", arch_version, step)
            self._writer.add_scalar("RL_Actions/AcceptedCount", len(accepted_actions), step)
            self._writer.add_scalar("RL_Actions/RejectedCount", len(rejected_actions), step)

            # Agent Activation Status (Binary 1/0 flags)
            for agent_id in ["planner", "coder", "researcher", "critic", "tool_executor", "finalizer"]:
                is_active = 1.0 if agent_id in arch_obj.active_agent_ids else 0.0
                self._writer.add_scalar(f"ActiveAgents/{agent_id}", is_active, step)
                is_invoked = 1.0 if agent_id in invoked_agents else 0.0
                self._writer.add_scalar(f"InvokedAgents/{agent_id}", is_invoked, step)

            # Text Summaries
            ascii_graph = render_langgraph_ascii(arch_obj)
            mermaid_graph = render_mermaid_graph(arch_obj)
            task_summary = (
                f"### Task: {task}\n\n"
                f"- **Classification:** {cost_profile['classification'].upper()}\n"
                f"- **Tokens:** {cost_profile['estimated_tokens']:,} | **Cost:** ${cost_profile['estimated_cost_usd']:.5f}\n"
                f"- **Active Agents:** {', '.join(cost_profile['active_agents'])}\n"
                f"- **Actions Applied:** {len(accepted_actions)}\n\n"
                f"#### Workflow Mermaid:\n```mermaid\n{mermaid_graph}\n```"
            )
            self._writer.add_text("Execution/Summary", task_summary, step)
            self._writer.add_text("Execution/AsciiGraph", f"```\n{ascii_graph}\n```", step)
            self._writer.flush()

        # 2. Append to structured JSONL logs
        log_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "step": step,
            "session_id": self.session_id,
            "user_id": str(user_id) if user_id else None,
            "thread_id": str(thread_id) if thread_id else None,
            "task": task,
            "required_capabilities": required_capabilities or [],
            "architecture_id": arch_obj.architecture_id,
            "architecture_version": arch_version,
            "active_agents": sorted(list(arch_obj.active_agent_ids)),
            "invoked_agents": invoked_agents,
            "accepted_actions": accepted_actions,
            "rejected_actions": rejected_actions,
            "estimated_tokens": cost_profile["estimated_tokens"],
            "estimated_cost_usd": cost_profile["estimated_cost_usd"],
            "critical_path_length": evaluation.topology.critical_path_length,
            "classification": cost_profile["classification"],
            "coverage_score": evaluation.alignment.coverage_score,
            "surplus_agents": evaluation.alignment.surplus_agents,
            "has_cycles": evaluation.topology.has_cycles,
            "net_utility": evaluation.pareto.net_utility,
            "overall_score": evaluation.overall_score,
            "final_response": final_response,
        }

        if reward_info:
            log_record["total_reward"] = reward_info.get("total_reward", 0.0)
            log_record["r_arch"] = reward_info.get("r_arch", 0.0)
            log_record["r_resp"] = reward_info.get("r_resp", 0.0)

        # Write to session log
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record) + "\n")

        # Write to global history log
        with open(self.global_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record) + "\n")

        # Write to MongoDB Atlas
        if self.runs_collection is not None:
            try:
                self.runs_collection.insert_one(dict(log_record))
            except Exception as e:
                from app.memory.mongo_client import check_mongo_network_error
                check_mongo_network_error(e)

        return log_record

    def get_user_metrics(
        self,
        user_id: str,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve and aggregate metrics strictly isolated to a specific user_id."""
        if not user_id:
            return {
                "user_id": None,
                "thread_id": thread_id,
                "summary": {
                    "total_runs": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "avg_coverage": 0.0,
                    "avg_net_utility": 0.0,
                    "avg_critical_path": 0.0,
                    "avg_reward": 0.0,
                },
                "agent_invocations": {},
                "complexity_breakdown": {},
                "runs": [],
            }

        records: List[Dict[str, Any]] = []

        # 1. Fetch from MongoDB Atlas runs collection
        if self.runs_collection is not None:
            try:
                query: Dict[str, Any] = {"user_id": str(user_id)}
                if thread_id:
                    query["thread_id"] = str(thread_id)
                cursor = self.runs_collection.find(query).sort("step", -1).limit(100)
                for doc in cursor:
                    doc_copy = dict(doc)
                    if "_id" in doc_copy:
                        doc_copy["_id"] = str(doc_copy["_id"])
                    records.append(doc_copy)
            except Exception as e:
                from app.memory.mongo_client import check_mongo_network_error
                check_mongo_network_error(e)

        # 2. Fallback to local structured JSONL logs if MongoDB returned nothing or unavailable
        if not records and self.global_jsonl_path.exists():
            try:
                with open(self.global_jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            if str(rec.get("user_id")) == str(user_id):
                                if not thread_id or str(rec.get("thread_id")) == str(thread_id):
                                    records.append(rec)
                        except Exception:
                            continue
                records.reverse()
                records = records[:100]
            except Exception:
                pass

        total_runs = len(records)
        total_tokens = sum(r.get("estimated_tokens", 0) for r in records)
        total_cost_usd = round(sum(r.get("estimated_cost_usd", 0.0) for r in records), 5)
        avg_coverage = (
            round(sum(r.get("coverage_score", 0.0) for r in records) / total_runs, 3)
            if total_runs > 0
            else 0.0
        )
        avg_net_utility = (
            round(sum(r.get("net_utility", 0.0) for r in records) / total_runs, 3)
            if total_runs > 0
            else 0.0
        )
        avg_critical_path = (
            round(sum(r.get("critical_path_length", 0.0) for r in records) / total_runs, 2)
            if total_runs > 0
            else 0.0
        )
        rewards = [r.get("total_reward") for r in records if r.get("total_reward") is not None]
        avg_reward = (
            round(sum(rewards) / len(rewards), 3)
            if rewards
            else 0.0
        )

        agent_invocations: Dict[str, int] = {
            "planner": 0,
            "researcher": 0,
            "coder": 0,
            "critic": 0,
            "tool_executor": 0,
            "finalizer": 0,
        }
        for r in records:
            for ag in r.get("invoked_agents", []):
                ag_str = str(ag).lower()
                agent_invocations[ag_str] = agent_invocations.get(ag_str, 0) + 1

        complexity_breakdown: Dict[str, int] = {
            "simple": 0,
            "moderate": 0,
            "complex": 0,
        }
        for r in records:
            cls_name = str(r.get("classification") or "simple").lower()
            complexity_breakdown[cls_name] = complexity_breakdown.get(cls_name, 0) + 1

        # Calculate per-query comparison fields for each actual run submitted by this user
        formatted_runs = []
        for r in records[:50]:
            task_str = str(r.get("task", ""))
            short_task = (task_str[:77] + "...") if len(task_str) > 80 else task_str
            tokens_used = r.get("estimated_tokens", 0)
            cost_used = r.get("estimated_cost_usd", 0.0)
            hops_used = r.get("critical_path_length", 5)
            cov_score = r.get("coverage_score", 1.0)
            net_util = r.get("net_utility", 0.0)

            tok_saved = max(0, 3500 - tokens_used)
            tok_savings_pct = round((tok_saved / 3500.0) * 100.0, 1) if tokens_used < 3500 else 0.0
            cost_saved = round(max(0.0, 0.00525 - cost_used), 5)
            cost_savings_pct = round(((0.00525 - cost_used) / 0.00525) * 100.0, 1) if cost_used < 0.00525 else 0.0
            hop_reduction_pct = round(((5.0 - hops_used) / 5.0) * 100.0, 1) if hops_used < 5 else 0.0

            formatted_runs.append({
                "step": r.get("step"),
                "timestamp": r.get("timestamp"),
                "thread_id": r.get("thread_id"),
                "task": short_task,
                "classification": r.get("classification", "standard"),
                "estimated_tokens": tokens_used,
                "static_tokens": 3500,
                "tokens_saved": tok_saved,
                "token_savings_pct": tok_savings_pct,
                "estimated_cost_usd": cost_used,
                "static_cost_usd": 0.00525,
                "cost_saved_usd": cost_saved,
                "cost_savings_pct": cost_savings_pct,
                "coverage_score": cov_score,
                "static_coverage": 1.0,
                "net_utility": net_util,
                "static_utility": 0.15,
                "critical_path_length": hops_used,
                "static_hops": 5,
                "latency_reduction_pct": hop_reduction_pct,
                "active_agents": r.get("active_agents", []),
                "invoked_agents": r.get("invoked_agents", []),
                "architecture_version": r.get("architecture_version", 0),
                "total_reward": r.get("total_reward", None),
            })

        baseline_tokens_total = total_runs * 3500
        baseline_cost_total = round(total_runs * 0.00525, 5)
        overall_token_savings_pct = (
            round(((baseline_tokens_total - total_tokens) / baseline_tokens_total) * 100.0, 1)
            if baseline_tokens_total > 0
            else 0.0
        )
        overall_cost_savings_pct = (
            round(((baseline_cost_total - total_cost_usd) / baseline_cost_total) * 100.0, 1)
            if baseline_cost_total > 0
            else 0.0
        )
        overall_latency_reduction_pct = (
            round(((5.0 - avg_critical_path) / 5.0) * 100.0, 1)
            if (total_runs > 0 and avg_critical_path > 0 and avg_critical_path < 5.0)
            else 0.0
        )

        # Dynamic comparison metrics derived 100% from this user's actual tasks
        user_comparison = {
            "baseline": {
                "tokens": baseline_tokens_total,
                "cost_usd": baseline_cost_total,
                "avg_hops": 5.0 if total_runs > 0 else 0.0,
                "coverage_pct": 100.0 if total_runs > 0 else 0.0,
                "avg_utility": 0.15 if total_runs > 0 else 0.0,
                "architecture_type": "Static Pipeline (5 Agents: Planner → Researcher → Coder → Critic → Finalizer)",
            },
            "dynamic": {
                "tokens": total_tokens,
                "cost_usd": total_cost_usd,
                "avg_hops": avg_critical_path,
                "coverage_pct": round(avg_coverage * 100.0, 1),
                "avg_utility": avg_net_utility,
                "architecture_type": "Adaptive AMAS (Context-Driven Active Agents)",
            },
            "savings": {
                "token_savings_pct": max(0.0, overall_token_savings_pct),
                "tokens_saved": max(0, baseline_tokens_total - total_tokens),
                "cost_savings_pct": max(0.0, overall_cost_savings_pct),
                "cost_saved_usd": round(max(0.0, baseline_cost_total - total_cost_usd), 5),
                "latency_reduction_pct": max(0.0, overall_latency_reduction_pct),
            },
            "metrics": [
                {
                    "id": "tokens",
                    "label": "Token Consumption",
                    "unit": "tokens",
                    "static_val": baseline_tokens_total,
                    "dynamic_val": total_tokens,
                    "delta_pct": -overall_token_savings_pct if total_runs > 0 else 0.0,
                    "desc": "Total tokens consumed across your queries versus static 5-agent execution",
                },
                {
                    "id": "latency",
                    "label": "Execution Latency / Critical Path",
                    "unit": "hops",
                    "static_val": 5.0 if total_runs > 0 else 0.0,
                    "dynamic_val": avg_critical_path,
                    "delta_pct": -overall_latency_reduction_pct if total_runs > 0 else 0.0,
                    "desc": "Average sequential agent hops required to answer your prompts",
                },
                {
                    "id": "cost",
                    "label": "Operating Cost",
                    "unit": "USD ($)",
                    "static_val": baseline_cost_total,
                    "dynamic_val": total_cost_usd,
                    "delta_pct": -overall_cost_savings_pct if total_runs > 0 else 0.0,
                    "desc": "Estimated dollar cost for your queries versus the static baseline",
                },
                {
                    "id": "accuracy",
                    "label": "Capability Coverage (Accuracy)",
                    "unit": "% alignment",
                    "static_val": 100.0 if total_runs > 0 else 0.0,
                    "dynamic_val": round(avg_coverage * 100.0, 1),
                    "delta_pct": round((avg_coverage - 1.0) * 100.0, 1) if total_runs > 0 else 0.0,
                    "desc": "Intent capability alignment achieved for your prompts with zero bloat",
                },
                {
                    "id": "utility",
                    "label": "Pareto Net Utility",
                    "unit": "score",
                    "static_val": 0.15 if total_runs > 0 else 0.0,
                    "dynamic_val": avg_net_utility,
                    "delta_pct": round((avg_net_utility - 0.15) * 100.0, 1) if total_runs > 0 else 0.0,
                    "desc": "Multi-objective efficiency score balancing accuracy against unnecessary agent bloat",
                },
            ],
        }

        return {
            "user_id": str(user_id),
            "thread_id": thread_id,
            "summary": {
                "total_runs": total_runs,
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost_usd,
                "avg_coverage": avg_coverage,
                "avg_net_utility": avg_net_utility,
                "avg_critical_path": avg_critical_path,
                "avg_reward": avg_reward,
            },
            "user_comparison": user_comparison,
            "agent_invocations": agent_invocations,
            "complexity_breakdown": complexity_breakdown,
            "runs": formatted_runs,
        }

    def log_reward(
        self,
        *,
        step: int,
        r_arch: float,
        r_resp: float,
        total_reward: float,
        components: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log dual-objective reward signals to TensorBoard."""
        if self._writer is not None:
            self._writer.add_scalar("Reward/Total", total_reward, step)
            self._writer.add_scalar("Reward/Architecture", r_arch, step)
            self._writer.add_scalar("Reward/ResponseQuality", r_resp, step)
            if components:
                for k, v in components.items():
                    if isinstance(v, (int, float)):
                        self._writer.add_scalar(f"RewardComponents/{k}", float(v), step)
            self._writer.flush()

    def close(self) -> None:
        """Flush and close all writer handles."""
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
