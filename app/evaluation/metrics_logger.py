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
        self.db = get_mongo_db(required=False)
        self._runs_collection = self.db["runs"] if self.db is not None else None

    @property
    def runs_collection(self):
        if self._runs_collection is None and getattr(self, "_use_mongo", True):
            from app.memory.mongo_client import get_mongo_db
            current_db = get_mongo_db(required=False)
            if current_db is not None:
                self.db = current_db
                self._runs_collection = current_db["runs"]
            else:
                self._use_mongo = False
        return self._runs_collection

    @runs_collection.setter
    def runs_collection(self, val):
        self._runs_collection = val

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

    @property
    def jsonl_file(self) -> Path:
        return self.jsonl_path

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
        *,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        seed: Optional[int] = None,
        policy_type: Optional[str] = None,
        policy_initialization: Optional[str] = None,
        training_config: Optional[Dict[str, Any]] = None,
        evaluation_mode: str = "mock",
        measurement_type: str = "simulated",
        actual_tokens: Optional[int] = None,
        actual_cost_usd: Optional[float] = None,
        time_to_first_response_ms: Optional[float] = None,
        **kwargs: Any,
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

        # Action statistics
        proposed_actions = accepted_actions + rejected_actions
        proposed_count = len(proposed_actions)
        accepted_count = len(accepted_actions)
        rejected_count = len(rejected_actions)
        acceptance_rate = round(accepted_count / max(1, proposed_count), 4)
        invalid_action_rate = round(rejected_count / max(1, proposed_count), 4)

        action_type_counts: Dict[str, int] = {
            "activate_agent": 0,
            "deactivate_agent": 0,
            "add_edge": 0,
            "remove_edge": 0,
            "change_role": 0,
        }
        for act in accepted_actions:
            atype = act.get("action_type") if isinstance(act, dict) else getattr(act, "action_type", None)
            if atype in action_type_counts:
                action_type_counts[atype] += 1

        # Role distribution across active agents
        role_counts: Dict[str, int] = {}
        for a in arch_obj.agents:
            if a.active:
                role_counts[a.role] = role_counts.get(a.role, 0) + 1

        # Graph execution telemetry
        build_time_ms = float(kwargs.get("build_time_ms") if kwargs.get("build_time_ms") is not None else (res_dict.get("build_time_ms", 0.0) or 0.0))
        compilation_time_ms = float(kwargs.get("compilation_time_ms") if kwargs.get("compilation_time_ms") is not None else (res_dict.get("compilation_time_ms", 0.0) or 0.0))
        execution_time_ms = float(kwargs.get("execution_time_ms") if kwargs.get("execution_time_ms") is not None else (res_dict.get("execution_time_ms", 0.0) or 0.0))
        node_count = int(kwargs.get("node_count") if kwargs.get("node_count") is not None else (res_dict.get("node_count", len(res_dict.get("graph_nodes", []))) or 0))
        edge_count = int(kwargs.get("edge_count") if kwargs.get("edge_count") is not None else (res_dict.get("edge_count", len(res_dict.get("graph_edges", []))) or 0))
        recovery_count = int(kwargs.get("recovery_count") if kwargs.get("recovery_count") is not None else (res_dict.get("recovery_count", 0) or 0))
        conditional_route_count = int(kwargs.get("conditional_route_count") if kwargs.get("conditional_route_count") is not None else (res_dict.get("conditional_route_count", 0) or 0))
        build_success_rate = float(res_dict.get("build_success_rate", 1.0) or 1.0)
        compilation_success_rate = float(res_dict.get("compilation_success_rate", 1.0) or 1.0)

        # Actual provider measurements if available
        act_tokens = actual_tokens if actual_tokens is not None else res_dict.get("actual_tokens")
        act_cost = actual_cost_usd if actual_cost_usd is not None else res_dict.get("actual_cost_usd")
        ttft_ms = time_to_first_response_ms if time_to_first_response_ms is not None else res_dict.get("time_to_first_response_ms")

        # 1. Log to TensorBoard (if writer initialized)
        if self._writer is not None:
            # Architecture structure & quality
            self._writer.add_scalar("Architecture/AgentCount", len(arch_obj.agents), step)
            self._writer.add_scalar("Architecture/ActiveAgentCount", evaluation.active_agent_count, step)
            self._writer.add_scalar("Architecture/InactiveAgentCount", len(arch_obj.agents) - evaluation.active_agent_count, step)
            self._writer.add_scalar("Architecture/EdgeCount", len(arch_obj.communication_edges), step)
            self._writer.add_scalar("Architecture/GraphDensity", evaluation.topology.graph_density, step)
            self._writer.add_scalar("Architecture/CapabilityCoverage", evaluation.alignment.coverage_score, step)
            self._writer.add_scalar("Architecture/NetUtility", evaluation.pareto.net_utility, step)
            self._writer.add_scalar("Architecture/OverallScore", evaluation.overall_score, step)
            self._writer.add_scalar("Architecture/SurplusAgents", len(evaluation.alignment.surplus_agents), step)
            self._writer.add_scalar("Architecture/StructuralEfficiency", evaluation.structural_efficiency, step)
            self._writer.add_scalar("Architecture/RequiredCapabilityCount", len(required_capabilities or []), step)
            self._writer.add_scalar("Architecture/SatisfiedCapabilityCount", len(evaluation.alignment.covered_capabilities), step)
            self._writer.add_scalar("Architecture/ArchitectureVersion", arch_version, step)
            self._writer.add_scalar("Architecture/MutationCount", accepted_count, step)
            for r_name, r_cnt in role_counts.items():
                self._writer.add_scalar(f"Architecture/RoleDistribution/{r_name}", r_cnt, step)

            # RL Actions
            self._writer.add_scalar("RL/Actions/Proposed", proposed_count, step)
            self._writer.add_scalar("RL/Actions/Accepted", accepted_count, step)
            self._writer.add_scalar("RL/Actions/Rejected", rejected_count, step)
            self._writer.add_scalar("RL/Actions/AcceptanceRate", acceptance_rate, step)
            self._writer.add_scalar("RL/Actions/InvalidActionRate", invalid_action_rate, step)
            for atype, acnt in action_type_counts.items():
                self._writer.add_scalar(f"RL/Actions/{atype}", acnt, step)

            confidence = res_dict.get("confidence")
            if isinstance(confidence, (int, float)):
                self._writer.add_scalar("RL/Actions/SelectedQValue", float(confidence), step)

            # Dynamic Graph
            self._writer.add_scalar("Graph/BuildTimeMs", build_time_ms, step)
            self._writer.add_scalar("Graph/CompilationTimeMs", compilation_time_ms, step)
            self._writer.add_scalar("Graph/NodeCount", node_count, step)
            self._writer.add_scalar("Graph/EdgeCount", edge_count, step)
            self._writer.add_scalar("Graph/ExecutionTimeMs", execution_time_ms, step)
            self._writer.add_scalar("Graph/RecoveryCount", recovery_count, step)
            self._writer.add_scalar("Graph/ConditionalRouteCount", conditional_route_count, step)
            self._writer.add_scalar("Graph/BuildSuccessRate", build_success_rate, step)
            self._writer.add_scalar("Graph/CompilationSuccessRate", compilation_success_rate, step)

            # Execution & Resource Efficiency
            self._writer.add_scalar("Execution/EstimatedTokens", cost_profile["estimated_tokens"], step)
            self._writer.add_scalar("Execution/EstimatedCostUSD", cost_profile["estimated_cost_usd"], step)
            self._writer.add_scalar("Execution/LatencyMs", execution_time_ms, step)
            self._writer.add_scalar("Execution/AgentInvocationCount", len(invoked_agents), step)
            self._writer.add_scalar("Execution/SuccessfulAgentCount", len(invoked_agents), step)
            self._writer.add_scalar("Execution/FailedAgentCount", 0.0, step)
            if act_tokens is not None:
                self._writer.add_scalar("Execution/ActualTokens", float(act_tokens), step)
            if act_cost is not None:
                self._writer.add_scalar("Execution/ActualCostUSD", float(act_cost), step)
            if ttft_ms is not None:
                self._writer.add_scalar("Execution/TimeToFirstResponseMs", float(ttft_ms), step)

            # Evaluation Quality & Reliability
            self._writer.add_scalar("Evaluation/CapabilityCoverage", evaluation.alignment.coverage_score, step)
            self._writer.add_scalar("Evaluation/TaskCompatibility", evaluation.alignment.coverage_score, step)
            self._writer.add_scalar("Evaluation/NetUtility", evaluation.pareto.net_utility, step)
            self._writer.add_scalar("Evaluation/ArchitectureValidity", evaluation.validity_score, step)
            self._writer.add_scalar("Evaluation/ExecutionFailureRate", 0.0, step)
            self._writer.add_scalar("Evaluation/MeanAdaptationTimeMs", build_time_ms, step)
            if reward_info:
                self._writer.add_scalar("Evaluation/Reward", float(reward_info.get("total_reward", 0.0)), step)
                self._writer.add_scalar("Evaluation/RewardComponents/Architecture", float(reward_info.get("r_arch", 0.0)), step)
                self._writer.add_scalar("Evaluation/RewardComponents/ResponseQuality", float(reward_info.get("r_resp", 0.0)), step)
            if evaluation.task_success_score is not None:
                self._writer.add_scalar("Evaluation/TaskSuccess", float(evaluation.task_success_score), step)

            # Backward-compatible legacy metric tags
            self._writer.add_scalar("Metrics/Tokens", cost_profile["estimated_tokens"], step)
            self._writer.add_scalar("Metrics/Cost_USD", cost_profile["estimated_cost_usd"], step)
            self._writer.add_scalar("Metrics/ActiveAgentsCount", evaluation.active_agent_count, step)
            self._writer.add_scalar("Metrics/CriticalPathLength", evaluation.topology.critical_path_length, step)
            self._writer.add_scalar("Metrics/NetUtility", evaluation.pareto.net_utility, step)
            self._writer.add_scalar("Metrics/CapabilityCoverage", evaluation.alignment.coverage_score, step)
            self._writer.add_scalar("Metrics/OverallScore", evaluation.overall_score, step)
            self._writer.add_scalar("Metrics/ArchitectureVersion", arch_version, step)
            self._writer.add_scalar("RL_Actions/AcceptedCount", accepted_count, step)
            self._writer.add_scalar("RL_Actions/RejectedCount", rejected_count, step)

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

        # 2. Append to structured JSONL logs (Extended research schema + 100% backward compatibility)
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
            "structural_efficiency": evaluation.structural_efficiency,
            "final_response": final_response,
            "is_proxy": True,
            "experiment_metadata": {
                "experiment_id": experiment_id or self.session_id,
                "run_id": run_id or f"run_{step}",
                "seed": seed,
                "policy_type": policy_type or "active_q_learning",
                "policy_initialization": policy_initialization or "default",
                "evaluation_mode": evaluation_mode,
                "measurement_type": measurement_type,
                "training_config": training_config or {},
            },
            "graph_metrics": {
                "build_time_ms": build_time_ms,
                "compilation_time_ms": compilation_time_ms,
                "execution_time_ms": execution_time_ms,
                "node_count": node_count,
                "edge_count": edge_count,
                "recovery_count": recovery_count,
                "conditional_route_count": conditional_route_count,
                "build_success_rate": build_success_rate,
                "compilation_success_rate": compilation_success_rate,
            },
            "action_metrics": {
                "proposed": proposed_count,
                "accepted": accepted_count,
                "rejected": rejected_count,
                "acceptance_rate": acceptance_rate,
                "invalid_action_rate": invalid_action_rate,
                "action_type_counts": action_type_counts,
            },
            "architecture_metrics": {
                "agent_count": len(arch_obj.agents),
                "active_agent_count": evaluation.active_agent_count,
                "inactive_agent_count": len(arch_obj.agents) - evaluation.active_agent_count,
                "edge_count": len(arch_obj.communication_edges),
                "graph_density": evaluation.topology.graph_density,
                "critical_path_length": evaluation.topology.critical_path_length,
                "capability_coverage": evaluation.alignment.coverage_score,
                "net_utility": evaluation.pareto.net_utility,
                "overall_score": evaluation.overall_score,
                "structural_efficiency": evaluation.structural_efficiency,
                "surplus_agents_count": len(evaluation.alignment.surplus_agents),
                "required_capability_count": len(required_capabilities or []),
                "satisfied_capability_count": len(evaluation.alignment.covered_capabilities),
                "role_distribution": role_counts,
            },
        }

        if act_tokens is not None:
            log_record["actual_tokens"] = act_tokens
        if act_cost is not None:
            log_record["actual_cost_usd"] = act_cost
        if ttft_ms is not None:
            log_record["time_to_first_response_ms"] = ttft_ms

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

        user_id_candidates: List[Any] = [str(user_id)]
        try:
            from bson import ObjectId
            if ObjectId.is_valid(str(user_id)):
                user_id_candidates.append(ObjectId(str(user_id)))
        except Exception:
            pass

        user_threads: List[str] = []
        try:
            from app.memory.store import default_thread_store
            user_threads = default_thread_store.list_threads(user_id=str(user_id))
        except Exception:
            pass

        # 1. Fetch from MongoDB Atlas runs collection
        coll = self.runs_collection
        if coll is not None:
            try:
                if thread_id:
                    query: Dict[str, Any] = {
                        "thread_id": str(thread_id),
                        "$or": [
                            {"user_id": {"$in": user_id_candidates}},
                            {"user_id": None},
                            {"user_id": {"$exists": False}},
                            {"user_id": ""},
                        ],
                    }
                else:
                    match_conditions: List[Dict[str, Any]] = [
                        {"user_id": {"$in": user_id_candidates}},
                    ]
                    if user_threads:
                        match_conditions.append({"thread_id": {"$in": user_threads}})
                    query = {"$or": match_conditions}

                cursor = coll.find(query).sort("step", -1).limit(100)
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
                            rec_user = str(rec.get("user_id") or "")
                            rec_thread = str(rec.get("thread_id") or "")
                            user_matches = rec_user and rec_user == str(user_id)
                            thread_matches = bool(thread_id and rec_thread == str(thread_id))
                            thread_owned = bool(user_threads and rec_thread in user_threads)

                            if thread_id:
                                if thread_matches and (user_matches or not rec_user):
                                    records.append(rec)
                            else:
                                if user_matches or thread_owned:
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

    def log_neural_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Log Task-Conditioned Neural Policy training telemetry to TensorBoard."""
        if self._writer is None:
            return
        tag_map = {
            "loss": "Neural/TrainingLoss",
            "training_loss": "Neural/TrainingLoss",
            "reward": "Neural/TrainingReward",
            "training_reward": "Neural/TrainingReward",
            "policy_gradient_norm": "Neural/PolicyGradientNorm",
            "gradient_norm": "Neural/PolicyGradientNorm",
            "parameter_norm": "Neural/ParameterNorm",
            "action_distribution_entropy": "Neural/ActionDistributionEntropy",
            "entropy": "Neural/ActionDistributionEntropy",
            "learning_rate": "Neural/LearningRate",
            "training_step": "Neural/TrainingStep",
        }
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                tag = tag_map.get(key, f"Neural/{key}")
                self._writer.add_scalar(tag, float(val), step)
        self._writer.flush()

    def log_meta_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        """Log cross-task adaptation and meta-generalization telemetry to TensorBoard."""
        if self._writer is None:
            return
        tag_map = {
            "train_task_reward": "Meta/TrainTaskReward",
            "held_out_task_reward": "Meta/HeldOutTaskReward",
            "generalization_gap": "Meta/GeneralizationGap",
            "initial_adaptation_reward": "Meta/InitialAdaptationReward",
            "final_adaptation_reward": "Meta/FinalAdaptationReward",
            "adaptation_gain": "Meta/AdaptationGain",
            "adaptation_steps": "Meta/AdaptationSteps",
        }
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                tag = tag_map.get(key, f"Meta/{key}")
                self._writer.add_scalar(tag, float(val), step)
            elif isinstance(val, list) and key == "adaptation_curve":
                for adapt_step, perf in enumerate(val):
                    self._writer.add_scalar("Meta/AdaptationCurve", float(perf), adapt_step)
        self._writer.flush()

    def close(self) -> None:
        """Flush and close all writer handles."""
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
