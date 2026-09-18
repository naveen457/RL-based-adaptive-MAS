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

        # Write to session log
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record) + "\n")

        # Write to global history log
        with open(self.global_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record) + "\n")

        return log_record

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
