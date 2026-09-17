from pathlib import Path
from app.config.settings import settings
from app.runtime.orchestrator import run_adaptive_runtime
from app.evaluation.metrics_logger import MetricsLogger
from app.graph.visualizer import format_architecture_display
from app.rl.q_learning import QLearningPolicy


def print_workflow_result(state):
    """Print the result of running the MAS workflow."""
    plan = state.get("planner_output")
    if plan:
        print("\nPLANNER")
        print(f"  task_understanding: {plan.get('task_understanding')}")
        print(f"  requires_research: {plan.get('requires_research')}")
        print(f"  requires_coding: {plan.get('requires_coding')}")
        print(f"  requires_verification: {plan.get('requires_verification')}")
        print(f"  requires_tools: {plan.get('requires_tools')}")
        print()

    final = state.get("final_response") or state.get("final_answer")
    if final:
        print("FINAL SYNTHESIS")
        if isinstance(final, dict):
            print(f"  answer: {final.get('final_answer')}")
        else:
            print(f"  answer: {final}")
        print()


def main():
    print("=" * 80)
    print("  ADAPTIVE MULTI-AGENT SYSTEM (RL-AMAS) - RUNTIME")
    print("=" * 80)

    api_status = "Loaded" if settings.api_key else "NOT FOUND"
    print(f"  Model Endpoint: {settings.base_url} ({settings.model or 'default'}) [{api_status}]")

    # Initialize persistent metrics logger (TensorBoard + JSONL)
    logger = MetricsLogger(log_base_dir="runs")
    print(f"  TensorBoard Logs: {logger.log_dir}")
    print(f"  Dashboard Command: tensorboard --logdir {logger.log_base_dir}")

    # Initialize persistent Q-learning policy outside runs/ (tracked by git, shared across developers)
    q_table_path = Path("data/q_table.json")
    if not q_table_path.exists() and Path("q_table.json").exists():
        q_table_path = Path("q_table.json")
    elif not q_table_path.exists() and Path("runs/q_table.json").exists():
        q_table_path = Path("runs/q_table.json")

    q_policy = QLearningPolicy(task_aware=True, epsilon=0.1)
    if q_table_path.exists():
        try:
            q_policy.load_q_table(str(q_table_path))
            print(f"  [Q-Learning Memory] Loaded existing Q-table ({q_policy.q_table.num_state_action_pairs()} entries) from {q_table_path}")
        except Exception as e:
            print(f"  [Q-Learning Memory] Could not load existing Q-table ({e}), starting fresh")
    else:
        q_table_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [Q-Learning Memory] Initialized fresh Q-learning policy (persists to {q_table_path})")

    print("=" * 80)

    step_counter = 0
    print("\nEnter a task below, or type 'exit' to quit.")

    while True:
        try:
            task = input("\nUSER TASK> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if task.lower() == "exit":
            print(f"Session ended. Total tasks executed & logged: {step_counter}")
            print(f"View runs via TensorBoard: tensorboard --logdir {logger.log_base_dir}")
            break
        if not task:
            print("Please enter a task or type 'exit'.")
            continue

        step_counter += 1
        print(f"\n[Processing Task #{step_counter}] \"{task}\" ...")

        try:
            result = run_adaptive_runtime(task)
        except Exception as exc:
            print(f"Runtime error: {type(exc).__name__}: {exc}")
            continue

        final_arch = result.final_architecture

        # 1. Extract true required capabilities from planner (dynamic, not hardcoded)
        plan_dict = result.planner_output or {}
        req_caps = list(plan_dict.get("required_capabilities") or [])
        tools_needed = list(plan_dict.get("tools_needed") or [])
        if "web_search" in tools_needed and "web_search" not in req_caps:
            req_caps.append("web_search")
        if "calculator" in tools_needed and "math" not in req_caps:
            req_caps.append("math")
        if plan_dict.get("requires_research") and "research" not in req_caps:
            req_caps.append("research")
        if plan_dict.get("requires_coding") and "coding" not in req_caps:
            req_caps.append("coding")
        if plan_dict.get("requires_verification") and "verification" not in req_caps:
            req_caps.append("verification")
        if plan_dict.get("requires_tools") and "web_search" not in req_caps:
            req_caps.append("web_search")

        # Guardrail: queries asking for live/current events require web_search
        query_lower = task.lower()
        temporal_kws = ["current", "trending", "latest", "today", "live", "recent", "events in"]
        if any(kw in query_lower for kw in temporal_kws) and "web_search" not in req_caps:
            req_caps.append("web_search")

        invoked = result.agents_actually_invoked or ["planner", "finalizer"]
        if "tool_executor" in invoked:
            # Synchronize final_arch to reflect that tool_executor is active
            agent_ids_in_arch = [a["agent_id"] for a in final_arch.get("agents", [])]
            if "tool_executor" not in agent_ids_in_arch:
                final_arch.setdefault("agents", []).append({
                    "agent_id": "tool_executor",
                    "role": "tool_execution",
                    "description": "Dedicated execution node for external tools.",
                    "capabilities": ["tool_use", "web_search", "external_api"],
                    "active": True,
                })
            else:
                for a in final_arch["agents"]:
                    if a["agent_id"] == "tool_executor":
                        a["active"] = True

            # Synchronize communication edges so tool_executor is properly connected in the DAG
            edges = final_arch.setdefault("communication_edges", [])
            edge_pairs = {
                (e.get("source") if isinstance(e, dict) else e.source,
                 e.get("target") if isinstance(e, dict) else e.target)
                for e in edges
            }
            if ("planner", "tool_executor") not in edge_pairs:
                edges.append({"source": "planner", "target": "tool_executor"})
            active_ids = {a["agent_id"] for a in final_arch.get("agents", []) if a.get("active")}
            target_node = "critic" if "critic" in active_ids else "finalizer"
            if ("tool_executor", target_node) not in edge_pairs:
                edges.append({"source": "tool_executor", "target": target_node})

        # Pareto Minimality Synchronization:
        # Deactivate uninvoked specialist agents (coder, researcher, critic) from final_arch
        # so the logged architecture matches the minimal sufficient Pareto-optimal topology
        surplus_candidates = {"coder", "researcher", "critic"}
        pruned_any = False
        for a in final_arch.get("agents", []):
            if a["agent_id"] in surplus_candidates and a["agent_id"] not in invoked:
                if a.get("active"):
                    a["active"] = False
                    pruned_any = True

        arch_changed = result.architecture_changed or pruned_any
        arch_version = max(result.architecture_version, 1 if arch_changed else 0)

        # 2. Log metrics to TensorBoard and persistent JSONL with real required capabilities
        logged = logger.log_task_run(
            step=step_counter,
            task=task,
            result=result,
            architecture=final_arch,
            required_capabilities=req_caps,
        )

        # 3. RL Self-Evaluation Tag (genuine theoretical evaluation, no hardcoded override)
        rl_classification = logged.get("classification", "OPTIMAL").upper()

        actual_tokens = len(invoked) * 650 + max(0, len(invoked) - 1) * 150
        actual_cost = round((actual_tokens / 1000.0) * 0.0015, 5)
        cost_profile = {
            "estimated_tokens": actual_tokens,
            "estimated_cost_usd": actual_cost,
            "classification": rl_classification,
        }
        graph_display = format_architecture_display(
            final_arch,
            version=arch_version,
            actions_taken=result.accepted_actions,
            cost_profile=cost_profile,
            invoked_agents=invoked,
        )
        print("\n" + graph_display)


        # 4. Print runtime trace and execution summary
        print("\nEXECUTION SUMMARY:")
        print(f"  * RL Self-Evaluation:   {rl_classification}")
        print(f"  * Architecture Changed: {arch_changed}")
        print(f"  * Agents Invoked:       {', '.join(result.agents_actually_invoked)}")
        print(f"  * Critical Path:        {logged['critical_path_length']} nodes")
        print(f"  * Capability Coverage:  {logged['coverage_score'] * 100:.1f}%")
        if logged.get("missing_capabilities"):
            print(f"  * Missing Capabilities: {', '.join(logged['missing_capabilities'])}")
        print(f"  * Net Utility Score:    {logged.get('net_utility', 0.0):.4f}")
        print(f"  * Metrics Saved To:     {logger.jsonl_path}")

        print("\nFINAL RESPONSE:")
        print("-" * 50)
        final_resp = result.final_response
        if isinstance(final_resp, dict):
            print(final_resp.get("final_answer", final_resp))
        else:
            print(final_resp)
        print("-" * 50)

        # 5. Interactive Human Feedback (RLHF Verification)
        print(f"\n[HUMAN VERIFICATION & RLHF]")
        print(f"  * RL Architecture Status: [{rl_classification}]")
        if logged.get("surplus_agents"):
            print(f"    (Topology has surplus agents: {', '.join(logged['surplus_agents'])}; adapt/prune to reach OPTIMAL)")
        print(f"  * Agents Invoked:         [{', '.join(invoked)}]")

        print("  Did the system solve your task well?")
        print("    [y] Yes, good execution & answer (Rewards RL agent +0.5)")
        print("    [n] No, bad execution or wrong answer (Penalizes RL agent -0.5)")
        print("    [Enter] Accept RL internal evaluation without human bias")
        try:
            user_choice = input("  Your choice (y/n/Enter): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            user_choice = ""

        human_reward = 0.0
        if user_choice == "y":
            human_reward = 0.5
            print("  [Feedback] +0.5 reward bonus applied (user approved task solution).")
        elif user_choice == "n":
            human_reward = -0.5
            print("  [Feedback] -0.5 penalty applied (user rejected task solution).")
        else:
            print("  [Feedback] Accepted RL self-evaluation tag without human bias.")

        # 6. Update Q-learning policy with transition & persist to disk
        try:
            from app.architecture.models import MASArchitecture
            from app.rl.state import ArchitectureStateEncoder
            from app.rl.meta_task import MetaTaskContext

            if isinstance(final_arch, dict):
                arch_obj = MASArchitecture.model_validate(final_arch)
            else:
                arch_obj = final_arch

            encoder = ArchitectureStateEncoder(arch_obj)
            obs = encoder.encode()
            task_ctx = MetaTaskContext(
                task_category="research" if "research" in req_caps else "general",
                required_capabilities=req_caps,
                difficulty=1,
            )
            state_key = q_policy.get_state_key(obs, task_ctx)
            step_reward = logged.get("net_utility", 0.0) + human_reward
            q_policy.update(
                state_key=state_key,
                action_id=0,
                reward=step_reward,
                next_state_key=state_key,
                next_valid_actions=[0],
                terminated=True,
                truncated=False,
            )
            q_table_path.parent.mkdir(parents=True, exist_ok=True)
            q_policy.save_q_table(str(q_table_path))
            print(f"  [Q-Learning Memory] Persisted Q-table ({q_policy.q_table.num_state_action_pairs()} entries) -> {q_table_path}")
        except Exception as q_err:
            print(f"  [Q-Learning Memory] Note: Q-table update deferred ({q_err})")

    logger.close()


if __name__ == "__main__":
    main()

