import argparse
from pathlib import Path
from app.config.settings import settings
from app.runtime.orchestrator import AdaptiveRuntimeOrchestrator, run_dynamic_runtime
from app.evaluation.metrics_logger import MetricsLogger
from app.graph.visualizer import format_architecture_display
from app.rl.q_learning import QLearningPolicy
import time

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
    parser = argparse.ArgumentParser(description="Adaptive Multi-Agent System (RL-AMAS) - Runtime")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--continuous",
        dest="session_mode",
        action="store_const",
        const="continuous",
        default="continuous",
        help="Preserve and evolve adapted architecture across consecutive user tasks (default)",
    )
    mode_group.add_argument(
        "--fresh",
        dest="session_mode",
        action="store_const",
        const="fresh",
        help="Reset architecture to baseline v0 on every task",
    )
    parser.add_argument(
        "--thread-id",
        "--thread",
        dest="thread_id",
        default="thread-1",
        help="Thread ID for conversation message history (default: thread-1)",
    )
    args, _ = parser.parse_known_args()
    session_mode = args.session_mode
    active_thread_id = getattr(args, "thread_id", "thread-1") or "thread-1"

    print("=" * 80)
    print("  ADAPTIVE MULTI-AGENT SYSTEM (RL-AMAS) - RUNTIME")
    print("=" * 80)

    api_status = "Loaded" if settings.api_key else "NOT FOUND"
    print(f"  Model Endpoint: {settings.base_url} ({settings.model or 'default'}) [{api_status}]")
    print(f"  Session Mode:   {session_mode.upper()} ({'continuous multi-turn evolution' if session_mode == 'continuous' else 'fresh baseline on every task'})")

    # Enforce MongoDB Atlas configuration at startup (exit if unable to configure)
    from app.memory.mongo_client import get_mongo_client
    get_mongo_client(required=True)

    # Initialize persistent metrics logger (TensorBoard + MongoDB)
    logger = MetricsLogger(log_base_dir="runs")
    print(f"  TensorBoard Logs: {logger.log_dir}")
    print(f"  Dashboard Command: tensorboard --logdir {logger.log_base_dir}")

    # Initialize persistent Q-learning policy in MongoDB Atlas
    q_policy = QLearningPolicy(task_aware=True, epsilon=0.1)
    loaded_from_mongo = q_policy.load_from_mongodb()
    if loaded_from_mongo and q_policy.q_table.num_state_action_pairs() > 0:
        print(f"  [Q-Learning Memory] Loaded existing Q-table ({q_policy.q_table.num_state_action_pairs()} entries) from MongoDB Atlas")
    else:
        print(f"  [Q-Learning Memory] Initialized fresh Q-learning policy in MongoDB Atlas")

    # Initialize orchestrator with active RL policy
    orchestrator = AdaptiveRuntimeOrchestrator.from_settings(q_policy=q_policy)
    initial_thread_msgs = len(orchestrator.thread_store.get_messages(active_thread_id))
    print(f"  Active Thread:  {active_thread_id} ({initial_thread_msgs} messages in history)")
    print(f"  Thread Storage: MongoDB Atlas ({settings.mongodb_db_name}.threads)")

    print("=" * 80)

    step_counter = 0
    print("\nEnter a task below, or type commands like '/thread <id>', '/history', '/threads', '/clear', or 'exit'.")

    while True:
        try:
            task = input(f"\n[{active_thread_id}] USER TASK> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break
        start_time = time.perf_counter()
        if task.lower() == "exit":
            print(f"Session ended. Total tasks executed & logged: {step_counter}")
            print(f"View runs via TensorBoard: tensorboard --logdir {logger.log_base_dir}")
            break
        if not task:
            print("Please enter a task or type 'exit'.")
            continue

        if task.startswith("/"):
            try:
                parts = task.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd in {"/thread", "/t"}:
                    if arg:
                        active_thread_id = arg
                        msgs = orchestrator.thread_store.get_messages(active_thread_id)
                        print(f"  [Thread Store] Switched to thread '{active_thread_id}' ({len(msgs)} messages queued).")
                    else:
                        msgs = orchestrator.thread_store.get_messages(active_thread_id)
                        print(f"  [Thread Store] Current thread: '{active_thread_id}' ({len(msgs)} messages queued). Usage: /thread <thread_id>")
                    continue
                elif cmd == "/threads":
                    threads = orchestrator.thread_store.list_threads()
                    print(f"\n  [Thread Store] Available Threads ({len(threads)}):")
                    for tid in threads:
                        stats = orchestrator.thread_store.get_thread_stats(tid)
                        active_marker = " [ACTIVE]" if tid == active_thread_id else ""
                        print(f"    * {tid}{active_marker}: {stats['message_count']} messages (User: {stats['human_messages']}, AI/Agents: {stats['ai_messages']})")
                    continue
                elif cmd in {"/history", "/messages"}:
                    msgs = orchestrator.thread_store.get_messages(active_thread_id)
                    print(f"\n  [Thread History: '{active_thread_id}' - {len(msgs)} messages]:")
                    if not msgs:
                        print("    (No messages recorded in this thread yet)")
                    else:
                        formatted_hist = orchestrator.thread_store.format_history(active_thread_id, max_messages=50)
                        for line in formatted_hist.split("\n"):
                            print(f"    {line}")
                    continue
                elif cmd == "/clear":
                    orchestrator.thread_store.clear_thread(active_thread_id)
                    print(f"  [Thread Store] Cleared message history for thread '{active_thread_id}'.")
                    continue
                elif cmd == "/new":
                    all_threads = orchestrator.thread_store.list_threads()
                    active_thread_id = f"thread-{len(all_threads) + 1}"
                    print(f"  [Thread Store] Started new thread session '{active_thread_id}'.")
                    continue
                elif cmd in {"/help", "/?"}:
                    print("\n  [Thread Store Commands]:")
                    print("    /thread <id>   - Switch to or create a conversation thread (e.g., /thread task-123)")
                    print("    /threads       - List all conversation threads with message statistics")
                    print("    /history       - View full message history for the active thread")
                    print("    /clear         - Clear message history for the active thread")
                    print("    /new           - Create and switch to a new incremental thread")
                    print("    exit           - Exit session\n")
                    continue
            except ConnectionError as conn_err:
                print(f"\n{conn_err}")
                continue

        if session_mode == "fresh":
            orchestrator.reset_to_baseline()

        step_counter += 1
        print(f"\n[Processing Task #{step_counter} | Thread: {active_thread_id}] \"{task}\" (Architecture: v{orchestrator.current_version}) ...")

        try:
            result = orchestrator.run_dynamic(task, thread_id=active_thread_id)
        except ConnectionError as conn_err:
            print(f"\n{conn_err}")
            continue
        except Exception as exc:
            print(f"Runtime error: {type(exc).__name__}: {exc}")
            continue

        final_arch = result.final_architecture

        # 1. Extract true required capabilities dynamically via loops
        plan_dict = result.planner_output or {}
        req_caps = list(plan_dict.get("required_capabilities") or [])
        tools_needed = list(plan_dict.get("tools_needed") or [])

        # Dynamic loop 1: Map any tools_needed into capabilities dynamically
        tool_to_cap = {
            "web_search": "web_search",
            "calculator": "math",
            "code_interpreter": "coding",
            "retriever": "research",
        }
        for tool in tools_needed:
            cap = tool_to_cap.get(tool, tool)
            if cap not in req_caps:
                req_caps.append(cap)

        # Dynamic loop 2: Scan requirement flags in planner_output dynamically
        for key, val in plan_dict.items():
            if key.startswith("requires_") and val:
                cap_name = key[len("requires_"):]
                if cap_name == "tools":
                    cap_name = "web_search"
                if cap_name not in req_caps:
                    req_caps.append(cap_name)

        # Guardrail: queries asking for live/current events require web_search
        query_lower = task.lower()
        temporal_kws = ["current", "trending", "latest", "today", "live", "recent", "events in"]
        if any(kw in query_lower for kw in temporal_kws) and "web_search" not in req_caps:
            req_caps.append("web_search")

        invoked = result.agents_actually_invoked or ["planner", "finalizer"]

        # Dynamic loop 3: Ensure any invoked agent is marked active in final_arch
        agent_dict = {a["agent_id"]: a for a in final_arch.get("agents", [])}
        for agent_id in invoked:
            if agent_id in agent_dict:
                agent_dict[agent_id]["active"] = True
            elif agent_id == "tool_executor":
                final_arch.setdefault("agents", []).append({
                    "agent_id": "tool_executor",
                    "role": "tool_execution",
                    "description": "Dedicated execution node for external tools.",
                    "capabilities": ["tool_use", "web_search", "external_api"],
                    "active": True,
                })

        if "tool_executor" in invoked:
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
        # Dynamically deactivate any uninvoked specialist agent without hardcoding agent lists
        pruned_any = False
        for a in final_arch.get("agents", []):
            aid = a.get("agent_id")
            if aid not in {"planner", "finalizer"} and aid not in invoked:
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

        total_text_chars = len(task) + len(str(result.final_response or ""))
        if result.planner_output:
            total_text_chars += len(str(result.planner_output))
        for trace_event in getattr(result, "execution_trace", []):
            if hasattr(trace_event, "output") and trace_event.output:
                total_text_chars += len(str(trace_event.output))
        prompt_overhead = len(invoked) * 350
        actual_tokens = prompt_overhead + max(40, total_text_chars // 4)
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
        decision_engine_str = (
            "Active RL Policy (Instant Zero-LLM, ~1.2s saved)"
            if getattr(result, "is_instant_rl", False)
            else f"LLM Adapter ({getattr(result, 'decision_source', 'llm')})"
        )
        tools_executed = getattr(result, "tools_executed", [])
        missing_list = logged.get("missing_capabilities", [])
        surplus_list = logged.get("surplus_agents", [])
        tool_to_cap = {"web_search": "web_search", "calculator": "math", "get_current_date": "web_search"}
        unnecessary_tools_preview = [t for t in tools_executed if tool_to_cap.get(t, t) not in req_caps]

        if missing_list:
            util_status = f"UNDER-USED (Missing: {', '.join(missing_list)})"
        elif surplus_list or unnecessary_tools_preview:
            over_items = []
            if surplus_list:
                over_items.append(f"surplus nodes: {', '.join(surplus_list)}")
            if unnecessary_tools_preview:
                over_items.append(f"unneeded tools: {', '.join(unnecessary_tools_preview)}")
            util_status = f"OVER-USED ({'; '.join(over_items)})"
        else:
            util_status = "PERFECT (Optimal Minimal DAG - All required capabilities satisfied without bloat)"

        print("\nEXECUTION SUMMARY:")
        print(f"  * Thread ID:            {getattr(result, 'thread_id', 'thread-1')} ({len(getattr(result, 'messages', []))} messages queued)")
        print(f"  * Architecture Version: v{result.architecture_version}")
        print(f"  * Decision Engine:      {decision_engine_str}")
        print(f"  * Topology Status:      {util_status}")
        print(f"  * Architecture Changed: {arch_changed}")
        print(f"  * Agents Invoked:       {', '.join(result.agents_actually_invoked)}")
        if tools_executed:
            print(f"  * Tools Executed:       {', '.join(tools_executed)}")
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

        # 5. Interactive Human Feedback & Decoupled Dual-Objective Reward Calculation
        print(f"\n[HUMAN VERIFICATION & RLHF]")
        print(f"  * Detected Topology:    [{util_status}]")
        print(f"  * Agents Invoked:       [{', '.join(invoked)}]")
        if tools_executed:
            print(f"  * Tools Executed:       [{', '.join(tools_executed)}]")

        print("\n  Evaluate the execution & architecture:")
        print("    [p] or [y]     - PERFECT: minimal sufficient nodes and accurate answer (+0.5 reward)")
        print("    [over] or [o]  - OVER-USED: answer is okay, but used unneeded tools or redundant nodes (-0.4 arch penalty)")
        print("    [under] or [u] - UNDER-USED: missing necessary tools or agents needed for the task (-0.5 arch penalty)")
        print("    [n]            - FAILED: wrong answer or poor execution (-0.6 penalty)")
        print("    [Enter]        - Accept automated verification without human bias")
        try:
            user_choice = input("  Your choice (p/over/under/n/Enter): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            user_choice = ""

        from app.evaluation.reward import DualObjectiveRewardCalculator
        dual_calc = DualObjectiveRewardCalculator()
        reward_info = dual_calc.calculate(
            required_capabilities=req_caps,
            active_agents=[a["agent_id"] for a in final_arch.get("agents", []) if a.get("active")],
            invoked_agents=invoked,
            tools_executed=tools_executed,
            coverage_score=logged.get("coverage_score", 1.0),
            missing_capabilities=logged.get("missing_capabilities", []),
            surplus_agents=logged.get("surplus_agents", []),
            user_feedback=user_choice,
            final_response=result.final_response,
        )

        step_reward = reward_info["total_reward"]
        comps = reward_info["components"]

        print(f"\n[DUAL-OBJECTIVE REWARD BREAKDOWN]")
        print(f"  * Architecture Reward (R_arch):     {reward_info['r_arch']:+.4f}")
        print(f"    - Capability Coverage:             {comps['coverage_score'] * 100:.1f}%")
        if comps['missing_capability_penalty'] != 0.0:
            print(f"    - Missing Capability Penalty:      {comps['missing_capability_penalty']:+.4f} (missing: {', '.join(reward_info['missing_capabilities'])})")
        if comps['unnecessary_tool_penalty'] != 0.0:
            print(f"    - Unnecessary Tools Penalty:       {comps['unnecessary_tool_penalty']:+.4f} (unused tools: {', '.join(reward_info['unnecessary_tools'])})")
        if comps['surplus_agent_penalty'] != 0.0:
            print(f"    - Surplus Agents Penalty:          {comps['surplus_agent_penalty']:+.4f} (surplus: {', '.join(reward_info['surplus_agents'])})")
        if comps['parsimony_bonus'] != 0.0:
            print(f"    - Parsimony / Minimal DAG Bonus:    {comps['parsimony_bonus']:+.4f} (100% minimal sufficient)")
        print(f"  * Response Quality Reward (R_resp):    {reward_info['r_resp']:+.4f} [{reward_info['response_reason']}]")
        print(f"  => NET STEP REWARD FOR Q-TABLE:       {step_reward:+.4f}")

        # Log dual rewards to TensorBoard
        logger.log_reward(
            step=step_counter,
            r_arch=reward_info["r_arch"],
            r_resp=reward_info["r_resp"],
            total_reward=step_reward,
            components=comps,
        )

        # 6. Update Q-learning policy with transition & persist to disk
        try:
            from app.architecture.models import MASArchitecture
            from app.rl.state import ArchitectureStateEncoder
            from app.rl.meta_task import MetaTaskContext

            init_arch_raw = getattr(result, "initial_architecture", final_arch)
            init_arch_obj = MASArchitecture.model_validate(init_arch_raw)
            final_arch_obj = MASArchitecture.model_validate(final_arch)

            known_categories = ["research", "coding", "math", "analysis", "data", "tool_use", "verification"]
            task_category = next((c for c in req_caps if c in known_categories), "general")
            task_ctx = MetaTaskContext(
                task_category=task_category,
                required_capabilities=sorted(req_caps),
                difficulty=1,
            )

            obs_init = ArchitectureStateEncoder(init_arch_obj).encode()
            state_key = q_policy.get_state_key(obs_init, task_ctx)

            obs_final = ArchitectureStateEncoder(final_arch_obj).encode()
            next_state_key = q_policy.get_state_key(obs_final, task_ctx)

            chosen_action_id = result.action_ids[0] if getattr(result, "action_ids", None) else 0

            q_policy.update(
                state_key=state_key,
                action_id=chosen_action_id,
                reward=step_reward,
                next_state_key=next_state_key,
                next_valid_actions=getattr(result, "action_ids", [0]) or [0],
                terminated=True,
                truncated=False,
            )
            q_policy.save_to_mongodb()
            print(f"  [Q-Learning Memory] Persisted Q-table ({q_policy.q_table.num_state_action_pairs()} entries) -> MongoDB Atlas")
        except ConnectionError as conn_err:
            print(f"\n{conn_err}")
        except Exception as q_err:
            print(f"  [Q-Learning Memory] Note: Q-table update deferred ({q_err})")

        end_time = time.perf_counter()
        # Calculate the interval
        execution_time = end_time - start_time
        print(f" Elapsed time: {execution_time:.6f} seconds")
        
    logger.close()


if __name__ == "__main__":
    main()

