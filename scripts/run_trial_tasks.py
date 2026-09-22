"""Test runner for diverse task types:
1. Coding with critique audit & code preservation
2. Tool execution (live tools: date & calculator)
3. Direct conversational query
Verifies code output preservation, interaction flow diagram, and Q-table persistence.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.config.settings import settings
from app.runtime.orchestrator import AdaptiveRuntimeOrchestrator
from app.rl.q_learning import QLearningPolicy
from app.memory.mongo_client import get_mongo_client
from app.graph.visualizer import format_architecture_display, save_graph_image
from app.evaluation.reward import DualObjectiveRewardCalculator
from app.agents.finalizer import format_final_response
from app.architecture.models import MASArchitecture
from app.rl.state import ArchitectureStateEncoder
from app.rl.meta_task import MetaTaskContext


def run_trials():
    print("=" * 80)
    print("STARTING DIVERSE TASK TRIALS & Q-TABLE UPDATES")
    print("=" * 80)

    # 1. Initialize Q-learning policy
    q_policy = QLearningPolicy(task_aware=True, epsilon=0.1)
    mongo_client = get_mongo_client(required=False)
    if mongo_client is not None:
        q_policy.load_from_mongodb()
    q_table_path = Path("data/q_table.json")
    if q_table_path.exists():
        q_policy.load_q_table(str(q_table_path))

    initial_q_entries = q_policy.q_table.num_state_action_pairs()
    print(f"Initial Q-table state-action entries: {initial_q_entries}")

    orchestrator = AdaptiveRuntimeOrchestrator.from_settings(q_policy=q_policy)
    dual_calc = DualObjectiveRewardCalculator()

    test_tasks = [
        (
            "Task 1 (Coding + Critic)",
            "Write a Python function to evaluate arithmetic expressions containing integers, operators +, -, *, /, and parentheses without using eval.",
            "thread-trial-coding",
        ),
        (
            "Task 2 (Tools: Date + Calculator)",
            "What is today's current date and what is 12345 * 67890?",
            "thread-trial-tools",
        ),
        (
            "Task 3 (Direct Conversational)",
            "Hello! My name is Naveen. What is my name?",
            "thread-trial-chat",
        ),
    ]

    for idx, (label, task_text, thread_id) in enumerate(test_tasks, 1):
        print(f"\n{'=' * 80}")
        print(f"EXECUTING [{label}]: \"{task_text}\"")
        print(f"{'=' * 80}")

        start_time = time.perf_counter()
        result = orchestrator.run_dynamic(task_text, thread_id=thread_id)
        duration = time.perf_counter() - start_time

        final_arch = result.final_architecture
        invoked = result.agents_actually_invoked or ["planner", "finalizer"]
        tools_executed = getattr(result, "tools_executed", [])
        plan_dict = result.planner_output or {}
        req_caps = list(plan_dict.get("required_capabilities") or [])

        # Display interaction flow
        display_str = format_architecture_display(
            final_arch,
            version=result.architecture_version,
            invoked_agents=invoked,
            tools_executed=tools_executed,
        )
        print(display_str)

        # Print Final Response
        final_resp = result.final_response
        formatted_resp = format_final_response(final_resp)
        print("\nFINAL RESPONSE:")
        print("-" * 60)
        print(formatted_resp)
        print("-" * 60)
        print(f"Elapsed time: {duration:.2f} seconds")

        # Specific assertion checks
        if "Coding" in label:
            has_code_block = "```" in formatted_resp or "def " in formatted_resp
            print(f"\n[Validation] Contains actual code implementation: {'YES (PASS)' if has_code_block else 'NO (FAIL)'}")
            assert has_code_block, "Coding task response MUST contain runnable code!"

        # Calculate reward
        reward_info = dual_calc.calculate(
            required_capabilities=req_caps,
            active_agents=[a["agent_id"] for a in final_arch.get("agents", []) if a.get("active")],
            invoked_agents=invoked,
            tools_executed=tools_executed,
            coverage_score=1.0,
            missing_capabilities=[],
            surplus_agents=[],
            user_feedback="p",
            final_response=result.final_response,
        )
        step_reward = reward_info["total_reward"]
        print(f"[Reward] Step Net Reward: {step_reward:+.4f} (R_arch: {reward_info['r_arch']:+.2f}, R_resp: {reward_info['r_resp']:+.2f})")

        # Update Q-table
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

        # Persist Q-table
        try:
            q_policy.save_to_mongodb()
        except Exception:
            pass
        q_policy.save_q_table(str(q_table_path))
        print(f"[Q-Learning] Updated & persisted Q-table. Current entries: {q_policy.q_table.num_state_action_pairs()}")

    final_q_entries = q_policy.q_table.num_state_action_pairs()
    print("\n" + "=" * 80)
    print(f"ALL TRIALS COMPLETED! Q-table grew from {initial_q_entries} -> {final_q_entries} entries.")
    print("=" * 80)


if __name__ == "__main__":
    run_trials()
