import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

import json
import csv
import numpy as np

# Create directory structure
base_dir = Path("research_results")
dirs = [
    base_dir / "extracted_data",
    base_dir / "statistical_analysis",
    base_dir / "literature",
    base_dir / "figures",
    base_dir / "tables",
    base_dir / "paper_material",
]
for d in dirs:
    d.mkdir(parents=True, exist_ok=True)

print("Created directory structure under research_results/")

# ---------------------------------------------------------------------------
# 1. Experiment Inventory (CSV & Markdown)
# ---------------------------------------------------------------------------
inventory_records = []

# Benchmark sessions
for s in sorted(Path("runs/benchmark").glob("session_*")):
    metrics_file = s / "metrics.jsonl"
    tf_files = list(s.glob("events.out.tfevents.*"))
    if metrics_file.exists():
        records = [json.loads(l) for l in metrics_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        exp_ids = sorted(list(set(r.get("experiment_metadata", {}).get("experiment_id", "unspecified") for r in records)))
        modes = sorted(list(set(r.get("experiment_metadata", {}).get("evaluation_mode", "mock") for r in records)))
        is_proxy = any(r.get("is_proxy", True) for r in records)
        inventory_records.append({
            "experiment_id": ", ".join(exp_ids) if exp_ids else s.name,
            "session_id": s.name,
            "experiment_type": "benchmark_multi_task",
            "path": str(s),
            "record_count": len(records),
            "episodes": len(exp_ids) if exp_ids != ["unspecified"] else (len(records) // 18 if len(records) >= 18 else 1),
            "seeds": "42 (default)",
            "evaluation_mode": ", ".join(modes) if modes else "mock",
            "measurement_type": "simulated_proxy" if is_proxy else "live",
            "has_tensorboard_events": len(tf_files) > 0,
            "tf_event_count": len(tf_files),
        })

# Interactive sessions
for s in sorted(Path("runs").glob("session_*")):
    if not s.is_dir():
        continue
    metrics_file = s / "metrics.jsonl"
    tf_files = list(s.glob("events.out.tfevents.*"))
    records = []
    if metrics_file.exists():
        records = [json.loads(l) for l in metrics_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    inventory_records.append({
        "experiment_id": f"interactive_{s.name}",
        "session_id": s.name,
        "experiment_type": "interactive_runtime",
        "path": str(s),
        "record_count": len(records),
        "episodes": 1,
        "seeds": "unspecified",
        "evaluation_mode": "interactive_mock",
        "measurement_type": "simulated_proxy",
        "has_tensorboard_events": len(tf_files) > 0,
        "tf_event_count": len(tf_files),
    })

# Neural multi-task baseline experiment
neural_path = Path("data/multi_task_baseline_results.json")
if neural_path.exists():
    n_data = json.loads(neural_path.read_text(encoding="utf-8"))
    inventory_records.append({
        "experiment_id": "neural_cross_task_adaptation_multiseed",
        "session_id": "step27_multitask_evaluation",
        "experiment_type": "neural_adaptation_4way_baseline",
        "path": str(neural_path),
        "record_count": len(n_data) * 5 * 4,  # 5 tasks * 5 seeds * 4 policies
        "episodes": 2,  # adaptation steps
        "seeds": "42, 101, 202, 303, 404",
        "evaluation_mode": "mock_neural_evaluator",
        "measurement_type": "simulated_proxy",
        "has_tensorboard_events": False,
        "tf_event_count": 0,
    })

# Write experiment_inventory.csv
inv_csv = base_dir / "extracted_data" / "experiment_inventory.csv"
with open(inv_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "experiment_id", "session_id", "experiment_type", "path", "record_count",
        "episodes", "seeds", "evaluation_mode", "measurement_type",
        "has_tensorboard_events", "tf_event_count"
    ])
    writer.writeheader()
    for row in inventory_records:
        writer.writerow(row)
print(f"Wrote {len(inventory_records)} experiment inventory records to {inv_csv}")

# Write experiment_inventory.md
inv_md = base_dir / "experiment_inventory.md"
with open(inv_md, "w", encoding="utf-8") as f:
    f.write("# Experiment Inventory & Dataset Catalog\n\n")
    f.write(f"Total discovered experiments/sessions: **{len(inventory_records)}**\n\n")
    f.write("### Discovered Experiment Runs\n\n")
    f.write("| Experiment ID | Session ID | Type | Records | Episodes | Seeds | Mode | TF Events |\n")
    f.write("|---|---|---|:---:|:---:|---|---|:---:|\n")
    for r in inventory_records:
        f.write(f"| `{r['experiment_id']}` | `{r['session_id']}` | {r['experiment_type']} | {r['record_count']} | {r['episodes']} | {r['seeds']} | {r['measurement_type']} | {'Yes' if r['has_tensorboard_events'] else 'No'} |\n")
print(f"Wrote inventory markdown to {inv_md}")

# ---------------------------------------------------------------------------
# 2. Neural Adaptation Results (CSV)
# ---------------------------------------------------------------------------
neural_csv = base_dir / "extracted_data" / "neural_adaptation_results.csv"
neural_rows = []

task_domain_map = {
    "task-research-002": ("Research", "research, synthesis"),
    "task-research-001": ("Research", "research"),
    "task-summarization-001": ("NLP", "synthesis"),
    "task-analysis-001": ("Analysis", "analysis, planning"),
    "task-coding-001": ("Coding", "coding, verification"),
}

if neural_path.exists():
    n_data = json.loads(neural_path.read_text(encoding="utf-8"))
    for task_id, t_dict in n_data.items():
        domain, caps = task_domain_map.get(task_id, ("General", "general"))
        seeds = t_dict.get("seeds", [])
        seed_recs = t_dict.get("seed_records", {})
        for policy_name, recs in seed_recs.items():
            for idx, rec in enumerate(recs):
                seed_val = seeds[idx] if idx < len(seeds) else idx
                init_perf = rec.get("initial_performance", 0.0)
                final_perf = rec.get("final_performance", 0.0)
                gain = rec.get("adaptation_gain", 0.0)
                curve = rec.get("adaptation_curve", [init_perf, final_perf])
                neural_rows.append({
                    "task_id": task_id,
                    "domain": domain,
                    "required_capabilities": caps,
                    "policy_name": policy_name,
                    "seed": seed_val,
                    "adaptation_steps": rec.get("adaptation_steps", 2),
                    "initial_performance": init_perf,
                    "final_performance": final_perf,
                    "adaptation_gain": gain,
                    "step_0": curve[0] if len(curve) > 0 else init_perf,
                    "step_1": curve[1] if len(curve) > 1 else (curve[-1] if curve else final_perf),
                    "step_2": curve[2] if len(curve) > 2 else (curve[-1] if curve else final_perf),
                    "evaluation_mode": "mock_neural_evaluator",
                    "measurement_type": "simulated_proxy",
                    "is_proxy": True,
                    "source_file": "data/multi_task_baseline_results.json"
                })

with open(neural_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "task_id", "domain", "required_capabilities", "policy_name", "seed",
        "adaptation_steps", "initial_performance", "final_performance", "adaptation_gain",
        "step_0", "step_1", "step_2", "evaluation_mode", "measurement_type", "is_proxy", "source_file"
    ])
    writer.writeheader()
    for row in neural_rows:
        writer.writerow(row)
print(f"Wrote {len(neural_rows)} neural adaptation records to {neural_csv}")

# ---------------------------------------------------------------------------
# 3. Benchmark Task Results & Sub-metrics (CSV & JSON)
# ---------------------------------------------------------------------------
# We extract from the newest fully-reconciled benchmark session: session_20261101_115443
bm_session = Path("runs/benchmark/session_20261101_115443")
bm_jsonl = bm_session / "metrics.jsonl"
bm_records = [json.loads(l) for l in bm_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]

# Load benchmark task categories and IDs from app/benchmark.py
from app.benchmark import BENCHMARK_TASKS
task_meta = {t.prompt: t for t in BENCHMARK_TASKS}

task_results_rows = []
arch_rows = []
rl_rows = []
graph_rows = []
resource_rows = []

for idx, r in enumerate(bm_records):
    task_prompt = r.get("task", "")
    bt = task_meta.get(task_prompt)
    task_id = bt.task_id if bt else f"task_{idx % 18 + 1:03d}"
    task_name = bt.task_name if bt else task_prompt[:25]
    category = bt.category if bt else "general"
    
    ep = (idx // 18) + 1
    step_in_ep = (idx % 18) + 1
    
    baseline_tokens = 3500
    baseline_cost = round((baseline_tokens / 1000.0) * 0.0015, 5)
    adapted_tokens = r.get("estimated_tokens", 0)
    adapted_cost = r.get("estimated_cost_usd", 0.0)
    token_savings_pct = round(((baseline_tokens - adapted_tokens) / baseline_tokens) * 100.0, 2)
    cost_savings_pct = round(((baseline_cost - adapted_cost) / baseline_cost) * 100.0, 2)
    
    # Engine determination
    # Note: In session_20261101_115443, we have 18 tasks in Ep 1 and 18 in Ep 2.
    # Active agents count:
    active_agents = r.get("active_agents", [])
    active_count = len(active_agents)
    apr_pct = round(((6 - active_count) / 6.0) * 100.0, 1)
    
    # Action metrics
    act_m = r.get("action_metrics", {})
    arch_m = r.get("architecture_metrics", {})
    graph_m = r.get("graph_metrics", {})
    
    # 1. Main task results row
    task_results_rows.append({
        "episode": ep,
        "step": step_in_ep,
        "global_step": r.get("step", idx + 1),
        "task_id": task_id,
        "task_name": task_name,
        "category": category,
        "prompt": task_prompt,
        "baseline_tokens": baseline_tokens,
        "adapted_tokens": adapted_tokens,
        "token_savings_pct": token_savings_pct,
        "baseline_cost_usd": baseline_cost,
        "adapted_cost_usd": adapted_cost,
        "cost_savings_pct": cost_savings_pct,
        "active_agents_count": active_count,
        "agent_pruning_rate_pct": apr_pct,
        "critical_path_length": r.get("critical_path_length", 3),
        "capability_coverage_pct": round(r.get("coverage_score", 1.0) * 100.0, 1),
        "net_utility": r.get("net_utility", 0.0),
        "overall_score": r.get("overall_score", 0.0),
        "structural_efficiency": r.get("structural_efficiency", 0.0),
        "classification": r.get("classification", "OPTIMAL"),
        "is_proxy": True,
        "source_session": bm_session.name,
    })
    
    # 2. Architecture metrics row
    arch_rows.append({
        "episode": ep,
        "task_id": task_id,
        "category": category,
        "agent_count": arch_m.get("agent_count", 6),
        "active_agent_count": arch_m.get("active_agent_count", active_count),
        "inactive_agent_count": arch_m.get("inactive_agent_count", 6 - active_count),
        "edge_count": arch_m.get("edge_count", 0),
        "graph_density": arch_m.get("graph_density", 0.0),
        "critical_path_length": arch_m.get("critical_path_length", r.get("critical_path_length", 3)),
        "capability_coverage": arch_m.get("capability_coverage", r.get("coverage_score", 1.0)),
        "net_utility": arch_m.get("net_utility", r.get("net_utility", 0.0)),
        "overall_score": arch_m.get("overall_score", r.get("overall_score", 0.0)),
        "structural_efficiency": arch_m.get("structural_efficiency", r.get("structural_efficiency", 0.0)),
        "surplus_agents_count": arch_m.get("surplus_agents_count", len(r.get("surplus_agents", []))),
        "required_capability_count": arch_m.get("required_capability_count", len(r.get("required_capabilities", []))),
        "satisfied_capability_count": arch_m.get("satisfied_capability_count", 0),
    })
    
    # 3. RL action metrics row
    rl_rows.append({
        "episode": ep,
        "task_id": task_id,
        "category": category,
        "proposed": act_m.get("proposed", len(r.get("accepted_actions", []))),
        "accepted": act_m.get("accepted", len(r.get("accepted_actions", []))),
        "rejected": act_m.get("rejected", len(r.get("rejected_actions", []))),
        "acceptance_rate": act_m.get("acceptance_rate", 1.0),
        "invalid_action_rate": act_m.get("invalid_action_rate", 0.0),
        "activate_agent_count": act_m.get("action_type_counts", {}).get("activate_agent", 0),
        "deactivate_agent_count": act_m.get("action_type_counts", {}).get("deactivate_agent", 0),
        "add_edge_count": act_m.get("action_type_counts", {}).get("add_edge", 0),
        "remove_edge_count": act_m.get("action_type_counts", {}).get("remove_edge", 0),
        "change_role_count": act_m.get("action_type_counts", {}).get("change_role", 0),
    })
    
    # 4. Graph execution metrics row
    graph_rows.append({
        "episode": ep,
        "task_id": task_id,
        "category": category,
        "build_time_ms": graph_m.get("build_time_ms", 0.0),
        "compilation_time_ms": graph_m.get("compilation_time_ms", 0.0),
        "execution_time_ms": graph_m.get("execution_time_ms", 0.0),
        "node_count": graph_m.get("node_count", 0),
        "edge_count": graph_m.get("edge_count", 0),
        "recovery_count": graph_m.get("recovery_count", 0),
        "conditional_route_count": graph_m.get("conditional_route_count", 0),
        "build_success_rate": graph_m.get("build_success_rate", 1.0),
        "compilation_success_rate": graph_m.get("compilation_success_rate", 1.0),
    })
    
    # 5. Resource metrics row
    resource_rows.append({
        "episode": ep,
        "task_id": task_id,
        "category": category,
        "baseline_tokens": baseline_tokens,
        "adapted_estimated_tokens": adapted_tokens,
        "token_reduction_pct": token_savings_pct,
        "baseline_cost_usd": baseline_cost,
        "adapted_estimated_cost_usd": adapted_cost,
        "cost_reduction_pct": cost_savings_pct,
        "agent_invocations": len(r.get("invoked_agents", [])),
        "measurement_type": "simulated_dag_proxy",
        "is_proxy": True,
    })

# Write CSV files
def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows to {path}")

write_csv(base_dir / "extracted_data" / "benchmark_task_results.csv", task_results_rows)
write_csv(base_dir / "extracted_data" / "architecture_metrics.csv", arch_rows)
write_csv(base_dir / "extracted_data" / "rl_action_metrics.csv", rl_rows)
write_csv(base_dir / "extracted_data" / "graph_execution_metrics.csv", graph_rows)
write_csv(base_dir / "extracted_data" / "resource_metrics.csv", resource_rows)

# Write aggregated_metrics.json
summary_json = Path("runs/benchmark/benchmark_summary.json")
agg_data = json.loads(summary_json.read_text(encoding="utf-8")) if summary_json.exists() else {}
with open(base_dir / "extracted_data" / "aggregated_metrics.json", "w", encoding="utf-8") as f:
    json.dump(agg_data, f, indent=2)
print("Saved aggregated_metrics.json")
