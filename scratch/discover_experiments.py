import json
import os
from pathlib import Path

runs_dir = Path("runs")
benchmark_dir = Path("runs/benchmark")
data_dir = Path("data")

print("=== DISCOVERING ALL EXPERIMENT RUNS ===")

# 1. Benchmark sessions
benchmark_sessions = sorted(list(benchmark_dir.glob("session_*")))
print(f"Total benchmark session directories: {len(benchmark_sessions)}")
for s in benchmark_sessions:
    m = s / "metrics.jsonl"
    tf = list(s.glob("events.out.tfevents.*"))
    if m.exists():
        records = [json.loads(line) for line in m.read_text(encoding="utf-8").splitlines() if line.strip()]
        episodes = set(r.get("experiment_metadata", {}).get("experiment_id", "") for r in records)
        tasks = [r.get("task") for r in records]
        print(f"  Benchmark Session: {s.name}")
        print(f"    Records: {len(records)}, TF event files: {len(tf)}")
        print(f"    Experiment IDs: {episodes}")
        if records:
            r0 = records[0]
            print(f"    First record keys: {list(r0.keys())}")
            print(f"    Evaluation mode: {r0.get('experiment_metadata', {}).get('evaluation_mode')}, is_proxy: {r0.get('is_proxy')}")

# 2. Interactive sessions
interactive_sessions = sorted([d for d in runs_dir.glob("session_*") if d.is_dir()])
print(f"\nTotal interactive session directories: {len(interactive_sessions)}")
for s in interactive_sessions:
    m = s / "metrics.jsonl"
    tf = list(s.glob("events.out.tfevents.*"))
    rec_count = 0
    if m.exists():
        rec_count = len([line for line in m.read_text(encoding="utf-8").splitlines() if line.strip()])
    print(f"  Interactive Session: {s.name} -> records: {rec_count}, TF event files: {len(tf)}")

# 3. Global logs
all_runs_benchmark = benchmark_dir / "all_runs.jsonl"
all_runs_root = runs_dir / "all_runs.jsonl"
print(f"\nall_runs in benchmark: {len([l for l in all_runs_benchmark.read_text(encoding='utf-8').splitlines() if l.strip()]) if all_runs_benchmark.exists() else 0} records")
print(f"all_runs in root: {len([l for l in all_runs_root.read_text(encoding='utf-8').splitlines() if l.strip()]) if all_runs_root.exists() else 0} records")

# 4. Neural baseline results
neural_file = data_dir / "multi_task_baseline_results.json"
if neural_file.exists():
    n_data = json.loads(neural_file.read_text(encoding="utf-8"))
    print(f"\nNeural baseline results file: {neural_file}")
    print(f"  Tasks evaluated: {list(n_data.keys())}")
    for t_id, t_res in n_data.items():
        seeds = t_res.get("seeds", [])
        policies = list(t_res.get("summary_statistics", {}).keys())
        print(f"    Task {t_id}: seeds={seeds}, policies={policies}")

# 5. Q-table memory
q_table = data_dir / "q_table.json"
if q_table.exists():
    q_data = json.loads(q_table.read_text(encoding="utf-8"))
    print(f"\nQ-table memory file: {q_table}")
    print(f"  Num states: {len(q_data)}")

