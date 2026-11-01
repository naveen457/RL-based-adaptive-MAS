import json

with open("runs/benchmark/session_20261101_111152/metrics.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

print(f"Total lines: {len(lines)}")
ep1 = lines[:18]
ep2 = lines[18:36]

tok1 = sum(r.get("estimated_tokens", 0) for r in ep1)
tok2 = sum(r.get("estimated_tokens", 0) for r in ep2)
print(f"Episode 1 adapted tokens: {tok1}")
print(f"Episode 2 adapted tokens: {tok2}")

print("\n--- Episode 2 Per-Task Details ---")
for i, r in enumerate(ep2):
    act = len(r.get("active_agents", []))
    cov = r.get("coverage_score", 1.0)
    tok = r.get("estimated_tokens", 0)
    task = r.get("task", "")[:35]
    crit = r.get("critical_path_length", 0)
    print(f"Task {i+1:02d}: tok={tok:4d} | active={act} | cov={cov:.2f} | crit={crit} | task={task}")

