import json
import numpy as np

with open("runs/benchmark/session_20261101_111152/metrics.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

ep2 = lines[18:36]

invoked_counts = [len(r.get("invoked_agents", [])) for r in ep2]
active_counts = [len(r.get("active_agents", [])) for r in ep2]
print(f"Active counts: {active_counts}")
print(f"Invoked counts: {invoked_counts}")
print(f"Mean active: {np.mean(active_counts):.4f}")
print(f"Mean invoked: {np.mean(invoked_counts):.4f}")
apr_active_pool6 = [(6 - a)/6.0 * 100.0 for a in active_counts]
apr_active_base5 = [(5 - a)/5.0 * 100.0 for a in active_counts]
apr_invoked_pool6 = [(6 - a)/6.0 * 100.0 for a in invoked_counts]
apr_invoked_base5 = [(5 - a)/5.0 * 100.0 for a in invoked_counts]
print(f"APR active vs pool=6: {np.mean(apr_active_pool6):.2f}%")
print(f"APR active vs base=5: {np.mean(apr_active_base5):.2f}%")
print(f"APR invoked vs pool=6: {np.mean(apr_invoked_pool6):.2f}%")
print(f"APR invoked vs base=5: {np.mean(apr_invoked_base5):.2f}%")

