import json
import csv
from pathlib import Path
import numpy as np
import scipy.stats as stats

base_dir = Path("research_results")
data_dir = base_dir / "extracted_data"

print("=== AUDITING BENCHMARK TASK RESULTS ===")
bm_csv = data_dir / "benchmark_task_results.csv"
bm_rows = list(csv.DictReader(open(bm_csv, encoding="utf-8")))

ep1_rows = [r for r in bm_rows if r["episode"] == "1"]
ep2_rows = [r for r in bm_rows if r["episode"] == "2"]

print(f"Total rows in benchmark_task_results.csv: {len(bm_rows)}")
print(f"Episode 1 rows: {len(ep1_rows)}, Episode 2 rows: {len(ep2_rows)}")

print("\n--- EPISODE 2 PER-TASK BREAKDOWN ---")
for r in ep2_rows:
    print(f"  {r['task_id']:<18} | Base: {r['baseline_tokens']} | Adapt: {r['adapted_tokens']} | Sav: {r['token_savings_pct']:>6}% | Act: {r['active_agents_count']} | APR: {r['agent_pruning_rate_pct']:>5}% | Cov: {r['capability_coverage_pct']:>5}% | NetUtil: {r['net_utility']:>6} | Class: {r['classification']}")

# Token calculations
b_tok = [int(r["baseline_tokens"]) for r in ep2_rows]
a_tok = [int(r["adapted_tokens"]) for r in ep2_rows]
tot_base = sum(b_tok)
tot_adapt = sum(a_tok)
token_red_pct = ((tot_base - tot_adapt) / tot_base) * 100.0

print("\n--- TOKEN TOTALS (EPISODE 2) ---")
print(f"Total Baseline: {tot_base}")
print(f"Total Adapted:  {tot_adapt}")
print(f"Net Reduction:  {token_red_pct:.2f}%")

# Paired test
diff = np.array(b_tok) - np.array(a_tok)
t_res = stats.ttest_rel(b_tok, a_tok)
print(f"Paired t-test: t = {t_res.statistic:.4f}, df = {len(b_tok) - 1}, p = {t_res.pvalue:.4e}")

# Active agents
agents = [int(r["active_agents_count"]) for r in ep2_rows]
print("\n--- AGENT COUNTS (EPISODE 2) ---")
print(f"Active Agents: Mean = {np.mean(agents):.2f}, Std = {np.std(agents, ddof=1):.2f}, Min = {min(agents)}, Max = {max(agents)}")

# APR methods
# Method 1: relative to 6-agent pool (total possible roster: planner, researcher, coder, critic, tool_executor, finalizer)
apr_pool6 = [(6 - a) / 6.0 * 100.0 for a in agents]
# Method 2: relative to 5-agent baseline (static baseline roster: planner, researcher, coder, critic, finalizer)
apr_base5 = [(5 - a) / 5.0 * 100.0 for a in agents]
# Method 3: runtime invoked agents relative to pool 6
# (We check from benchmark_summary.json)
summary_json = json.loads(open("runs/benchmark/benchmark_summary.json").read())
print(f"APR relative to pool=6 (active agents):    Mean = {np.mean(apr_pool6):.2f}% +/- {np.std(apr_pool6, ddof=1):.2f}%")
print(f"APR relative to baseline=5 (active agents):Mean = {np.mean(apr_base5):.2f}% +/- {np.std(apr_base5, ddof=1):.2f}%")
print(f"APR recorded in benchmark_summary.json:    {summary_json.get('mean_agent_pruning_rate_pct')}%")

# Capability coverage
covs = [float(r["capability_coverage_pct"]) for r in ep2_rows]
print("\n--- CAPABILITY COVERAGE (EPISODE 2) ---")
print(f"Mean Coverage: {np.mean(covs):.2f}% +/- {np.std(covs, ddof=1):.2f}%")
optimal_tasks = [r["task_id"] for r in ep2_rows if float(r["capability_coverage_pct"]) == 100.0]
under_tasks = [r["task_id"] for r in ep2_rows if float(r["capability_coverage_pct"]) < 100.0]
print(f"Optimal Tasks (100% coverage): {len(optimal_tasks)}/18 ({len(optimal_tasks)/18*100:.1f}%) -> {optimal_tasks}")
print(f"Under-engineered Tasks (<100%): {len(under_tasks)}/18 ({len(under_tasks)/18*100:.1f}%) -> {under_tasks}")

# Critical path
lcrit = [int(r["critical_path_length"]) for r in ep2_rows]
print("\n--- CRITICAL PATH (EPISODE 2) ---")
print(f"Mean L_crit: {np.mean(lcrit):.2f} +/- {np.std(lcrit, ddof=1):.2f} hops (Baseline = 5.0)")

# Net utility
net_util = [float(r["net_utility"]) for r in ep2_rows]
print("\n--- PARETO NET UTILITY (EPISODE 2) ---")
print(f"Mean Net Utility: {np.mean(net_util):.3f} +/- {np.std(net_util, ddof=1):.3f}")

# Active RL hits
rl_hits = summary_json.get("active_rl_hits")
rl_rate = summary_json.get("active_rl_hit_rate_pct")
print("\n--- ACTIVE RL SELECTION (EPISODE 2) ---")
print(f"Active RL Policy Hits: {rl_hits}/18 ({rl_rate}%)")
print(f"LLM Adapter Fallbacks: {summary_json.get('adapter_fallback_count')}/18")
print(f"Heuristic Fast Paths:  {summary_json.get('heuristic_bypass_count')}/18")

print("\n=== AUDITING NEURAL ADAPTATION RESULTS ===")
neural_csv = data_dir / "neural_adaptation_results.csv"
neural_rows = list(csv.DictReader(open(neural_csv, encoding="utf-8")))
print(f"Total rows in neural_adaptation_results.csv: {len(neural_rows)}")

tasks = sorted(list(set(r["task_id"] for r in neural_rows)))
for t in tasks:
    print(f"\nTask: {t}")
    for p in ["shared_meta_policy", "fresh_scratch_policy", "deterministic_baseline", "random_policy"]:
        p_recs = [r for r in neural_rows if r["task_id"] == t and r["policy_name"] == p]
        inits = [float(r["initial_performance"]) for r in p_recs]
        finals = [float(r["final_performance"]) for r in p_recs]
        gains = [float(r["adaptation_gain"]) for r in p_recs]
        print(f"  {p:<25}: Init={np.mean(inits):.4f}, Final={np.mean(finals):.4f} +/- {np.std(finals):.4f}, Gain={np.mean(gains):+.4f} +/- {np.std(gains):.4f}")

