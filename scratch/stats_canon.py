import json
import numpy as np
import mpmath
from pathlib import Path

session_path = Path("runs/benchmark/session_20261101_111152/metrics.jsonl")
records = [json.loads(l) for l in session_path.read_text(encoding="utf-8").splitlines() if l.strip()]

ep1 = records[:18]
ep2 = records[18:36]

b_tok = [3500] * 18
a_tok = [r.get("estimated_tokens", 0) for r in ep2]
b_cost = [0.00525] * 18
a_cost = [r.get("estimated_cost_usd", 0.0) for r in ep2]

diff_tok = np.array(b_tok) - np.array(a_tok)
mean_diff = np.mean(diff_tok)
std_diff = np.std(diff_tok, ddof=1)
se_diff = std_diff / np.sqrt(18)
t_stat = mean_diff / se_diff
df = 17
x = df / (df + t_stat**2)
p_val = float(mpmath.betainc(df/2, 0.5, 0, x, regularized=True))

print("=== CANONICAL SESSION (session_20261101_111152) EPISODE 2 ===")
print(f"Total Baseline Tokens: {sum(b_tok)}")
print(f"Total Adapted Tokens:  {sum(a_tok)}")
print(f"Token Reduction:       {((sum(b_tok) - sum(a_tok))/sum(b_tok))*100:.2f}%")
print(f"Mean Baseline Tokens:  {np.mean(b_tok):.1f}")
print(f"Mean Adapted Tokens:   {np.mean(a_tok):.1f} +/- {np.std(a_tok, ddof=1):.1f}")
print(f"Paired t-test:         t = {t_stat:.4f}, df = {df}, p = {p_val:.4e}")

active_agents = [len(r.get("active_agents", [])) for r in ep2]
invoked_agents = [len(r.get("invoked_agents", [])) for r in ep2]
print(f"\nActive Agents:         Mean = {np.mean(active_agents):.2f} +/- {np.std(active_agents, ddof=1):.2f} (Min={min(active_agents)}, Max={max(active_agents)})")
print(f"Invoked Agents:        Mean = {np.mean(invoked_agents):.2f} +/- {np.std(invoked_agents, ddof=1):.2f}")

apr_active_pool6 = [(6 - a)/6.0 * 100.0 for a in active_agents]
apr_active_base5 = [(5 - a)/5.0 * 100.0 for a in active_agents]
apr_invoked_pool6 = [(6 - a)/6.0 * 100.0 for a in invoked_agents]
apr_invoked_base5 = [(5 - a)/5.0 * 100.0 for a in invoked_agents]

print(f"APR (Active vs Pool=6):    {np.mean(apr_active_pool6):.2f}% +/- {np.std(apr_active_pool6, ddof=1):.2f}%")
print(f"APR (Active vs Base=5):    {np.mean(apr_active_base5):.2f}% +/- {np.std(apr_active_base5, ddof=1):.2f}%")
print(f"APR (Invoked vs Pool=6):   {np.mean(apr_invoked_pool6):.2f}% +/- {np.std(apr_invoked_pool6, ddof=1):.2f}%")
print(f"APR (Invoked vs Base=5):   {np.mean(apr_invoked_base5):.2f}% +/- {np.std(apr_invoked_base5, ddof=1):.2f}%")

crit = [r.get("critical_path_length", 3) for r in ep2]
print(f"\nCritical Path Length:  Mean = {np.mean(crit):.2f} +/- {np.std(crit, ddof=1):.2f} hops (Baseline = 5.0 hops)")

cov = [r.get("coverage_score", 1.0) * 100.0 for r in ep2]
print(f"Capability Coverage:   Mean = {np.mean(cov):.2f}% +/- {np.std(cov, ddof=1):.2f}% (All {sum(1 for c in cov if c >= 100.0)}/18 at 100%)")

util = [r.get("net_utility", 0.0) for r in ep2]
print(f"Pareto Net Utility:    Mean = {np.mean(util):.3f} +/- {np.std(util, ddof=1):.3f}")

classes = [r.get("classification", "OPTIMAL") for r in ep2]
print(f"Classifications:       Optimal = {sum(1 for c in classes if c.lower() == 'optimal')}/18")

# Check Episode 1 + Episode 2 totals
tot_b_all = sum(b_tok) * 2
tot_a_all = sum(r.get("estimated_tokens", 0) for r in records)
print(f"\n--- Cumulative 2-Episode (36 Tasks) ---")
print(f"Cumulative Baseline:   {tot_b_all}")
print(f"Cumulative Adapted:    {tot_a_all}")
print(f"Cumulative Savings:    {((tot_b_all - tot_a_all)/tot_b_all)*100:.2f}%")

