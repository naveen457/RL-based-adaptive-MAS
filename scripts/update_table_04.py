"""
Generate Table IV LaTeX table from live empirical static vs dynamic benchmark results.
Updates:
- Research/tables/table_04_per_task_resources.tex
- Research_Paper/tables/table_04_per_task_resources.tex
- research_results/tables/table_04_per_task_resources.tex
"""

import json
from pathlib import Path

# Prefer the combined live static vs dynamic results if available
combined_json_path = Path("research_results/extracted_data/live_static_vs_dynamic_results.json")
single_json_path = Path("research_results/extracted_data/live_llm_benchmark_results.json")

if combined_json_path.exists():
    with open(combined_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    is_combined = True
    tasks = data.get("tasks", [])
elif single_json_path.exists():
    with open(single_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    is_combined = False
    tasks = data.get("tasks", [])
else:
    print("Error: No benchmark results found.")
    exit(1)

CAT_MAP = {
    "greeting": "Greeting",
    "math": "Mathematics",
    "coding": "Coding",
    "research": "Research",
    "web_search": "Web Search",
    "cross_domain": "Cross-Domain"
}

lines = [
    r"\begin{table*}[!t]",
    r"\centering",
    r"\caption{Per-Task Resource Consumption and Efficiency Comparison (Empirical Live NVIDIA NIM Validation)}",
    r"\label{tab:per_task_resources}",
    r"\footnotesize",
    r"\begin{tabular}{llcccccc}",
    r"\toprule",
    r"\textbf{Task ID} & \textbf{Category} & \textbf{Base Tokens} & \textbf{Adapted Tokens} & \textbf{Token Savings} & \textbf{Base Cost (\$) } & \textbf{Adapted Cost (\$) } & \textbf{Selection Engine} \\",
    r"\midrule"
]

total_base_tokens = 0
total_adapted_tokens = 0
total_base_cost = 0.0
total_adapted_cost = 0.0
active_rl_count = 0

for t in tasks:
    task_id = t["task_id"]
    category = CAT_MAP.get(t["category"], t["category"].capitalize())

    if is_combined:
        b_tok = t["static_tokens"]
        a_tok = t["dynamic_tokens"]
        savings_pct = t["token_savings_pct"]
        b_cost = t["static_cost_usd"]
        a_cost = t["dynamic_cost_usd"]
        engine = t["selection_source"]
    else:
        b_tok = t["live_baseline_tokens"]
        a_tok = t["live_adapted_tokens"]
        savings_pct = t["live_token_savings_pct"]
        b_cost = t["live_baseline_cost_usd"]
        a_cost = t["live_adapted_cost_usd"]
        engine = t["selection_source"]

    total_base_tokens += b_tok
    total_adapted_tokens += a_tok
    total_base_cost += b_cost
    total_adapted_cost += a_cost
    if "Active RL" in engine:
        active_rl_count += 1

    sign = "+" if savings_pct >= 0 else ""
    line = f"\\texttt{{{task_id}}} & {category} & {b_tok:,} & {a_tok:,} & ${sign}{savings_pct:.2f}\\%$ & \\${b_cost:.5f} & \\${a_cost:.5f} & {engine} \\\\"
    lines.append(line)

macro_savings = (total_base_tokens - total_adapted_tokens) / total_base_tokens * 100.0 if total_base_tokens else 0.0
hit_rate = (active_rl_count / len(tasks) * 100.0) if tasks else 0.0

sign_tot = "+" if macro_savings >= 0 else ""
lines.append(r"\midrule")
lines.append(
    f"\\textbf{{Total / Summary}} & --- & \\textbf{{{total_base_tokens:,}}} & \\textbf{{{total_adapted_tokens:,}}} & $\\mathbf{{{sign_tot}{macro_savings:.2f}\\%}}$ & \\textbf{{\\${total_base_cost:.5f}}} & \\textbf{{\\${total_adapted_cost:.5f}}} & \\textbf{{{hit_rate:.1f}\\% Active RL Hit Rate}} \\\\"
)
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table*}")

table_content = "\n".join(lines) + "\n"

out_paths = [
    Path("Research/tables/table_04_per_task_resources.tex"),
    Path("Research_Paper/tables/table_04_per_task_resources.tex"),
    Path("research_results/tables/table_04_per_task_resources.tex"),
]

for p in out_paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(table_content)
    print(f"Updated {p}")

print("\nTable IV successfully regenerated with live empirical measurements!")
