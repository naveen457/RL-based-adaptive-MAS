import os
import sys
from pathlib import Path
import json
import csv

# Ensure matplotlib uses a writable cache directory
os.environ["MPLCONFIGDIR"] = str(Path("scratch/mpl_cache").resolve())
Path("scratch/mpl_cache").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Apply clean academic publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})

base_dir = Path("research_results")
fig_dir = base_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# Load data
neural_csv = base_dir / "extracted_data" / "neural_adaptation_results.csv"
neural_rows = []
with open(neural_csv, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        neural_rows.append(row)

bm_csv = base_dir / "extracted_data" / "benchmark_task_results.csv"
bm_rows = []
with open(bm_csv, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        bm_rows.append(row)

graph_csv = base_dir / "extracted_data" / "graph_execution_metrics.csv"
graph_rows = []
with open(graph_csv, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        graph_rows.append(row)

rl_csv = base_dir / "extracted_data" / "rl_action_metrics.csv"
rl_rows = []
with open(rl_csv, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rl_rows.append(row)

ep2_bm = [r for r in bm_rows if int(r["episode"]) == 2]
ep2_graph = [r for r in graph_rows if int(r["episode"]) == 2]
ep2_rl = [r for r in rl_rows if int(r["episode"]) == 2]

def save_fig(fig, name):
    png_path = fig_dir / f"{name}.png"
    pdf_path = fig_dir / f"{name}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {name}.png and {name}.pdf")

# ---------------------------------------------------------------------------
# FIGURE 1: Neural Cross-Task Adaptation Trajectory
# ---------------------------------------------------------------------------
print("Generating Figure 1: Neural Adaptation Curve...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# Panel A: task-research-002 (Held-out Research)
ax1 = axes[0]
r002 = [r for r in neural_rows if r["task_id"] == "task-research-002"]
steps = [0, 1, 2]
palette = {
    "shared_meta_policy": ("#1f77b4", "o-", "Shared Meta Policy (Pretrained)"),
    "fresh_scratch_policy": ("#ff7f0e", "s--", "Fresh Scratch Policy"),
    "deterministic_baseline": ("#2ca02c", "^:", "Deterministic Baseline"),
    "random_policy": ("#7f7f7f", "x-.", "Random Uniform Policy")
}

for pol, (color, style, label) in palette.items():
    pol_rows = [r for r in r002 if r["policy_name"] == pol]
    curves = np.array([[float(r["step_0"]), float(r["step_1"]), float(r["step_2"])] for r in pol_rows])
    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0)
    ax1.plot(steps, mean_curve, style, color=color, label=label, linewidth=2, markersize=7)
    ax1.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)

ax1.set_title("(a) Research Task (`task-research-002`)\nStructural Role Constraint ($Role \\notin \\mathcal{A}$)")
ax1.set_xlabel("Fine-Tuning Adaptation Steps")
ax1.set_ylabel("Task Compatibility Score ($Jaccard \\in [0, 1]$)")
ax1.set_xticks(steps)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(loc="lower right", frameon=True, framealpha=0.9)

# Panel B: task-coding-001 (Coding Task - Destructive Exploration)
ax2 = axes[1]
c001 = [r for r in neural_rows if r["task_id"] == "task-coding-001"]
for pol, (color, style, label) in palette.items():
    pol_rows = [r for r in c001 if r["policy_name"] == pol]
    curves = np.array([[float(r["step_0"]), float(r["step_1"]), float(r["step_2"])] for r in pol_rows])
    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0)
    ax2.plot(steps, mean_curve, style, color=color, label=label, linewidth=2, markersize=7)
    ax2.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)

ax2.set_title("(b) Coding Task (`task-coding-001`)\nUnconstrained Exploration Degradation")
ax2.set_xlabel("Fine-Tuning Adaptation Steps")
ax2.set_xticks(steps)
ax2.legend(loc="lower left", frameon=True, framealpha=0.9)

fig.suptitle("Cross-Task Neural Policy Adaptation Trajectories (Mean $\\pm$ 1 Std across 5 Seeds)", y=1.02)
save_fig(fig, "figure_01_adaptation_curve")

# ---------------------------------------------------------------------------
# FIGURE 2: Held-Out Multi-Task Comparison (Grouped Bar Chart)
# ---------------------------------------------------------------------------
print("Generating Figure 2: Held-Out Task Comparison...")
fig, ax = plt.subplots(figsize=(11, 5.5))
tasks = ["task-research-002", "task-research-001", "task-summarization-001", "task-analysis-001", "task-coding-001"]
task_labels = ["Research-002\n(Res, Syn)", "Research-001\n(Research)", "Summarize-001\n(Synthesis)", "Analysis-001\n(Analysis, Plan)", "Coding-001\n(Code, Verify)"]
x = np.arange(len(tasks))
width = 0.20

pol_order = ["shared_meta_policy", "fresh_scratch_policy", "deterministic_baseline", "random_policy"]
pol_names = ["Shared Meta Policy", "Fresh Scratch Policy", "Deterministic Baseline", "Random Policy"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"]

for i, (p, p_label, c) in enumerate(zip(pol_order, pol_names, colors)):
    means = []
    stds = []
    for t in tasks:
        p_rows = [r for r in neural_rows if r["task_id"] == t and r["policy_name"] == p]
        finals = [float(r["final_performance"]) for r in p_rows]
        means.append(np.mean(finals))
        stds.append(np.std(finals))
    rects = ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, label=p_label, color=c, capsize=4, alpha=0.88, edgecolor="black", linewidth=0.8)

ax.set_ylabel("Final Task Performance Score ($Jaccard \\in [0, 1]$)")
ax.set_title("Held-Out Generalization Across 5 Tasks and 5 Random Seeds (Mean $\\pm$ Std)")
ax.set_xticks(x)
ax.set_xticklabels(task_labels)
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right", frameon=True, framealpha=0.9)
save_fig(fig, "figure_02_heldout_comparison")

# ---------------------------------------------------------------------------
# FIGURE 3: Capability Coverage vs Active Agent Count
# ---------------------------------------------------------------------------
print("Generating Figure 3: Capability vs Active Agents...")
fig, ax = plt.subplots(figsize=(8, 5.5))

# Plot baseline (5 agents, 100% coverage)
ax.scatter([5.0], [100.0], color="#d62728", s=250, marker="D", edgecolors="black", linewidth=1.5, zorder=5, label="Static Full-Team Baseline (5 Agents)")

# Plot adapted tasks from Episode 2
agents = [float(r["active_agents_count"]) for r in ep2_bm]
covs = [float(r["capability_coverage_pct"]) for r in ep2_bm]

# Add slight jitter for visualization
np.random.seed(42)
jitter_x = [a + np.random.uniform(-0.12, 0.12) for a in agents]
jitter_y = [c + np.random.uniform(-1.0, 1.0) if c == 100.0 else c for c in covs]

scatter = ax.scatter(jitter_x, jitter_y, c=agents, cmap="viridis", s=130, alpha=0.85, edgecolors="black", linewidth=0.8, zorder=4, label="Adapted Topologies (AMAS)")

# Annotate Pareto frontier
ax.axhline(100.0, color="gray", linestyle=":", alpha=0.6)
ax.annotate("Pareto-Optimal Frontier\n(100% Coverage with 2-4 Agents)", xy=(2.0, 100.0), xytext=(2.3, 85.0),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
            fontsize=10, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc", ec="gray", lw=0.8))

ax.annotate("Under-Engineered Region\n(Cross-Domain Tasks)", xy=(3.0, 33.3), xytext=(3.4, 45.0),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#ffdddd", ec="gray", lw=0.8))

ax.set_xlabel("Number of Active Agents in Architecture")
ax.set_ylabel("Capability Coverage Rate (%)")
ax.set_title("Architecture Parsimony vs. Capability Coverage (Episode 2)")
ax.set_xlim(1.5, 5.5)
ax.set_ylim(20, 105)
ax.set_xticks([2, 3, 4, 5])
ax.legend(loc="lower right", frameon=True, framealpha=0.9)
save_fig(fig, "figure_03_capability_vs_agents")

# ---------------------------------------------------------------------------
# FIGURE 4: Token Efficiency Comparison (Grouped Bar Chart across 18 Tasks)
# ---------------------------------------------------------------------------
print("Generating Figure 4: Token Efficiency Comparison...")
fig, ax = plt.subplots(figsize=(14, 6))

task_ids = [r["task_id"] for r in ep2_bm]
b_tok = [int(r["baseline_tokens"]) for r in ep2_bm]
a_tok = [int(r["adapted_tokens"]) for r in ep2_bm]
sav_pct = [float(r["token_savings_pct"]) for r in ep2_bm]

x = np.arange(len(task_ids))
width = 0.38

rects1 = ax.bar(x - width/2, b_tok, width, label="Static Baseline (3,500 tok)", color="#9467bd", alpha=0.7, edgecolor="black", linewidth=0.8)

# Color adapted bars: green if savings > 0, red if savings < 0 (expansion)
adapt_colors = ["#2ca02c" if s >= 0 else "#d62728" for s in sav_pct]
rects2 = ax.bar(x + width/2, a_tok, width, label="Adapted Architecture (AMAS)", color=adapt_colors, alpha=0.85, edgecolor="black", linewidth=0.8)

# Annotate savings % on top of bars
for i, s in enumerate(sav_pct):
    y_pos = max(b_tok[i], a_tok[i]) + 80
    ax.text(x[i], y_pos, f"{s:+.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            color="#1b5e20" if s >= 0 else "#b71c1c")

ax.set_ylabel("Estimated Tokens ($L_{crit}$ DAG Proxy)")
ax.set_title("Per-Task Token Consumption: Static Baseline vs. Adaptive AMAS (18 Benchmark Tasks)")
ax.set_xticks(x)
ax.set_xticklabels(task_ids, rotation=45, ha="right", fontsize=9)
ax.set_ylim(0, 4800)
ax.axhline(3500, color="purple", linestyle="--", alpha=0.5, label="Baseline Ceiling")
ax.legend(loc="upper left", frameon=True, framealpha=0.9)
save_fig(fig, "figure_04_token_comparison")

# ---------------------------------------------------------------------------
# FIGURE 5: Critical Path Length Compression across Categories
# ---------------------------------------------------------------------------
print("Generating Figure 5: Critical Path Length...")
fig, ax = plt.subplots(figsize=(9, 5))

cats = sorted(list(set(r["category"] for r in ep2_bm)))
cat_labels = [c.capitalize() for c in cats]

adapt_lcrit_by_cat = []
for c in cats:
    vals = [int(r["critical_path_length"]) for r in ep2_bm if r["category"] == c]
    adapt_lcrit_by_cat.append(vals)

x = np.arange(len(cats))
width = 0.35

baseline_lcrit = [5] * len(cats)
mean_adapt_lcrit = [np.mean(vals) for vals in adapt_lcrit_by_cat]

rects1 = ax.bar(x - width/2, baseline_lcrit, width, label="Static Baseline Critical Path (5.0 Hops)", color="#d62728", alpha=0.6, edgecolor="black")
rects2 = ax.bar(x + width/2, mean_adapt_lcrit, width, label="Adapted Critical Path (AMAS)", color="#1f77b4", alpha=0.85, edgecolor="black")

# Annotate mean values
for i in range(len(cats)):
    ax.text(x[i] + width/2, mean_adapt_lcrit[i] + 0.1, f"{mean_adapt_lcrit[i]:.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

ax.set_ylabel("Critical Path Length ($L_{crit}$ Sequential Hops)")
ax.set_title("Critical Path Compression Across Functional Domains (GEMMAS/CARD Latency Proxy)")
ax.set_xticks(x)
ax.set_xticklabels(cat_labels)
ax.set_ylim(0, 6.0)
ax.legend(loc="upper right", frameon=True, framealpha=0.9)
save_fig(fig, "figure_05_critical_path")

# ---------------------------------------------------------------------------
# FIGURE 6: Architecture Structural Complexity vs. Net Utility
# ---------------------------------------------------------------------------
print("Generating Figure 6: Architecture Utility vs Complexity...")
fig, ax = plt.subplots(figsize=(8.5, 5.5))

# Baseline point: 5 agents + 4 edges = complexity 9, net utility ~ 0.50
ax.scatter([9.0], [0.500], color="#d62728", s=220, marker="D", edgecolors="black", linewidth=1.5, label="Static Baseline (5 Agents, 4 Edges)")

# Adapted points
complexities = [float(r["active_agents_count"]) + float(r["critical_path_length"]) for r in ep2_bm]
utilities = [float(r["net_utility"]) for r in ep2_bm]

scatter = ax.scatter(complexities, utilities, c=utilities, cmap="coolwarm", s=130, alpha=0.88, edgecolors="black", linewidth=0.8, label="Adapted Topologies (AMAS)")

# Trend line
z = np.polyfit(complexities, utilities, 1)
p = np.poly1d(z)
x_line = np.linspace(min(complexities) - 0.5, max(complexities) + 0.5, 50)
ax.plot(x_line, p(x_line), "k--", alpha=0.5, label=f"Trendline (slope: {z[0]:+.3f})")

ax.set_xlabel("Structural Complexity ($|V_{active}| + L_{crit}$)")
ax.set_ylabel("Pareto Net Utility ($U_{pareto} \\in [0, 1]$)")
ax.set_title("Architecture Structural Complexity vs. Multi-Objective Net Utility")
ax.set_xlim(3.5, 10.0)
ax.set_ylim(0.0, 1.05)
ax.legend(loc="lower left", frameon=True, framealpha=0.9)
save_fig(fig, "figure_06_architecture_utility")

# ---------------------------------------------------------------------------
# FIGURE 7: Dynamic Graph Latency Decomposition
# ---------------------------------------------------------------------------
print("Generating Figure 7: Dynamic Graph Overhead...")
fig, ax = plt.subplots(figsize=(10, 5.5))

cats = sorted(list(set(r["category"] for r in ep2_bm)))
cat_labels = [c.capitalize() for c in cats]

t_build_cat = []
t_comp_cat = []
t_exec_cat = []

for c in cats:
    indices = [i for i, r in enumerate(ep2_bm) if r["category"] == c]
    t_build_cat.append(np.mean([float(ep2_graph[i]["build_time_ms"]) for i in indices]))
    t_comp_cat.append(np.mean([float(ep2_graph[i]["compilation_time_ms"]) for i in indices]))
    t_exec_cat.append(np.mean([float(ep2_graph[i]["execution_time_ms"]) for i in indices]))

x = np.arange(len(cats))
width = 0.55

# Build stacked bars for architecture overhead (build + compile)
p1 = ax.bar(x, t_build_cat, width, label="Graph Construction ($T_{build}$)", color="#1f77b4", alpha=0.85, edgecolor="black")
p2 = ax.bar(x, t_comp_cat, width, bottom=t_build_cat, label="LangGraph Compilation ($T_{comp}$)", color="#ff7f0e", alpha=0.85, edgecolor="black")

ax.set_ylabel("Latency (Milliseconds)")
ax.set_title("Dynamic LangGraph Compilation & Adaptation Overhead (Sub-Millisecond Pipeline)")
ax.set_xticks(x)
ax.set_xticklabels(cat_labels)
ax.legend(loc="upper right", frameon=True, framealpha=0.9)

# Add text callout noting workflow execution scale
total_arch = [b + c for b, c in zip(t_build_cat, t_comp_cat)]
for i, tot in enumerate(total_arch):
    ax.text(x[i], tot + 0.005, f"{tot:.2f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

save_fig(fig, "figure_07_graph_overhead")

# ---------------------------------------------------------------------------
# FIGURE 8: RL Adaptation Behavior & Decision Decomposition
# ---------------------------------------------------------------------------
print("Generating Figure 8: RL Decision Decomposition...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: Decision Engine Breakdown (Episode 1 vs Episode 2)
ep1_bm_rows = [r for r in bm_rows if int(r["episode"]) == 1]
# Calculate hit rates
# Ep 1: 14 RL hits, 4 LLM adapter (from benchmark stdout)
# Ep 2: 13 RL hits, 5 LLM adapter
labels_dec = ["Episode 1", "Episode 2"]
rl_hits = [14, 13]
bypass = [0, 0]
adapters = [4, 5]

x_dec = np.arange(len(labels_dec))
w_dec = 0.5

ax1.bar(x_dec, rl_hits, w_dec, label="Active RL Policy Hit (0 tok, <1ms)", color="#2ca02c", alpha=0.85, edgecolor="black")
ax1.bar(x_dec, bypass, w_dec, bottom=rl_hits, label="Heuristic Fast Path (Trivial Bypass)", color="#17becf", alpha=0.85, edgecolor="black")
ax1.bar(x_dec, adapters, w_dec, bottom=[r + b for r, b in zip(rl_hits, bypass)], label="LLM Adapter Fallback (~450 tok, 850ms)", color="#d62728", alpha=0.85, edgecolor="black")

for i in range(len(labels_dec)):
    hit_pct = (rl_hits[i] / 18.0) * 100.0
    ax1.text(x_dec[i], rl_hits[i] / 2, f"{hit_pct:.1f}%\n({rl_hits[i]}/18)", ha="center", va="center", color="white", fontweight="bold", fontsize=10)

ax1.set_ylabel("Number of Task Executions")
ax1.set_title("(a) Adaptation Decision Engine Breakdown\n(Policy Convergence across Episodes)")
ax1.set_xticks(x_dec)
ax1.set_xticklabels(labels_dec)
ax1.set_ylim(0, 22)
ax1.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8.5)

# Panel B: Distribution of Accepted Mutation Actions
act_types = ["activate_agent", "deactivate_agent", "add_edge", "remove_edge", "change_role"]
act_labels = ["Activate\nAgent", "Deactivate\nAgent", "Add\nEdge", "Remove\nEdge", "Change\nRole"]
act_counts = [0, 0, 0, 0, 0]

for r in ep2_rl:
    act_counts[0] += int(r["activate_agent_count"])
    act_counts[1] += int(r["deactivate_agent_count"])
    act_counts[2] += int(r["add_edge_count"])
    act_counts[3] += int(r["remove_edge_count"])
    act_counts[4] += int(r["change_role_count"])

ax2.bar(act_labels, act_counts, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"], alpha=0.85, edgecolor="black")
for i, count in enumerate(act_counts):
    ax2.text(i, count + 0.3, str(count), ha="center", va="bottom", fontweight="bold", fontsize=10)

ax2.set_ylabel("Total Actions Applied")
ax2.set_title("(b) Accepted Graph Mutation Operations\n(Topological Refinements in Ep 2)")
ax2.set_ylim(0, max(act_counts) + 3 if max(act_counts) > 0 else 5)

fig.suptitle("Reinforcement Learning Adaptation Behavior & Mutation Space", y=1.02)
save_fig(fig, "figure_08_rl_actions")

print("All 8 publication-grade figures successfully generated!")

