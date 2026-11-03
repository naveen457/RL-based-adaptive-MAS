"""
Refined, publication-grade Figure 1:
System Architecture Schematic with zero collisions, spacious cards, and clean visual hierarchy.
"""

import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path("scratch/mpl_cache").resolve())
Path("scratch/mpl_cache").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig = plt.figure(figsize=(13.2, 7.8))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# ===========================================================================
# 1. TOP PIPELINE: RUNTIME EXECUTION PATHWAY (CONTRIBUTION C2)
# ===========================================================================

# STAGE 1: Task Context
s1_box = patches.FancyBboxPatch(
    (2.5, 52), 17.5, 41, boxstyle="round,pad=0.8",
    ec="#1d4ed8", fc="#eff6ff", lw=1.6
)
ax.add_patch(s1_box)

s1_hdr = patches.FancyBboxPatch(
    (2.5, 85.5), 17.5, 7.5, boxstyle="round,pad=0.5",
    ec="#1d4ed8", fc="#dbeafe", lw=1.2
)
ax.add_patch(s1_hdr)
ax.text(11.25, 89.2, "Stage 1: Context", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1e3a8a")

ax.text(11.25, 78.5, "User Request\n& Task Profiler", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#0f172a")
ax.text(11.25, 68.0, "Context Vector $c_t$:", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#1e40af")
ax.text(11.25, 63.5, r"$\langle \kappa_t, d_t, \mathcal{C}_t, \mathbf{f}_t \rangle$", ha="center", va="center", fontsize=9.0, color="#1e3a8a")
ax.text(11.25, 56.5, "Complexity, Domain,\nRequired Capabilities", ha="center", va="center", fontsize=7.8, color="#475569")

# STAGE 2: Runtime Optimization Container (Contribution C2)
c2_box = patches.FancyBboxPatch(
    (24.0, 50), 51.0, 43, boxstyle="round,pad=0.8",
    ec="#15803d", fc="#f0fdf4", lw=2.0
)
ax.add_patch(c2_box)

c2_hdr = patches.FancyBboxPatch(
    (24.0, 86.5), 51.0, 6.5, boxstyle="round,pad=0.5",
    ec="#15803d", fc="#dcfce7", lw=1.3
)
ax.add_patch(c2_hdr)
ax.text(49.5, 89.7, "Pathway 2B (Contribution C2): Live Runtime Q-Learning Pipeline", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#14532d")

# Inside C2: Card 1 (Tabular Q-Selector) - Width 14.0
q_box = patches.FancyBboxPatch(
    (25.8, 54), 14.0, 29, boxstyle="round,pad=0.6",
    ec="#16a34a", fc="#ffffff", lw=1.2
)
ax.add_patch(q_box)
ax.text(32.8, 79.5, "Tabular\nQ-Selector", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#166534")
ax.plot([27.0, 38.6], [74.5, 74.5], color="#bbf7d0", lw=1.0)
ax.text(32.8, 70.0, "State $s_t$\n(Topology, Roles)", ha="center", va="center", fontsize=7.8, color="#1e293b")
ax.text(32.8, 62.5, r"$\epsilon$-greedy Selection" + "\n" + r"$Q(s, a)$ Table", ha="center", va="center", fontsize=7.8, color="#1e293b")
ax.text(32.8, 56.5, "Fallback LLM Router", ha="center", va="center", fontsize=7.5, fontstyle="italic", color="#475569")

# Inside C2: Card 2 (Architecture Manager) - Width 14.0
mgr_box = patches.FancyBboxPatch(
    (42.5, 54), 14.0, 29, boxstyle="round,pad=0.6",
    ec="#16a34a", fc="#ffffff", lw=1.2
)
ax.add_patch(mgr_box)
ax.text(49.5, 79.5, "Architecture\nManager", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#166534")
ax.plot([43.7, 55.3], [74.5, 74.5], color="#bbf7d0", lw=1.0)
ax.text(49.5, 70.0, r"Mutation Operator" + "\n" + r"$a_t \in \mathcal{A}$", ha="center", va="center", fontsize=7.8, color="#1e293b")
ax.text(49.5, 62.5, r"Invariant Check $\mathcal{V}$:" + "\nAcyclicity (DAG)", ha="center", va="center", fontsize=7.8, color="#1e293b")
ax.text(49.5, 56.5, "Path Reachability", ha="center", va="center", fontsize=7.5, fontstyle="italic", color="#475569")

# Inside C2: Card 3 (Dynamic Builder) - Width 14.0
bld_box = patches.FancyBboxPatch(
    (59.2, 54), 14.0, 29, boxstyle="round,pad=0.6",
    ec="#16a34a", fc="#ffffff", lw=1.2
)
ax.add_patch(bld_box)
ax.text(66.2, 79.5, "Dynamic\nBuilder", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#166534")
ax.plot([60.4, 72.0], [74.5, 74.5], color="#bbf7d0", lw=1.0)
ax.text(66.2, 70.0, "LangGraph\nCompilation", ha="center", va="center", fontsize=7.8, color="#1e293b")
ax.text(66.2, 62.5, "Compiled Workflow\n$W_{t+1}$ StateGraph", ha="center", va="center", fontsize=7.8, color="#1e293b")
ax.text(66.2, 56.5, "Latency <15 ms", ha="center", va="center", fontsize=7.5, fontstyle="italic", color="#15803d", fontweight="bold")

# Inter-card arrows inside C2 (Generous 2.7 unit arrow length)
ax.annotate("", xy=(42.5, 68), xytext=(39.8, 68), arrowprops=dict(arrowstyle="->", lw=1.8, color="#16a34a"))
ax.annotate("", xy=(59.2, 68), xytext=(56.5, 68), arrowprops=dict(arrowstyle="->", lw=1.8, color="#16a34a"))

# STAGE 4: Execution Container
s4_box = patches.FancyBboxPatch(
    (79.0, 52), 18.5, 41, boxstyle="round,pad=0.8",
    ec="#b91c1c", fc="#fef2f2", lw=1.6
)
ax.add_patch(s4_box)

s4_hdr = patches.FancyBboxPatch(
    (79.0, 85.5), 18.5, 7.5, boxstyle="round,pad=0.5",
    ec="#b91c1c", fc="#fee2e2", lw=1.2
)
ax.add_patch(s4_hdr)
ax.text(88.25, 89.2, "Stage 4: Execution", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#991b1b")

ax.text(88.25, 78.5, "Pruned Multi-Agent\nSpecialist Team", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#0f172a")
ax.text(88.25, 68.0, "Active Specialist Nodes:\nPlanner $\to$ Finalizer", ha="center", va="center", fontsize=7.8, color="#1e293b")
ax.text(88.25, 61.5, "Live Telemetry:\nTokens & Latency Logs", ha="center", va="center", fontsize=7.8, color="#1e293b")
ax.text(88.25, 55.5, "NVIDIA NIM Production", ha="center", va="center", fontsize=7.5, fontstyle="italic", color="#991b1b", fontweight="bold")

# Main horizontal arrows
ax.annotate("", xy=(24.0, 70), xytext=(20.0, 70), arrowprops=dict(arrowstyle="->", lw=2.2, color="#15803d"))
ax.annotate("", xy=(79.0, 70), xytext=(75.0, 70), arrowprops=dict(arrowstyle="->", lw=2.2, color="#15803d"))

# Clean Reward Feedback Loop (Routed beneath Pathway 2B in dedicated channel)
fb_path = patches.FancyBboxPatch(
    (27.0, 43.5), 57.0, 5.0, boxstyle="round,pad=0.4",
    ec="#dc2626", fc="#fff1f2", lw=1.1, linestyle="--"
)
ax.add_patch(fb_path)
ax.text(55.5, 46.0, "Runtime Reward Feedback Loop: $R_t$ updates Tabular Q-Table $Q(s_t, a_t)$", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#991b1b")

ax.annotate("", xy=(84.0, 46.0), xytext=(84.0, 52.0), arrowprops=dict(arrowstyle="->", lw=1.5, color="#dc2626", linestyle="--"))
ax.annotate("", xy=(32.8, 54.0), xytext=(32.8, 48.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#dc2626", linestyle="--"))


# ===========================================================================
# 2. BOTTOM PIPELINE: EXPERIMENTAL NEURAL PATHWAY (CONTRIBUTION C1)
# ===========================================================================

c1_box = patches.FancyBboxPatch(
    (2.5, 5), 95.0, 33.5, boxstyle="round,pad=0.8",
    ec="#ea580c", fc="#fff7ed", lw=1.8, linestyle="--"
)
ax.add_patch(c1_box)

c1_hdr = patches.FancyBboxPatch(
    (2.5, 32.5), 95.0, 6.0, boxstyle="round,pad=0.5",
    ec="#ea580c", fc="#ffedd5", lw=1.2
)
ax.add_patch(c1_hdr)
ax.text(50.0, 35.5, "Pathway 2A (Contribution C1): Experimental Task-Conditioned Neural Meta-Policy (Decoupled Offline Exploration)", ha="center", va="center", fontsize=10.0, fontweight="bold", color="#9a3412")

# Card 1: Encoders
enc_box = patches.FancyBboxPatch(
    (4.5, 8), 19.5, 22.5, boxstyle="round,pad=0.6",
    ec="#f97316", fc="#ffffff", lw=1.1
)
ax.add_patch(enc_box)
ax.text(14.25, 26.5, "Task & Arch\nEncoders", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#9a3412")
ax.plot([6.5, 22.0], [22.2, 22.2], color="#fed7aa", lw=1.0)
ax.text(14.25, 18.2, r"Architecture: $s_t \mapsto x_t$" + "\n" + r"Task Context: $c_t \mapsto z_t$", ha="center", va="center", fontsize=7.8, color="#1e293b")
ax.text(14.25, 12.0, r"Concat Feature Vector" + "\n" + r"$[x_t \Vert z_t] \in \mathbb{R}^d$", ha="center", va="center", fontsize=7.8, color="#1e293b")

# Card 2: Shared Neural Policy
mlp_box = patches.FancyBboxPatch(
    (28.0, 8), 19.5, 22.5, boxstyle="round,pad=0.6",
    ec="#f97316", fc="#ffffff", lw=1.1
)
ax.add_patch(mlp_box)
ax.text(37.75, 26.5, "Shared Neural\nPolicy $f_\\theta$", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#9a3412")
ax.plot([30.0, 45.5], [22.2, 22.2], color="#fed7aa", lw=1.0)
ax.text(37.75, 18.2, "2-Layer MLP (64 units)\nReLU Activation", ha="center", va="center", fontsize=7.8, color="#1e293b")
ax.text(37.75, 12.0, r"Temporal-Difference TD" + "\n" + r"Shared Across $\mathcal{T}_{train}$", ha="center", va="center", fontsize=7.8, color="#1e293b")

# Card 3: Deterministic Proxy Evaluator
eval_box = patches.FancyBboxPatch(
    (51.5, 8), 19.5, 22.5, boxstyle="round,pad=0.6",
    ec="#f97316", fc="#ffffff", lw=1.1
)
ax.add_patch(eval_box)
ax.text(61.25, 26.5, "Compatibility\nProxy Evaluator", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#9a3412")
ax.plot([53.5, 69.0], [22.2, 22.2], color="#fed7aa", lw=1.0)
ax.text(61.25, 18.2, r"Deterministic Score" + "\n" + r"$Q_{task}^{proxy} \in [0, 1]$", ha="center", va="center", fontsize=7.8, color="#1e293b")
ax.text(61.25, 12.0, "Offline Compatibility\n(Zero Live LLM Calls)", ha="center", va="center", fontsize=7.8, color="#1e293b")

# Card 4: Held-Out Generalization Test
gen_box = patches.FancyBboxPatch(
    (75.0, 8), 19.5, 22.5, boxstyle="round,pad=0.6",
    ec="#f97316", fc="#ffffff", lw=1.1
)
ax.add_patch(gen_box)
ax.text(84.75, 26.5, "Held-Out Task\nEvaluation", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#9a3412")
ax.plot([77.0, 92.5], [22.2, 22.2], color="#fed7aa", lw=1.0)
ax.text(84.75, 18.2, "5 Held-Out Tasks\n5 Random Seeds", ha="center", va="center", fontsize=7.8, color="#1e293b")
ax.text(84.75, 12.0, "$K=2$ Fine-Tuning Steps\nStability vs Scratch", ha="center", va="center", fontsize=7.8, color="#1e293b")

# Horizontal arrows inside C1 (Clear 4.0 unit arrow space)
ax.annotate("", xy=(28.0, 18.5), xytext=(24.0, 18.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#ea580c"))
ax.annotate("", xy=(51.5, 18.5), xytext=(47.5, 18.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#ea580c"))
ax.annotate("", xy=(75.0, 18.5), xytext=(71.0, 18.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#ea580c"))

out_dirs = [
    Path("Research/figures"),
    Path("Research_Paper/figures"),
    Path("research_results/figures")
]
for d in out_dirs:
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "figure_01_architecture_overview.png", bbox_inches="tight")
    fig.savefig(d / "figure_01_architecture_overview.pdf", bbox_inches="tight")

# Also copy to brain artifact directory
brain_dir = Path(r"C:\Users\navee\.gemini\antigravity\brain\512773e5-eefc-4b54-9e23-79d9a9906a1a")
fig.savefig(brain_dir / "figure_01_redesigned.png", bbox_inches="tight")

plt.close(fig)
print("Saved refined Figure 1 successfully in all directories including brain!")
