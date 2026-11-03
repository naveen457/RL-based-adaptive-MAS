"""
Refined, publication-grade compact Figure 2:
Representative Task Query Execution: Static MAS vs AMAS Dynamic Architecture.
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
    "font.size": 9.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig = plt.figure(figsize=(12.0, 4.4))

# ---------------------------------------------------------------------------
# TOP BANNER: Representative Task Query
# ---------------------------------------------------------------------------
ax_banner = fig.add_axes([0.03, 0.77, 0.94, 0.19])
ax_banner.axis("off")
ax_banner.set_xlim(0, 100)
ax_banner.set_ylim(0, 100)

banner_box = patches.FancyBboxPatch(
    (0, 8), 100, 84, boxstyle="round,pad=1.0",
    ec="#1e3a8a", fc="#f8fafc", lw=1.3
)
ax_banner.add_patch(banner_box)

ax_banner.text(
    50, 64,
    'Representative Benchmark Query [res-001]: "Research the architectural differences between Transformer and Mamba SSM models."',
    ha="center", va="center", fontsize=9.2, fontweight="bold", color="#0f172a"
)
ax_banner.text(
    50, 26,
    "Target Capabilities: [Research, Information Synthesis]   |   Irrelevant Capabilities: [Code Generation, Tool Execution, Verification]",
    ha="center", va="center", fontsize=8.2, color="#475569"
)

# ---------------------------------------------------------------------------
# SUBPLOTS: Left (Static) vs Right (AMAS Dynamic)
# ---------------------------------------------------------------------------
ax1 = fig.add_axes([0.03, 0.04, 0.45, 0.68])
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 100)
ax1.axis("off")

ax2 = fig.add_axes([0.52, 0.04, 0.45, 0.68])
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 100)
ax2.axis("off")

# Section Titles (Placed safely inside subplot coordinate space)
ax1.text(
    50, 94, "(a) Static Multi-Agent Baseline (Rigid Full-Team Traversal)",
    ha="center", va="center", fontsize=9.8, fontweight="bold", color="#b91c1c"
)
ax2.text(
    50, 94, "(b) AMAS Dynamic Execution (Task-Aware Optimized Path)",
    ha="center", va="center", fontsize=9.8, fontweight="bold", color="#15803d"
)

# Standard node x coordinates (0 to 100 scale)
node_x = {
    "Planner": 10,
    "Researcher": 30,
    "Coder": 50,
    "Critic": 70,
    "Finalizer": 90,
}
y_node = 54

def draw_agent_node(ax, name, role_desc, x, y, state="active"):
    """Draw circular agent node with role annotation."""
    if state == "active":
        fc = "#dcfce7"
        ec = "#16a34a"
        tc = "#166534"
        badge_fc = "#bbf7d0"
        badge_txt = "Active"
        ls = "-"
        lw = 1.5
    elif state == "redundant":
        fc = "#fee2e2"
        ec = "#dc2626"
        tc = "#991b1b"
        badge_fc = "#fecaca"
        badge_txt = "Forced Hop"
        ls = "--"
        lw = 1.5
    else:  # pruned
        fc = "#f8fafc"
        ec = "#cbd5e1"
        tc = "#94a3b8"
        badge_fc = "#f1f5f9"
        badge_txt = "Pruned"
        ls = ":"
        lw = 1.2

    # Agent Circle
    circle = patches.Circle((x, y), radius=6.5, fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=4)
    ax.add_patch(circle)
    ax.text(x, y, name, ha="center", va="center", fontsize=7.8, fontweight="bold", color=tc, zorder=5)

    # Role subtitle below
    ax.text(x, y - 10.5, role_desc, ha="center", va="top", fontsize=7.0, color="#475569", zorder=5)

    # State Badge above
    badge = patches.FancyBboxPatch(
        (x - 6.2, y + 7.8), 12.4, 4.8,
        boxstyle="round,pad=0.4", fc=badge_fc, ec=ec, lw=0.8, zorder=5
    )
    ax.add_patch(badge)
    ax.text(x, y + 10.2, badge_txt, ha="center", va="center", fontsize=6.2, fontweight="bold", color=tc, zorder=6)


# ----------------- PANEL 1: STATIC BASELINE -----------------
# Draw linear sequential traversal arrows
seq = ["Planner", "Researcher", "Coder", "Critic", "Finalizer"]
for i in range(len(seq) - 1):
    u = seq[i]
    v = seq[i+1]
    color = "#dc2626" if i >= 1 else "#2563eb"
    ax1.annotate(
        "", xy=(node_x[v] - 7.0, y_node), xytext=(node_x[u] + 7.0, y_node),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.8, shrinkA=0, shrinkB=0, zorder=3)
    )

draw_agent_node(ax1, "Planner", "Goal Decomp.", node_x["Planner"], y_node, state="active")
draw_agent_node(ax1, "Researcher", "Target Spec.", node_x["Researcher"], y_node, state="active")
draw_agent_node(ax1, "Coder", "No code needed!", node_x["Coder"], y_node, state="redundant")
draw_agent_node(ax1, "Critic", "Unneeded review", node_x["Critic"], y_node, state="redundant")
draw_agent_node(ax1, "Finalizer", "Synthesis", node_x["Finalizer"], y_node, state="active")

# Metrics Card Panel 1
m_box1 = patches.FancyBboxPatch(
    (2, 4), 96, 21, boxstyle="round,pad=0.8",
    ec="#dc2626", fc="#fff5f5", lw=1.1
)
ax1.add_patch(m_box1)
ax1.text(
    50, 18.0,
    "Rigid Linear Flaw: Traverses unneeded Coder & Critic nodes (No code requested)",
    ha="center", va="center", fontsize=7.8, fontweight="bold", color="#b91c1c"
)
ax1.text(
    50, 9.5,
    "Critical Path: 5 hops   |   Live Tokens: 9,656   |   Latency: 65.8s   |   Cost: $0.0145",
    ha="center", va="center", fontsize=7.5, color="#1e293b"
)


# ----------------- PANEL 2: AMAS DYNAMIC ADAPTATION -----------------
# Planner -> Researcher
ax2.annotate(
    "", xy=(node_x["Researcher"] - 7.0, y_node), xytext=(node_x["Planner"] + 7.0, y_node),
    arrowprops=dict(arrowstyle="->", color="#16a34a", lw=2.0, shrinkA=0, shrinkB=0, zorder=3)
)

# Curved shortcut: Researcher -> Finalizer (arching directly over Coder & Critic!)
ax2.annotate(
    "", xy=(node_x["Finalizer"] - 5.0, y_node + 5.5), xytext=(node_x["Researcher"] + 5.0, y_node + 5.5),
    arrowprops=dict(
        arrowstyle="->", color="#15803d", lw=2.2,
        connectionstyle="arc3,rad=-0.36", shrinkA=0, shrinkB=0, zorder=3
    )
)

# Shortcut Banner on the arch
s_box = patches.FancyBboxPatch(
    (36, 75.5), 48, 8.5, boxstyle="round,pad=0.5",
    fc="#dcfce7", ec="#16a34a", lw=0.9, zorder=6
)
ax2.add_patch(s_box)
ax2.text(
    60, 79.7,
    "Direct Compiled LangGraph Edge (Bypasses Pruned Nodes)",
    ha="center", va="center", fontsize=7.2, fontweight="bold", color="#166534", zorder=7
)

draw_agent_node(ax2, "Planner", "Task Routing", node_x["Planner"], y_node, state="active")
draw_agent_node(ax2, "Researcher", "Specialist Exec.", node_x["Researcher"], y_node, state="active")
draw_agent_node(ax2, "Coder", "Deactivated", node_x["Coder"], y_node, state="pruned")
draw_agent_node(ax2, "Critic", "Deactivated", node_x["Critic"], y_node, state="pruned")
draw_agent_node(ax2, "Finalizer", "Synthesis", node_x["Finalizer"], y_node, state="active")

# Metrics Card Panel 2
m_box2 = patches.FancyBboxPatch(
    (2, 4), 96, 21, boxstyle="round,pad=0.8",
    ec="#16a34a", fc="#f0fdf4", lw=1.1
)
ax2.add_patch(m_box2)
ax2.text(
    50, 18.0,
    "AMAS Topological Optimization: Q-Learning prunes redundant roles; compiles direct edge",
    ha="center", va="center", fontsize=7.8, fontweight="bold", color="#15803d"
)
ax2.text(
    50, 9.5,
    "Critical Path: 3 hops (40% reduction)   |   Live Tokens: 3,642 (62.3% Savings!)   |   Latency: 9.1s (7.2x Speedup)",
    ha="center", va="center", fontsize=7.5, fontweight="bold", color="#166534"
)

out_dirs = [
    Path("Research/figures"),
    Path("Research_Paper/figures"),
    Path("research_results/figures")
]
for d in out_dirs:
    d.mkdir(parents=True, exist_ok=True)
    png_p = d / "figure_02_dynamic_topologies.png"
    pdf_p = d / "figure_02_dynamic_topologies.pdf"
    fig.savefig(png_p, bbox_inches="tight")
    fig.savefig(pdf_p, bbox_inches="tight")

plt.close(fig)
print("Saved refined figure_02_dynamic_topologies.png and .pdf across all directories!")
