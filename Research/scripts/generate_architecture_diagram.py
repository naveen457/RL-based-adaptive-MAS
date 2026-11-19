import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_architecture_diagram():
    # 15.2 x 6.8 inches publication layout
    fig, ax = plt.subplots(figsize=(15.2, 6.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

    # Colors
    c_c2_border = '#1E4E8C'       # Primary navy border
    c_c2_bg = '#F4F7FB'           # Very soft slate tint
    c_c1_border = '#64748B'       # Slate gray dashed
    c_c1_bg = '#FAFAFA'           # Very faint neutral
    c_box_bg = '#FFFFFF'          # White card
    c_box_border = '#CBD5E1'      # Subtle card border
    c_box_header = '#0F2942'      # Deep navy header
    c_text_body = '#334155'       # Dark slate body
    c_arrow = '#2B6CB0'           # Blue workflow arrow
    c_reject = '#DC2626'          # Red rejection
    c_feedback = '#15803D'        # Green feedback

    # Title Banner
    ax.text(50, 97.4, "AMAS: Task-Adaptive Multi-Agent System Architecture",
            ha='center', va='center', fontsize=13.5, fontweight='bold', color='#0F172A')
    ax.text(50, 94.2, "End-to-End Operational Execution Pipeline (C2) & Decoupled Offline Transfer Analysis (C1)",
            ha='center', va='center', fontsize=9.2, style='italic', color='#475569')

    # =========================================================================
    # C2 RUNTIME PIPELINE (Y: 26.5 to 91.8)
    # =========================================================================
    c2_box = FancyBboxPatch((1.2, 26.5), 97.6, 65.0,
                            boxstyle="round,pad=0.5,rounding_size=1.2",
                            facecolor=c_c2_bg, edgecolor=c_c2_border, linewidth=2.0)
    ax.add_patch(c2_box)
    
    # C2 Header Tag
    c2_tag = FancyBboxPatch((2.6, 88.2), 33.5, 3.1,
                            boxstyle="round,pad=0.2,rounding_size=0.5",
                            facecolor='#1E4E8C', edgecolor='none')
    ax.add_patch(c2_tag)
    ax.text(19.35, 89.7, "PRIMARY CONTRIBUTION C2: LIVE RUNTIME PIPELINE",
            ha='center', va='center', fontsize=8.2, fontweight='bold', color='#FFFFFF')

    # -------------------------------------------------------------------------
    # ROW 1: Stages 1, 2, 3, 4, 5, 6 (Y: 67.5 to 85.5)
    # -------------------------------------------------------------------------
    # Stage 1: User Query
    q_box = FancyBboxPatch((2.6, 67.5), 11.2, 18.0,
                           boxstyle="round,pad=0.3,rounding_size=0.7",
                           facecolor=c_box_bg, edgecolor=c_box_border, linewidth=1.2)
    ax.add_patch(q_box)
    ax.text(8.2, 82.5, "1. User Query", ha='center', va='center', fontsize=8.8, fontweight='bold', color=c_box_header)
    ax.text(8.2, 78.8, r"Task Input $q$", ha='center', va='center', fontsize=8.0, fontweight='semibold', color='#1E4E8C')
    ax.text(8.2, 73.5, "• User Prompt\n• Domain Context\n• Latency/Cost Goal", ha='center', va='center', fontsize=7.2, color='#475569')
    ax.text(8.2, 69.2, "(18 Benchmark Tasks)", ha='center', va='center', fontsize=6.6, style='italic', color='#64748B')

    # Arrow 1 -> 2
    ax.annotate("", xy=(15.4, 76.5), xytext=(13.8, 76.5),
                arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.6, mutation_scale=11))

    # Stage 2: Context Extraction
    ctx_box = FancyBboxPatch((15.4, 67.5), 13.0, 18.0,
                             boxstyle="round,pad=0.3,rounding_size=0.7",
                             facecolor=c_box_bg, edgecolor=c_box_border, linewidth=1.2)
    ax.add_patch(ctx_box)
    ax.text(21.9, 82.5, "2. Context Extractor", ha='center', va='center', fontsize=8.8, fontweight='bold', color=c_box_header)
    ax.text(21.9, 78.8, "Planner Parsing", ha='center', va='center', fontsize=8.0, fontweight='semibold', color='#1E4E8C')
    ax.text(21.9, 73.5, r"Extracts Context $\mathbf{c}_t$:" + "\n" + r"• Category $\kappa_t$" + "\n" + r"• Difficulty $d_t$" + "\n" + r"• Capabilities $\mathcal{C}_t$",
            ha='center', va='center', fontsize=7.2, color=c_text_body)
    ax.text(21.9, 69.2, r"Affordance Vector $\mathbf{f}_t$", ha='center', va='center', fontsize=6.6, color='#64748B')

    # Arrow 2 -> 3
    ax.annotate("", xy=(30.2, 76.5), xytext=(28.4, 76.5),
                arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.6, mutation_scale=11))

    # Stage 3: Q-Learning Controller
    qc_box = FancyBboxPatch((30.2, 67.5), 14.8, 18.0,
                            boxstyle="round,pad=0.3,rounding_size=0.7",
                            facecolor=c_box_bg, edgecolor=c_box_border, linewidth=1.2)
    ax.add_patch(qc_box)
    ax.text(37.6, 82.5, "3. Q-Controller", ha='center', va='center', fontsize=8.8, fontweight='bold', color=c_box_header)
    ax.text(37.6, 78.8, "Action Selection", ha='center', va='center', fontsize=8.0, fontweight='semibold', color='#1E4E8C')
    ax.text(37.6, 73.5, r"State $s_t = (G_t, \mathbf{c}_t)$" + "\n" + r"Policy $a_t = \mathrm{argmax}_a Q(s,a)$" + "\n" + r"72.2% Hit Rate (<1 ms)",
            ha='center', va='center', fontsize=7.1, color=c_text_body)
    ax.text(37.6, 69.2, "[LLM Fallback: 5/18 tasks]", ha='center', va='center', fontsize=6.6, style='italic', color='#64748B')

    # Arrow 3 -> 4
    ax.annotate("", xy=(46.8, 76.5), xytext=(45.0, 76.5),
                arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.6, mutation_scale=11))

    # Stage 4: Graph Mutation Manager
    mut_box = FancyBboxPatch((46.8, 67.5), 15.2, 18.0,
                             boxstyle="round,pad=0.3,rounding_size=0.7",
                             facecolor=c_box_bg, edgecolor=c_box_border, linewidth=1.2)
    ax.add_patch(mut_box)
    ax.text(54.4, 82.5, "4. Mutation Manager", ha='center', va='center', fontsize=8.8, fontweight='bold', color=c_box_header)
    ax.text(54.4, 78.8, "5 Topology Operators", ha='center', va='center', fontsize=8.0, fontweight='semibold', color='#1E4E8C')
    ax.text(54.4, 73.2, "$O_1$: Node Deactivation (12x)\n$O_2$: Edge Deactivation (8x)\n$O_3$: Node Reactivation\n$O_4$: Shortcut Synthesis\n$O_5$: Edge Rewiring",
            ha='center', va='center', fontsize=6.8, color=c_text_body)

    # Arrow 4 -> 5
    ax.annotate("", xy=(63.8, 76.5), xytext=(62.0, 76.5),
                arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.6, mutation_scale=11))

    # Stage 5: Structural Invariant Validation
    inv_box = FancyBboxPatch((63.8, 67.5), 15.6, 18.0,
                             boxstyle="round,pad=0.3,rounding_size=0.7",
                             facecolor=c_box_bg, edgecolor=c_box_border, linewidth=1.2)
    ax.add_patch(inv_box)
    ax.text(71.6, 82.5, "5. Invariant Validation", ha='center', va='center', fontsize=8.8, fontweight='bold', color=c_box_header)
    ax.text(71.6, 78.8, r"Safety Gate $\mathcal{V}(G)$", ha='center', va='center', fontsize=8.0, fontweight='semibold', color='#1E4E8C')
    ax.text(71.6, 73.2, r"• Root/Terminal Preservation" + "\n" + r"• Acyclicity Check (DAG)" + "\n" + r"• Specialist Reachability" + "\n" + r"• Coverage $C_{\mathrm{cov}} \geq C_{\mathrm{min}}$",
            ha='center', va='center', fontsize=7.0, color=c_text_body)
    ax.text(71.6, 69.2, "100% DAG Cycle Prevention", ha='center', va='center', fontsize=6.6, style='italic', color='#166534')

    # Arrow 5 -> 6 (Valid branch)
    ax.annotate("", xy=(81.2, 76.5), xytext=(79.4, 76.5),
                arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.6, mutation_scale=11))
    ax.text(80.3, 78.5, "Valid", ha='center', va='center', fontsize=7.0, fontweight='bold', color='#16A34A')

    # Stage 6: Dynamic Graph Construction & LangGraph Compilation
    bld_box = FancyBboxPatch((81.2, 67.5), 16.0, 18.0,
                             boxstyle="round,pad=0.3,rounding_size=0.7",
                             facecolor=c_box_bg, edgecolor=c_box_border, linewidth=1.2)
    ax.add_patch(bld_box)
    ax.text(89.2, 82.5, "6. Build & Compile", ha='center', va='center', fontsize=8.8, fontweight='bold', color=c_box_header)
    ax.text(89.2, 78.8, "LangGraph StateGraph", ha='center', va='center', fontsize=8.0, fontweight='semibold', color='#1E4E8C')
    ax.text(89.2, 73.5, r"• Dynamic DAG Construction" + "\n" + r"  ($6.19 \pm 1.88$\,ms)" + "\n" + r"• StateGraph Compilation" + "\n" + r"  ($5.58 \pm 1.77$\,ms)" + "\n" + r"Low Overhead: 11.78 ms",
            ha='center', va='center', fontsize=7.0, color=c_text_body)
    ax.text(89.2, 69.2, "Conditional Routing Ready", ha='center', va='center', fontsize=6.6, color='#64748B')

    # REJECTION LOOP (Red curved arrow below Row 1: 5 -> 4)
    # Drops cleanly below box 5 and arcs smoothly to the bottom of box 4
    ax.annotate("", xy=(54.4, 67.5), xytext=(68.0, 67.5),
                arrowprops=dict(arrowstyle="-|>", color=c_reject, lw=1.5,
                                connectionstyle="arc3,rad=0.35", mutation_scale=11))
    ax.text(61.2, 61.2, r"Invalid Mutation: Reject ($-P_{\mathrm{reject}}$ penalty) & Re-sample",
            ha='center', va='center', fontsize=6.8, fontweight='bold', color=c_reject,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#FEF2F2", edgecolor="#FECACA", lw=0.6))

    # Downward Arrow 6 -> 7
    ax.annotate("", xy=(89.2, 53.0), xytext=(89.2, 67.5),
                arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.6, mutation_scale=11))

    # -------------------------------------------------------------------------
    # ROW 2: Stages 7, 8 (Y: 33.5 to 53.0)
    # -------------------------------------------------------------------------
    # Stage 7: Multi-Agent Execution Engine
    exec_box = FancyBboxPatch((55.0, 33.5), 42.2, 19.5,
                              boxstyle="round,pad=0.3,rounding_size=0.7",
                              facecolor=c_box_bg, edgecolor=c_box_border, linewidth=1.2)
    ax.add_patch(exec_box)
    ax.text(76.1, 49.8, "7. Multi-Agent Execution Engine (Compiled Stateful DAG)",
            ha='center', va='center', fontsize=8.8, fontweight='bold', color=c_box_header)
    
    # Active Agent badges inside Execution Engine
    agents = [
        ("Planner", "#DBEAFE", "#1E40AF"),
        ("Researcher", "#E0E7FF", "#3730A3"),
        ("Coder", "#DCFCE7", "#166534"),
        ("Tool", "#FEF3C7", "#92400E"),
        ("Critic", "#FEE2E2", "#991B1B"),
        ("Finalizer", "#F3E8FF", "#6B21A8"),
    ]
    for idx, (ag_name, bg_col, text_col) in enumerate(agents):
        bx = 56.6 + idx * 6.5
        by = 42.5
        abox = FancyBboxPatch((bx, by), 5.8, 4.6, boxstyle="round,pad=0.15,rounding_size=0.4",
                              facecolor=bg_col, edgecolor=text_col, linewidth=0.8)
        ax.add_patch(abox)
        ax.text(bx + 2.9, by + 2.3, ag_name, ha='center', va='center', fontsize=6.8, fontweight='bold', color=text_col)

    ax.text(76.1, 37.8, r"Dynamic Pruning: Only required specialists execute ($|V_{\mathrm{act}}| = 3.00 \pm 0.59$ vs. 5.0 static, 50% APR)" + "\n" + r"Critical path hops compressed: $L_{\mathrm{crit}} = 2.89 \pm 0.47$ vs. 5.0 static ($-42.2\%$, $p < 10^{-12}$)",
            ha='center', va='center', fontsize=6.9, color=c_text_body)

    # Arrow 7 -> 8
    ax.annotate("", xy=(52.5, 43.25), xytext=(55.0, 43.25),
                arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.6, mutation_scale=11))

    # Stage 8: Output Telemetry & Reward Calculation
    out_box = FancyBboxPatch((2.6, 33.5), 49.9, 19.5,
                             boxstyle="round,pad=0.3,rounding_size=0.7",
                             facecolor=c_box_bg, edgecolor=c_box_border, linewidth=1.2)
    ax.add_patch(out_box)
    ax.text(27.55, 49.8, "8. Output Telemetry & Composite Reward Calculation",
            ha='center', va='center', fontsize=8.8, fontweight='bold', color=c_box_header)
    ax.text(27.55, 44.5, "• Live Tokens Measured: 211,078 -> 83,457 (60.46% Empirical Reduction, p=0.0031)\n• Live Cost Measured: $0.31662 -> $0.12517 (-60.5% Inference Cost Savings)\n" + r"• Composite Reward: $R_t = 0.4 C_{\mathrm{cov}} + 0.3\,(1 - |V_{\mathrm{act}}|/6) + 0.3\,(1 - L_{\mathrm{crit}}/5) - P_{\mathrm{reject}}$",
            ha='center', va='center', fontsize=7.1, color=c_text_body)
    ax.text(27.55, 37.0, "[Verified Response Returned to User: 16/18 Full Capability, 94.44% Macro Coverage]",
            ha='center', va='center', fontsize=6.8, fontweight='bold', color='#15803D')

    # CLOSED-LOOP TD FEEDBACK ARROW: 8 -> 3 (Straight vertical upward green arrow)
    ax.annotate("", xy=(37.6, 67.5), xytext=(37.6, 53.0),
                arrowprops=dict(arrowstyle="-|>", color=c_feedback, lw=1.6, mutation_scale=11))
    ax.text(37.6, 57.0, r"Reward Telemetry & TD Q-Update: $Q(s_t, a_t) \leftarrow Q + \alpha [ R_t + \gamma \max Q - Q ]$",
            ha='center', va='center', fontsize=6.8, fontweight='bold', color=c_feedback,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#F0FDF4", edgecolor="#86EFAC", lw=0.7))

    # =========================================================================
    # C1 OFFLINE NEURAL PATHWAY (Y: 2.5 to 23.5)
    # =========================================================================
    c1_box = FancyBboxPatch((1.2, 2.5), 97.6, 21.0,
                            boxstyle="round,pad=0.5,rounding_size=1.2",
                            facecolor=c_c1_bg, edgecolor=c_c1_border, linewidth=1.4, linestyle='--')
    ax.add_patch(c1_box)

    # C1 Header Tag
    c1_tag = FancyBboxPatch((2.6, 20.4), 48.0, 2.8,
                            boxstyle="round,pad=0.2,rounding_size=0.5",
                            facecolor='#475569', edgecolor='none')
    ax.add_patch(c1_tag)
    ax.text(26.6, 21.8, "OFFLINE EXPERIMENTAL PATH (C1): NEURAL CROSS-TASK TRANSFER",
            ha='center', va='center', fontsize=7.8, fontweight='bold', color='#FFFFFF')
    ax.text(77.0, 21.8, "[Evaluated offline on synthetic compatibility proxy; strictly does not control live C2 runtime]",
            ha='center', va='center', fontsize=7.0, style='italic', color='#64748B')

    # C1-A: Dual Continuous Encoders
    c1a_box = FancyBboxPatch((4.0, 5.0), 26.5, 13.5,
                             boxstyle="round,pad=0.3,rounding_size=0.6",
                             facecolor=c_box_bg, edgecolor=c_box_border, linewidth=1.0)
    ax.add_patch(c1a_box)
    ax.text(17.25, 15.5, "Dual Continuous Encoders", ha='center', va='center', fontsize=8.0, fontweight='bold', color=c_box_header)
    ax.text(17.25, 10.5, r"• State Encoder: $\mathbf{x}_t = E_G(s_t) \in \mathbb{R}^{32}$" + "\n" + r"• Task Encoder: $\mathbf{z}_t = E_C(\mathbf{c}_t) \in \mathbb{R}^{16}$" + "\n" + r"Joint Embedding: $[\mathbf{x}_t, \mathbf{z}_t]$",
            ha='center', va='center', fontsize=7.0, color=c_text_body)

    # Arrow C1-A -> C1-B
    ax.annotate("", xy=(36.0, 11.75), xytext=(30.5, 11.75),
                arrowprops=dict(arrowstyle="-|>", color='#64748B', lw=1.3, mutation_scale=10))

    # C1-B: Shared MLP Policy
    c1b_box = FancyBboxPatch((36.0, 5.0), 27.0, 13.5,
                             boxstyle="round,pad=0.3,rounding_size=0.6",
                             facecolor=c_box_bg, edgecolor=c_box_border, linewidth=1.0)
    ax.add_patch(c1b_box)
    ax.text(49.5, 15.5, "Shared MLP Policy Network", ha='center', va='center', fontsize=8.0, fontweight='bold', color=c_box_header)
    ax.text(49.5, 10.5, r"• 2-Layer MLP ($d=64$, ReLU)" + "\n" + r"• Predicts Mutation Distribution $q_\theta$" + "\n" + r"• Trained across $\mathcal{T}_{\mathrm{train}}$ domains" + "\n" + r"Target: $y_t = R_t + \gamma \max_{a'} Q(s', a')$",
            ha='center', va='center', fontsize=7.0, color=c_text_body)

    # Arrow C1-B -> C1-C
    ax.annotate("", xy=(68.5, 11.75), xytext=(63.0, 11.75),
                arrowprops=dict(arrowstyle="-|>", color='#64748B', lw=1.3, mutation_scale=10))

    # C1-C: Held-out Evaluation
    c1c_box = FancyBboxPatch((68.5, 5.0), 28.5, 13.5,
                             boxstyle="round,pad=0.3,rounding_size=0.6",
                             facecolor=c_box_bg, edgecolor=c_box_border, linewidth=1.0)
    ax.add_patch(c1c_box)
    ax.text(82.75, 15.5, "Held-Out Transfer Evaluation", ha='center', va='center', fontsize=8.0, fontweight='bold', color=c_box_header)
    ax.text(82.75, 10.5, r"• Evaluated on 5 Held-Out Tasks (5 Seeds)" + "\n" + r"• Bounded Action Space: $\Delta Q = 0.00 \pm 0.00$" + "\n" + r"• 100% Stability on Summarization ($T_5$)" + "\n" + r"• Identifies action ceiling in non-generative roles",
            ha='center', va='center', fontsize=7.0, color=c_text_body)

    # Dotted link from Stage 2 Context to C1
    ax.annotate("", xy=(17.25, 18.5), xytext=(21.9, 67.5),
                arrowprops=dict(arrowstyle="-|>", color='#94A3B8', lw=1.1, linestyle=':',
                                connectionstyle="arc3,rad=-0.12", mutation_scale=10))
    ax.text(16.5, 30.5, "Shared Task\nContext Vector", ha='center', va='center', fontsize=6.6, style='italic', color='#64748B')

    plt.tight_layout(pad=0.1)
    
    # Save PDF and PNG
    pdf_path = 'figures/amas_system_architecture.pdf'
    png_path = 'figures/amas_system_architecture.png'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated {pdf_path} and {png_path} successfully!")

if __name__ == '__main__':
    create_architecture_diagram()
