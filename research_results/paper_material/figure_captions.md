# Publication Figure Captions and Paper Mapping Document

This document provides complete, publication-grade captions, subfigure notes, and the mandatory mapping table linking every figure to its research contribution, coordinate axes, and corresponding paper section.

---

## 1. Mandatory Figure Mapping Table

| Figure | X-Axis | Y-Axis | Research Contribution | Target Paper Section |
|---|---|---|---|---|
| **Figure 1** (`figure_01_adaptation_curve`) | Fine-Tuning Adaptation Steps (0, 1, 2) | Task Compatibility Score ($Jaccard \in [0, 1]$) | Contribution 1: Neural Cross-Task Policy Adaptation | Section IV-A: Neural Adaptation Trajectories |
| **Figure 2** (`figure_02_heldout_comparison`) | Held-Out Task IDs Across 5 Domains | Final Task Performance Score ($[0, 1]$) | Contribution 1: Neural Held-Out Generalization | Section IV-B: Cross-Task Generalization & Stability |
| **Figure 3** (`figure_03_capability_vs_agents`) | Number of Active Agents ($|V_{\text{active}}|$) | Capability Coverage Rate (%) | Contribution 2: Dynamic Architecture Adaptation | Section V-A: Topological Parsimony & Coverage |
| **Figure 4** (`figure_04_token_comparison`) | 18 Benchmark Tasks (Grouped by Category) | Estimated Tokens ($L_{\text{crit}}$ DAG Proxy) | Contribution 2: Resource Efficiency & Token Reduction | Section V-B: Resource Efficiency & Negative Savings |
| **Figure 5** (`figure_05_critical_path`) | Functional Task Domains (6 Categories) | Critical Path Length ($L_{\text{crit}}$ Sequential Hops) | Contribution 2: Latency Compression | Section V-C: Critical Path & Latency Proxy |
| **Figure 6** (`figure_06_architecture_utility`) | Structural Complexity ($|V_{\text{active}}| + L_{\text{crit}}$) | Pareto Net Utility ($U_{\text{pareto}} \in [0, 1]$) | Contribution 2: Multi-Objective Pareto Optimization | Section V-D: Pareto Frontier & Net Utility |
| **Figure 7** (`figure_07_graph_overhead`) | Functional Task Categories | Latency (Milliseconds) | Contribution 2: Dynamic Graph Execution | Section V-E: LangGraph Compilation Overhead |
| **Figure 8** (`figure_08_rl_actions`) | (a) Episode; (b) Graph Mutation Action Types | (a) Decision Count; (b) Applied Action Count | Contribution 2: Runtime RL Adaptation Behavior | Section VI: Tabular Q-Policy Behavior & Mutation Space |

---

## 2. Comprehensive Figure Captions

### Figure 1: Neural Cross-Task Adaptation Trajectory
- **File:** `research_results/figures/figure_01_adaptation_curve.png` (PDF: `figure_01_adaptation_curve.pdf`)
- **Caption:** *Cross-task neural policy adaptation trajectories across 2 fine-tuning steps (mean $\pm 1$ standard deviation across 5 pseudo-random seeds: `[42, 101, 202, 303, 404]`). Subfigure (a) illustrates adaptation on `task-research-002`, where the policy is constrained by the discrete action alphabet ($Role \notin \mathcal{A}$), bounding adaptation gains at $0.0000$. Subfigure (b) illustrates `task-coding-001`, where unconstrained $\epsilon$-greedy exploratory updates degrade initially optimal baseline topologies ($1.0000 \to 0.6000$), demonstrating the necessity of capability-preserving action masking.*

### Figure 2: Held-Out Multi-Task Generalization
- **File:** `research_results/figures/figure_02_heldout_comparison.png` (PDF: `figure_02_heldout_comparison.pdf`)
- **Caption:** *Final held-out task compatibility score across 5 target tasks for 4 baseline policies (evaluated over 5 pseudo-random seeds). While structural action space boundaries prevent capability restoration on research and analysis tasks, the shared meta-policy demonstrates superior stability on summarization tasks, completely avoiding destructive actions observed in fresh scratch initialization.*

### Figure 3: Architecture Parsimony vs. Capability Coverage
- **File:** `research_results/figures/figure_03_capability_vs_agents.png` (PDF: `figure_03_capability_vs_agents.pdf`)
- **Caption:** *Architecture parsimony versus capability coverage rate on Episode 2 convergence. The static full-team baseline maintains 5 active agents (red diamond), incurring surplus coordination cost. Adapted AMAS topologies achieve 100% capability coverage with only 2 to 4 active agents across 15 of 18 tasks, establishing a Pareto-optimal frontier.*

### Figure 4: Per-Task Token Consumption Comparison
- **File:** `research_results/figures/figure_04_token_comparison.png` (PDF: `figure_04_token_comparison.pdf`)
- **Caption:** *Per-task estimated token consumption comparing the static 5-agent baseline (3,500 tokens) against adapted AMAS topologies across all 18 benchmark tasks. Green bars denote positive token savings (up to 58.6% on greetings and 35.7% on math/research). Negative savings on high-complexity cross-domain tasks (-5.7% on chess, -24.3% on smart contract audit) reflect necessary architectural expansion for multi-specialist verification.*

### Figure 5: Critical Path Length Compression
- **File:** `research_results/figures/figure_05_critical_path.png` (PDF: `figure_05_critical_path.pdf`)
- **Caption:** *Critical path length ($L_{\text{crit}}$ sequential hops) across 6 functional task domains. Adapted AMAS topologies compress the serial 5-hop baseline down to an average of 2.89 hops, reducing latency proxy overhead by 42.2% without degrading execution DAG validity.*

### Figure 6: Architecture Structural Complexity vs. Net Utility
- **File:** `research_results/figures/figure_06_architecture_utility.png` (PDF: `figure_06_architecture_utility.pdf`)
- **Caption:** *Structural complexity ($|V_{\text{active}}| + L_{\text{crit}}$) versus Pareto net utility ($U_{\text{pareto}}$). Topologies within complexity 5.0 to 7.0 achieve optimal net utilities ($0.770$ to $0.930$), outperforming the over-engineered static baseline ($U_{\text{pareto}} \approx 0.500$).*

### Figure 7: Dynamic LangGraph Latency Decomposition
- **File:** `research_results/figures/figure_07_graph_overhead.png` (PDF: `figure_07_graph_overhead.pdf`)
- **Caption:** *Dynamic LangGraph compilation and construction latency decomposition. The combined architecture adaptation pipeline overhead ($T_{\text{build}} + T_{\text{comp}}$) averages only 0.081 ms, confirming that dynamic LangGraph reconfiguration introduces negligible runtime latency.*

### Figure 8: RL Adaptation Behavior & Decision Breakdown
- **File:** `research_results/figures/figure_08_rl_actions.png` (PDF: `figure_08_rl_actions.pdf`)
- **Caption:** *Reinforcement learning adaptation behavior across training episodes. (a) Decision engine breakdown showing convergence from exploration to 72.2% instant active RL policy hits (0 tokens, $<1$ ms) by Episode 2. (b) Distribution of accepted graph mutation operations applied to optimize topologies.*

