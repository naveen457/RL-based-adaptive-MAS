# Research Results Directory: Task-Adaptive Architecture Optimization for LLM-Based Multi-Agent Systems

This directory contains the complete experimental evaluation, extracted datasets, statistical analyses, literature benchmarks, publication-quality figures, and LaTeX manuscript files for the research project:

**"Task-Adaptive Architecture Optimization for LLM-Based Multi-Agent Systems using Reinforcement Learning and Cross-Task Adaptation"**

---

## Directory Organization

```text
research_results/
├── README.md                            # This directory guide
├── experiment_inventory.md              # Catalog of all 51 discovered experimental runs
├── experimental_findings.md             # Comprehensive research findings and evidence synthesis
│
├── extracted_data/                      # Structured CSV & JSON datasets extracted from raw logs
│   ├── experiment_inventory.csv         # Inventory of benchmark, interactive, and neural runs
│   ├── neural_adaptation_results.csv    # 100 evaluation trajectories (5 tasks x 5 seeds x 4 policies)
│   ├── benchmark_task_results.csv       # Per-task execution results across benchmark episodes
│   ├── architecture_metrics.csv         # Topological parsimony, density, and critical path metrics
│   ├── rl_action_metrics.csv            # RL action counts, acceptance rates, and mutation types
│   ├── graph_execution_metrics.csv      # LangGraph build, compile, and execution latencies
│   ├── resource_metrics.csv             # Token and cost consumption data (baseline vs adapted)
│   └── aggregated_metrics.json          # Overall benchmark summary JSON
│
├── statistical_analysis/                # Inferential statistics, hypothesis tests & CI95 tables
│   ├── statistical_analysis.md          # Master statistical methodology and overview
│   ├── neural_comparison.md             # Four-way policy comparison, paired t-tests, stability
│   ├── architecture_analysis.md         # Topological compression, APR%, and LangGraph timings
│   └── benchmark_analysis.md            # Resource efficiency, category breakdown, paired tests
│
├── literature/                          # State-of-the-art comparison with published works
│   ├── literature_comparison.md         # Narrative qualitative and quantitative analysis
│   └── literature_comparison.csv        # Tabular comparison (DyLAN, GPTSwarm, AFlow, MaAS, CARD)
│
├── figures/                             # Publication-quality vector (PDF) and raster (PNG, 300 DPI) figures
│   ├── figure_01_adaptation_curve.png / .pdf   # Neural adaptation trajectories (mean +/- std)
│   ├── figure_02_heldout_comparison.png / .pdf # Held-out task compatibility across policies
│   ├── figure_03_capability_vs_agents.png / .pdf # Active agents vs. capability coverage
│   ├── figure_04_token_comparison.png / .pdf   # Per-task token savings & negative expansions
│   ├── figure_05_critical_path.png / .pdf      # Critical path compression across domains
│   ├── figure_06_architecture_utility.png / .pdf # Structural complexity vs. net utility
│   ├── figure_07_graph_overhead.png / .pdf     # Sub-millisecond graph compilation decomposition
│   └── figure_08_rl_actions.png / .pdf         # RL decision breakdown & mutation space
│
├── tables/                              # Standalone CSV tables suitable for paper insertion
│   ├── benchmark_comparison.csv         # 18-task convergence metrics
│   ├── neural_policy_comparison.csv     # 5-task multi-seed neural comparison
│   ├── architecture_efficiency.csv      # Category-level architecture compression metrics
│   └── literature_comparison.csv        # Comprehensive literature benchmark table
│
└── paper_material/                      # IEEE-formatted manuscript assets
    ├── experimental_results.md          # Complete draft text for the Results section
    ├── discussion.md                    # Complete draft text for the Discussion section
    ├── figure_captions.md               # Detailed captions, subfigures & mandatory mapping table
    ├── latex_tables.tex                 # IEEE two-column booktabs LaTeX table environments
    └── latex_figures.tex                # IEEE two-column figure/figure* LaTeX environments
```

---

## Core Research Findings Summary

1. **Dynamic Architecture Optimization (Contribution 2):**
   - **36.19% net token reduction** on Episode 2 ($p < 0.001$, paired t-test).
   - **40.74% mean Agent Pruning Rate (APR)** across tasks, maintaining 100% capability coverage across 15 of 18 tasks.
   - **42.2% critical path compression** ($5.0 \to 2.89$ sequential hops), matching the latency reduction principles of CARD (2024).
   - **72.22% Active RL Policy Hit Rate** by Episode 2, delivering instant zero-LLM adaptations ($0$ search tokens, $<1$ ms latency).
   - **0.081 ms total adaptation overhead** ($T_{\text{build}} = 0.052$ ms, $T_{\text{comp}} = 0.029$ ms) with 100% LangGraph compilation success.
   - Transparent reporting of negative token savings on complex cross-domain tasks ($-5.7\%$ to $-24.3\%$), proving that optimization expands architectures when required.

2. **Neural Cross-Task Adaptation (Contribution 1):**
   - Empirical evaluation across 5 held-out tasks and 5 seeds revealed that adaptation gain is strictly bounded at $0.0000$ when missing capabilities lie outside the discrete action alphabet (`role_options=['analysis', 'planning']`).
   - The pre-trained shared meta-policy demonstrated superior stability on summarization tasks, completely avoiding destructive actions observed in fresh scratch initialization (0% vs 20% destructive rate).
   - Unconstrained $\epsilon$-greedy exploratory updates on coding tasks degraded initially optimal baseline topologies ($\Delta = -0.4000$), demonstrating the necessity of capability-preserving action masks.

---

## Reproducing Figures and Analyses

To reproduce the extracted datasets, statistical analyses, and publication figures:

```powershell
# 1. Extract datasets from raw experiment logs
python scratch/extract_and_build_datasets.py

# 2. Run statistical analyses and generate summary tables
python scratch/compute_statistical_analysis.py

# 3. Build literature comparisons
python scratch/build_literature_comparison.py

# 4. Generate all 8 publication figures (PNG + PDF)
python scratch/generate_publication_figures.py
```

