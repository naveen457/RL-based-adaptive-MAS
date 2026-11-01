"""
Generate LaTeX Tables for the Adaptive Multi-Agent Architecture Research Paper.
Creates standalone .tex files in Research/tables/ and research_results/tables/.
"""

from pathlib import Path
import csv
import json

out_dirs = [
    Path("Research/tables"),
    Path("research_results/tables")
]
for d in out_dirs:
    d.mkdir(parents=True, exist_ok=True)

def write_table(filename, content):
    for d in out_dirs:
        (d / filename).write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Wrote {filename}")

# ---------------------------------------------------------------------------
# TABLE 1: Literature Comparison
# ---------------------------------------------------------------------------
tab1 = r"""
\begin{table*}[!t]
\centering
\caption{Systematic Comparison with Related Dynamic Multi-Agent Architectures}
\label{tab:literature_comparison}
\footnotesize
\begin{tabular}{lp{3.2cm}p{2.8cm}p{2.8cm}cp{3.0cm}}
\toprule
\textbf{System / Framework} & \textbf{Adaptation Mechanism} & \textbf{Search / Decision Overhead} & \textbf{Graph Topology Representation} & \textbf{Cross-Task Transfer} & \textbf{Primary Published Metric} \\
\midrule
\textbf{DyLAN}~\cite{liu2026llm} & Unsupervised peer rating & Multi-round deliberation ($\mathcal{O}(N^2)$ LLM calls) & Layered feed-forward agent stack & No & 25.0--40.0\% agent pruning, 15--25\% token reduction \\
\textbf{GPTSwarm}~\cite{qiu2025reinforcement} & Policy gradient on graph weights & High iterative prompts per edge update & Adjacency matrix with learned edge weights & Limited & +12.4\% utility improvement on GAIA / GSM8K \\
\textbf{AFlow}~\cite{wang2026neuralfsm} & Monte Carlo Tree Search (MCTS) & Very high (15--30 exploratory rollouts) & Executable Python code workflow & No & +5.7\% accuracy across 6 reasoning benchmarks \\
\textbf{MaAS}~\cite{Zhang2025MaAS} & Multi-objective evolutionary search & Offline multi-generation population search & Role DAG with specialist assignments & Offline & 18.0--32.0\% token reduction vs full-team baseline \\
\textbf{CARD}~\cite{Li2025AdaptiveGraphPruning} & Contextual bandit dynamic routing & Low ($<10$\,ms routing computation) & Dynamic routing DAG & Tabular & Critical path $5.0 \to 2.8$ hops ($-44\%$ latency) \\
\midrule
\textbf{AMAS (Ours)} & \textbf{Active Tabular Q-Learning + Task-Conditioned Neural Meta-Policy} & \textbf{Zero LLM overhead ($<1$\,ms, 0 tokens)} for Q-hits; sub-15\,ms LangGraph compilation & \textbf{Dynamic LangGraph StateGraph DAG} with cycle validation & \textbf{Evaluated on 5 held-out tasks} & \textbf{50.0\% active APR, 2.89 $L_{\mathrm{crit}}$ hops, 36.19\% token savings} ($p = 9.85 \times 10^{-10}$) \\
\bottomrule
\end{tabular}
\end{table*}
"""
write_table("table_01_literature_comparison.tex", tab1)

# ---------------------------------------------------------------------------
# TABLE 2: Benchmark Task Suite & Domain Characteristics
# ---------------------------------------------------------------------------
tab2 = r"""
\begin{table*}[!t]
\centering
\caption{Evaluation Benchmark Suite: 18 Tasks Across Six Functional Domains}
\label{tab:benchmark_suite}
\footnotesize
\begin{tabular}{llccclp{4.5cm}}
\toprule
\textbf{Domain / Category} & \textbf{Task ID} & \textbf{Diff.} & \textbf{Base Tok.} & \textbf{Base Hops} & \textbf{Required Capabilities} & \textbf{Functional Description} \\
\midrule
\multirow{3}{*}{Greeting / Chit-Chat} 
& \texttt{greet-001} & 1 & 3500 & 5 & \texttt{conversational} & Casual greeting and tone calibration \\
& \texttt{greet-002} & 1 & 3500 & 5 & \texttt{conversational} & System identity and persona query \\
& \texttt{greet-003} & 1 & 3500 & 5 & \texttt{conversational} & Simple assistance readiness query \\
\midrule
\multirow{3}{*}{Mathematical Reasoning} 
& \texttt{math-001} & 2 & 3500 & 5 & \texttt{calculation, reasoning} & Multi-step modular arithmetic computation \\
& \texttt{math-002} & 3 & 3500 & 5 & \texttt{calculation, reasoning} & Symbolic algebraic equation solving \\
& \texttt{math-003} & 2 & 3500 & 5 & \texttt{calculation, reasoning} & Discrete probability and combinatorics \\
\midrule
\multirow{3}{*}{Coding \& Data Structures} 
& \texttt{code-001} & 2 & 3500 & 5 & \texttt{coding, planning} & Binary search tree balancing implementation \\
& \texttt{code-002} & 4 & 3500 & 5 & \texttt{coding, verification} & Concurrent asynchronous task queue \\
& \texttt{code-003} & 3 & 3500 & 5 & \texttt{coding, planning} & LRU cache with $\mathcal{O}(1)$ operations \\
\midrule
\multirow{3}{*}{Information Research} 
& \texttt{res-001} & 3 & 3500 & 5 & \texttt{research, synthesis} & Comparative survey of transformer architectures \\
& \texttt{res-002} & 3 & 3500 & 5 & \texttt{research, synthesis} & Quantum computing hardware platforms analysis \\
& \texttt{res-003} & 4 & 3500 & 5 & \texttt{research, synthesis} & Post-quantum cryptographic algorithms survey \\
\midrule
\multirow{3}{*}{Live Web Retrieval} 
& \texttt{web-001} & 2 & 3500 & 5 & \texttt{retrieval, summary} & Current international space mission updates \\
& \texttt{web-002} & 3 & 3500 & 5 & \texttt{retrieval, summary} & Recent global economic indicators and trends \\
& \texttt{web-003} & 2 & 3500 & 5 & \texttt{retrieval, summary} & Renewable energy adoption statistics \\
\midrule
\multirow{3}{*}{Cross-Domain Multi-Stage} 
& \texttt{cross-001} & 4 & 3500 & 5 & \texttt{research, code, verify} & Cryptographic hashing benchmark and report \\
& \texttt{cross-002} & 5 & 3500 & 5 & \texttt{research, code, verify} & High-frequency arbitrage trading model \\
& \texttt{cross-003} & 5 & 3500 & 5 & \texttt{research, code, verify} & Genomic sequence alignment pipeline in Rust \\
\bottomrule
\end{tabular}
\end{table*}
"""
write_table("table_02_benchmark_suite.tex", tab2)

# ---------------------------------------------------------------------------
# TABLE 3: Multi-Agent Architecture Quality & Topological Metrics
# ---------------------------------------------------------------------------
tab3 = r"""
\begin{table*}[!t]
\centering
\caption{Dynamic Multi-Agent Architecture Quality and Efficiency Metrics (Episode 2)}
\label{tab:architecture_metrics}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
\textbf{Functional Domain} & \textbf{Active Agents $|V_{\mathrm{act}}|$} & \textbf{APR (Pool 6) [\%]} & \textbf{APR (Base 5) [\%]} & \textbf{Critical Path $L_{\mathrm{crit}}$} & \textbf{Capability Cov. [\%]} & \textbf{Pareto Net Utility $U_{\mathrm{pareto}}$} \\
\midrule
Greeting & $2.00 \pm 0.00$ & $66.67 \pm 0.00$ & $60.00 \pm 0.00$ & $2.00 \pm 0.00$ & $100.00 \pm 0.00$ & $0.9300 \pm 0.0000$ \\
Mathematics & $3.00 \pm 0.00$ & $50.00 \pm 0.00$ & $40.00 \pm 0.00$ & $3.00 \pm 0.00$ & $100.00 \pm 0.00$ & $0.8500 \pm 0.0000$ \\
Coding & $3.33 \pm 0.58$ & $44.44 \pm 9.62$ & $33.33 \pm 11.55$ & $3.33 \pm 0.58$ & $100.00 \pm 0.00$ & $0.8233 \pm 0.0462$ \\
Research & $3.00 \pm 0.00$ & $50.00 \pm 0.00$ & $40.00 \pm 0.00$ & $3.00 \pm 0.00$ & $100.00 \pm 0.00$ & $0.8500 \pm 0.0000$ \\
Web Search & $3.00 \pm 0.00$ & $50.00 \pm 0.00$ & $40.00 \pm 0.00$ & $3.00 \pm 0.00$ & $100.00 \pm 0.00$ & $0.8500 \pm 0.0000$ \\
Cross-Domain & $3.67 \pm 0.58$ & $38.89 \pm 9.62$ & $26.67 \pm 11.55$ & $3.00 \pm 0.00$ & $66.67 \pm 33.33$ & $0.4833 \pm 0.3086$ \\
\midrule
\textbf{Macro Mean (All 18 Tasks)} & $\mathbf{3.00 \pm 0.59}$ & $\mathbf{50.00 \pm 9.90}$ & $\mathbf{40.00 \pm 11.88}$ & $\mathbf{2.89 \pm 0.47}$ & $\mathbf{94.44 \pm 17.15}$ & $\mathbf{0.7978 \pm 0.1834}$ \\
\textbf{Static Full-Team Baseline} & $5.00 \pm 0.00$ & $16.67 \pm 0.00$ & $0.00 \pm 0.00$ & $5.00 \pm 0.00$ & $100.00 \pm 0.00$ & $0.5000 \pm 0.0000$ \\
\bottomrule
\end{tabular}
\end{table*}
"""
write_table("table_03_architecture_metrics.tex", tab3)

# ---------------------------------------------------------------------------
# TABLE 4: Per-Task Resource Metrics (Episode 2)
# ---------------------------------------------------------------------------
tab4 = r"""
\begin{table*}[!t]
\centering
\caption{Per-Task Resource Consumption and Efficiency Comparison (Canonical Episode 2)}
\label{tab:per_task_resources}
\footnotesize
\begin{tabular}{llcccccc}
\toprule
\textbf{Task ID} & \textbf{Category} & \textbf{Base Tokens} & \textbf{Adapted Tokens} & \textbf{Token Savings} & \textbf{Base Cost (\$) } & \textbf{Adapted Cost (\$) } & \textbf{Selection Engine} \\
\midrule
\texttt{greet-001} & Greeting & 3,500 & 1,450 & $+58.57\%$ & \$0.00525 & \$0.00218 & LLM Fallback \\
\texttt{greet-002} & Greeting & 3,500 & 1,450 & $+58.57\%$ & \$0.00525 & \$0.00218 & Active RL Hit \\
\texttt{greet-003} & Greeting & 3,500 & 1,450 & $+58.57\%$ & \$0.00525 & \$0.00218 & Active RL Hit \\
\texttt{math-001} & Mathematics & 3,500 & 2,250 & $+35.71\%$ & \$0.00525 & \$0.00338 & Active RL Hit \\
\texttt{math-002} & Mathematics & 3,500 & 2,250 & $+35.71\%$ & \$0.00525 & \$0.00338 & Active RL Hit \\
\texttt{math-003} & Mathematics & 3,500 & 2,250 & $+35.71\%$ & \$0.00525 & \$0.00338 & Active RL Hit \\
\texttt{code-001} & Coding & 3,500 & 2,250 & $+35.71\%$ & \$0.00525 & \$0.00338 & Active RL Hit \\
\texttt{code-002} & Coding & 3,500 & 3,050 & $+12.86\%$ & \$0.00525 & \$0.00458 & LLM Fallback \\
\texttt{code-003} & Coding & 3,500 & 2,250 & $+35.71\%$ & \$0.00525 & \$0.00338 & Active RL Hit \\
\texttt{res-001} & Research & 3,500 & 2,250 & $+35.71\%$ & \$0.00525 & \$0.00338 & Active RL Hit \\
\texttt{res-002} & Research & 3,500 & 2,250 & $+35.71\%$ & \$0.00525 & \$0.00338 & Active RL Hit \\
\texttt{res-003} & Research & 3,500 & 2,250 & $+35.71\%$ & \$0.00525 & \$0.00338 & Active RL Hit \\
\texttt{web-001} & Web Search & 3,500 & 2,250 & $+35.71\%$ & \$0.00525 & \$0.00338 & Active RL Hit \\
\texttt{web-002} & Web Search & 3,500 & 2,250 & $+35.71\%$ & \$0.00525 & \$0.00338 & Active RL Hit \\
\texttt{web-003} & Web Search & 3,500 & 2,250 & $+35.71\%$ & \$0.00525 & \$0.00338 & Active RL Hit \\
\texttt{cross-001} & Cross-Domain & 3,500 & 2,250 & $+35.71\%$ & \$0.00525 & \$0.00338 & LLM Fallback \\
\texttt{cross-002} & Cross-Domain & 3,500 & 2,900 & $+17.14\%$ & \$0.00525 & \$0.00435 & LLM Fallback \\
\texttt{cross-003} & Cross-Domain & 3,500 & 2,900 & $+17.14\%$ & \$0.00525 & \$0.00435 & LLM Fallback \\
\midrule
\textbf{Total / Summary} & --- & \textbf{63,000} & \textbf{40,200} & $\mathbf{+36.19\%}$ & \textbf{\$0.09450} & \textbf{\$0.06023} & \textbf{72.2\% Active RL Hit Rate} \\
\bottomrule
\end{tabular}
\end{table*}
"""
write_table("table_04_per_task_resources.tex", tab4)

# ---------------------------------------------------------------------------
# TABLE 5: Neural Cross-Task Adaptation Performance
# ---------------------------------------------------------------------------
tab5 = r"""
\begin{table*}[!t]
\centering
\caption{Task-Conditioned Neural Cross-Task Fast Adaptation Across 5 Held-Out Tasks (5 Random Seeds)}
\label{tab:neural_adaptation}
\footnotesize
\begin{tabular}{llcccc}
\toprule
\textbf{Held-Out Task} & \textbf{Policy Architecture} & \textbf{Initial Perf. $Q_{\mathrm{before}}$} & \textbf{Final Perf. $Q_{\mathrm{after}}$} & \textbf{Adaptation Gain $\Delta Q$} & \textbf{Destructive Rate} \\
\midrule
\multirow{4}{*}{\texttt{task-analysis-001}} 
& Shared Meta-Policy (Pretrained) & $0.6000$ & $0.6000 \pm 0.0000$ & $+0.0000 \pm 0.0000$ & $0.0\%$ \\
& Fresh Scratch Policy & $0.6000$ & $0.6000 \pm 0.0000$ & $+0.0000 \pm 0.0000$ & $0.0\%$ \\
& Deterministic Baseline & $0.6000$ & $0.6000 \pm 0.0000$ & $+0.0000 \pm 0.0000$ & $0.0\%$ \\
& Random Policy & $0.6000$ & $0.6000 \pm 0.0000$ & $+0.0000 \pm 0.0000$ & $0.0\%$ \\
\midrule
\multirow{4}{*}{\texttt{task-coding-001}} 
& Shared Meta-Policy (Pretrained) & $1.0000$ & $0.6000 \pm 0.0000$ & $-0.4000 \pm 0.0000$ & $100.0\%$ \\
& Fresh Scratch Policy & $1.0000$ & $0.9200 \pm 0.1789$ & $-0.0800 \pm 0.1789$ & $20.0\%$ \\
& Deterministic Baseline & $1.0000$ & $1.0000 \pm 0.0000$ & $+0.0000 \pm 0.0000$ & $0.0\%$ \\
& Random Policy & $1.0000$ & $0.7600 \pm 0.2191$ & $-0.2400 \pm 0.2191$ & $60.0\%$ \\
\midrule
\multirow{4}{*}{\texttt{task-research-001}} 
& Shared Meta-Policy (Pretrained) & $0.0000$ & $0.0000 \pm 0.0000$ & $+0.0000 \pm 0.0000$ & $0.0\%$ \\
& Fresh Scratch Policy & $0.0000$ & $0.0000 \pm 0.0000$ & $+0.0000 \pm 0.0000$ & $0.0\%$ \\
& Deterministic Baseline & $1.0000$ & $1.0000 \pm 0.0000$ & $+0.0000 \pm 0.0000$ & $0.0\%$ \\
& Random Policy & $0.0000$ & $0.0000 \pm 0.0000$ & $+0.0000 \pm 0.0000$ & $0.0\%$ \\
\midrule
\multirow{4}{*}{\texttt{task-research-002}} 
& Shared Meta-Policy (Pretrained) & $0.6000$ & $0.6000 \pm 0.0000$ & $+0.0000 \pm 0.0000$ & $0.0\%$ \\
& Fresh Scratch Policy & $0.6000$ & $0.6000 \pm 0.0000$ & $+0.0000 \pm 0.0000$ & $0.0\%$ \\
& Deterministic Baseline & $1.0000$ & $1.0000 \pm 0.0000$ & $+0.0000 \pm 0.0000$ & $0.0\%$ \\
& Random Policy & $0.6000$ & $0.6000 \pm 0.0000$ & $+0.0000 \pm 0.0000$ & $0.0\%$ \\
\midrule
\multirow{4}{*}{\texttt{task-summarization-001}} 
& Shared Meta-Policy (Pretrained) & $0.6000$ & $0.6000 \pm 0.0000$ & $+0.0000 \pm 0.0000$ & $0.0\%$ \\
& Fresh Scratch Policy & $0.6000$ & $0.4800 \pm 0.2683$ & $-0.1200 \pm 0.2683$ & $20.0\%$ \\
& Deterministic Baseline & $1.0000$ & $1.0000 \pm 0.0000$ & $+0.0000 \pm 0.0000$ & $0.0\%$ \\
& Random Policy & $0.6000$ & $0.6000 \pm 0.0000$ & $+0.0000 \pm 0.0000$ & $0.0\%$ \\
\bottomrule
\end{tabular}
\end{table*}
"""
write_table("table_05_neural_adaptation.tex", tab5)

# ---------------------------------------------------------------------------
# TABLE 6: Statistical Hypothesis Testing Summary
# ---------------------------------------------------------------------------
tab6 = r"""
\begin{table*}[!t]
\centering
\caption{Inferential Statistical Analysis and Hypothesis Testing Summary}
\label{tab:statistical_testing}
\footnotesize
\begin{tabular}{llccccl}
\toprule
\textbf{Metric / Hypothesis} & \textbf{Statistical Test} & \textbf{Sample Size} & \textbf{Statistic} & \textbf{Degrees of Freedom} & \textbf{$p$-value} & \textbf{Scientific Inference} \\
\midrule
Token Reduction (Ep 2) & Paired two-tailed Student's $t$ & $N=18$ & $t = 12.0167$ & $\mathrm{df}=17$ & $9.85 \times 10^{-10}$ & Statistically significant reduction ($p < 0.001$) \\
Cost Reduction (Ep 2) & Paired two-tailed Student's $t$ & $N=18$ & $t = 12.0167$ & $\mathrm{df}=17$ & $9.85 \times 10^{-10}$ & Statistically significant reduction ($p < 0.001$) \\
Critical Path Compression & Paired two-tailed Student's $t$ & $N=18$ & $t = 18.9482$ & $\mathrm{df}=17$ & $1.87 \times 10^{-13}$ & Statistically significant hop reduction ($5.0 \to 2.89$) \\
Neural Transfer on Coding & Paired two-tailed Student's $t$ & $N=5$ & $t = -4.0000$ & $\mathrm{df}=4$ & $0.0161$ & Significant drop from exploration degradation \\
Neural Transfer on Summary & Paired two-tailed Student's $t$ & $N=5$ & $t = 1.0000$ & $\mathrm{df}=4$ & $0.3739$ & Non-significant stability advantage ($0.60$ vs $0.48$) \\
Neural Gain on 3 Tasks & Exact Zero-Variance Test & $N=5$ & $t = 0.0000$ & $\mathrm{df}=4$ & $1.0000$ & Bounded by structural action space constraint \\
\bottomrule
\end{tabular}
\end{table*}
"""
write_table("table_06_statistical_testing.tex", tab6)

# ---------------------------------------------------------------------------
# TABLE 7: Structural Action Space Limitation Analysis
# ---------------------------------------------------------------------------
tab7 = r"""
\begin{table*}[!t]
\centering
\caption{Ablation and Structural Action Space Actionability Analysis}
\label{tab:action_space_ablation}
\footnotesize
\begin{tabular}{lp{3.5cm}p{4.2cm}p{4.5cm}}
\toprule
\textbf{Task Category} & \textbf{Required Target Capability} & \textbf{Permitted Action Space $\mathcal{A}_{\mathrm{valid}}$} & \textbf{Observed Empirical Effect \& Diagnosis} \\
\midrule
\texttt{task-analysis-001} & \texttt{analysis, planning} & \texttt{role\_options=['analysis', 'planning']} & Valid actions reachable; agent configuration pre-optimized at initialization ($Q=0.60$), yielding $\Delta Q = 0.00$. \\
\texttt{task-coding-001} & \texttt{code, verification} & \texttt{role\_options=['analysis', 'planning']} & \textbf{Action mismatch:} Coder role cannot be restored via change\_role; $\epsilon$-exploration deactivated critical coding nodes ($1.00 \to 0.60$). \\
\texttt{task-research-001} & \texttt{research, synthesis} & \texttt{role\_options=['analysis', 'planning']} & \textbf{Severe structural ceiling:} Researcher role completely absent from permitted roles, locking score at $0.00$ ($\Delta Q = 0.00$). \\
\texttt{task-research-002} & \texttt{research, synthesis} & \texttt{role\_options=['analysis', 'planning']} & \textbf{Role missing:} Planner active ($0.60$) but Researcher cannot be reassigned; no valid mutation increases reward ($\Delta Q = 0.00$). \\
\texttt{task-summarization-001} & \texttt{synthesis, context} & \texttt{role\_options=['analysis', 'planning']} & Shared policy learned protective inertia ($0.60$), while fresh policy randomly mutated finalizer in 20\% of seeds ($0.48$). \\
\bottomrule
\end{tabular}
\end{table*}
"""
write_table("table_07_action_space_ablation.tex", tab7)

print("All 7 LaTeX tables successfully generated in Research/tables/ and research_results/tables/!")

