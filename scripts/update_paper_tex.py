"""
Script to apply table consolidation and narrative updates to Research/main.tex and Research_Paper/main.tex.
"""

from pathlib import Path

def update_file(p: Path):
    text = p.read_text(encoding="utf-8")
    norm = text.replace("\r\n", "\n")

    # 1. Update Benchmark Suite text & remove old table_02_benchmark_suite.tex input
    old_bmark = r"""\subsection{Benchmark Task Suite}
To evaluate adaptation across diverse requirements, we construct a 18-task benchmark suite across six functional categories (3 tasks per category), summarized in Table~\ref{tab:benchmark_suite}.

\input{tables/table_02_benchmark_suite.tex}"""

    new_bmark = r"""\subsection{Benchmark Task Suite}
To evaluate adaptation across diverse operational requirements, we construct an 18-task benchmark suite spanning six functional categories (Greeting, Mathematics, Coding, Information Research, Live Web Search, and Cross-Domain Multi-Stage; 3 tasks per category)."""

    if old_bmark in norm:
        norm = norm.replace(old_bmark, new_bmark)
        print(f"Replaced benchmark suite section in {p}")
    else:
        print(f"Warning: old_bmark not found in {p}")

    # 2. Replace old Table 3 input with combined table_02_domain_performance.tex
    old_arch = r"""\subsection{Topological Efficiency and Agent Pruning}
Table~\ref{tab:architecture_metrics} summarizes the architectural efficiency metrics achieved by AMAS across all functional domains in canonical Episode 2.

\input{tables/table_03_architecture_metrics.tex}"""

    new_arch = r"""\subsection{Topological Efficiency and Domain-Level Performance}
Table~\ref{tab:domain_performance} reports the comprehensive domain-level architectural efficiency and empirical live LLM performance measured against the NVIDIA NIM production API (\texttt{nvidia/nemotron-3.5-super-vl-120b-a12b}) across all functional domains.

\input{tables/table_02_domain_performance.tex}"""

    if old_arch in norm:
        norm = norm.replace(old_arch, new_arch)
        print(f"Replaced architecture section in {p}")
    else:
        print(f"Warning: old_arch not found in {p}")

    # 3. Update Table 1 reference sentence
    old_t1_ref = "Table~\\ref{tab:literature_comparison} provides a systematic comparison between our framework and prominent literature baselines."
    new_t1_ref = "Table~\\ref{tab:literature_comparison} provides a systematic comparison between our framework and prominent literature baselines (organized with evaluation dimensions in rows and systems in columns, with AMAS capabilities and empirical bounds bolded)."

    if old_t1_ref in norm:
        norm = norm.replace(old_t1_ref, new_t1_ref)
        print(f"Replaced Table 1 reference in {p}")

    # 4. Remove Table V input and replace with clean narrative
    old_t5 = r"""We evaluate Contribution C1 across 5 held-out tasks and 5 random seeds (100 total trajectories). Table~\ref{tab:neural_adaptation} reports the empirical results.

\input{tables/table_05_neural_adaptation.tex}"""

    new_t5 = r"""We evaluate Contribution C1 across 5 held-out tasks and 5 random seeds (100 total trajectories). On \texttt{task-analysis-001}, \texttt{task-research-001}, and \texttt{task-research-002}, the shared neural meta-policy, fresh scratch policy, and baseline all achieve an adaptation gain of $\Delta Q = 0.0000 \pm 0.0000$. On \texttt{task-coding-001}, exploratory mutations degrade performance from 1.00 to 0.60 ($t = -4.00, p = 0.0161$). Conversely, on \texttt{task-summarization-001}, the pre-trained shared meta-policy preserves 100\% stability across all seeds ($0.6000 \pm 0.0000$), acting as a protective regularizer against destructive mutations that degraded the scratch policy (20\% degradation rate, $0.4800 \pm 0.2683$)."""

    if old_t5 in norm:
        norm = norm.replace(old_t5, new_t5)
        print(f"Replaced Table 5 in {p}")
    else:
        print(f"Warning: old_t5 not found in {p}")

    # 5. Remove Table VI input
    old_t6 = r"""We conduct rigorous inferential statistical testing across all quantitative metrics. Table~\ref{tab:statistical_testing} summarizes the results.

\input{tables/table_06_statistical_testing.tex}"""

    new_t6 = r"""We conduct rigorous inferential statistical hypothesis testing across all quantitative performance metrics to validate observed resource gains and critical path compression."""

    if old_t6 in norm:
        norm = norm.replace(old_t6, new_t6)
        print(f"Replaced Table 6 in {p}")
    else:
        print(f"Warning: old_t6 not found in {p}")

    if "\r\n" in text:
        norm = norm.replace("\n", "\r\n")

    p.write_text(norm, encoding="utf-8")
    print(f"Saved {p} successfully!")

for p in [Path("Research/main.tex"), Path("Research_Paper/main.tex")]:
    update_file(p)

print("Done updating LaTeX sources.")

