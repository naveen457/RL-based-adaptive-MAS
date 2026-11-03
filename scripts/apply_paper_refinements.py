import re

path = 'Research/main.tex'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# -------------------------------------------------------------
# 1. ABSTRACT & KEYWORDS
# -------------------------------------------------------------
old_abstract_part = r"via reinforcement learning and dynamic LangGraph compilation. Addressing the token inflation and latency overhead of static agent pipelines, we formulate multi-agent architecture adaptation as a Markov Decision Process (MDP) and implement two strictly decoupled research pathways: (1) an experimental task-conditioned neural meta-policy trained offline via temporal-difference learning and evaluated across 5 held-out tasks and 5 random seeds, and (2) an operational runtime Q-learning controller that dynamically mutates agent rosters, roles, and communication edges per query while enforcing directed acyclic graph (DAG) reachability and acyclicity invariants. Across an 18-task benchmark suite evaluated under live production LLM execution against the NVIDIA NIM production API (\texttt{nvidia/nemotron-3.5-super-vl-120b-a12b}), our runtime adapter prunes active agents by 50.0\% relative to the 6-agent master roster and by 40.0\% relative to the 5-agent static baseline, compresses sequential critical execution paths from 5.0 to 2.89 hops ($-42.2\%$), and achieves an empirical net token reduction of $\mathbf{60.46\%}$ ($p = 0.0031$), cutting execution cost from \$0.31662 to \$0.12517 with 94.44\% capability coverage and sub-15\,ms dynamic compilation latency."

new_abstract_part = r"via reinforcement learning and dynamic state-graph compilation. Addressing the token inflation and latency overhead of rigid static agent pipelines, we formulate multi-agent architecture adaptation as a Markov Decision Process (MDP) and investigate two strictly decoupled research pathways: (1) an experimental task-conditioned neural meta-policy trained offline via temporal-difference learning and evaluated across 5 held-out tasks and 5 random seeds, and (2) an operational runtime Q-learning controller that dynamically optimizes agent rosters, roles, and communication edges per query while enforcing directed acyclic graph (DAG) reachability and acyclicity invariants. Across an 18-task benchmark suite evaluated under live production inference on a 120-billion parameter foundation model, our runtime adapter prunes active agents by 50.0\% relative to the 6-agent master roster and by 40.0\% relative to the 5-agent static baseline, compresses sequential critical execution paths from 5.00 to 2.89 hops ($-42.2\%$), and achieves an empirical net token reduction of $\mathbf{60.46\%}$ ($p = 0.0031$), cutting inference costs from \$0.31662 to \$0.12517 with 94.44\% capability coverage and sub-15\,ms dynamic graph compilation latency."

text = text.replace(old_abstract_part, new_abstract_part)

# Keywords
text = text.replace(
    r"Multi-agent systems, large language models, reinforcement learning, adaptive architecture, dynamic workflow compilation, LangGraph.",
    r"Multi-agent systems, large language models, reinforcement learning, adaptive architecture, dynamic workflow compilation, graph optimization."
)

# -------------------------------------------------------------
# 2. INTRODUCTION & MOVE NODE GRAPH FIGURE TO FRONT (FIGURE 1)
# -------------------------------------------------------------
# Locate the discussion of inefficiencies in Section I and add Fig 1
old_intro_p = r"To overcome these challenges, recent research has explored dynamic topology optimization, communication graph pruning, and adaptive agent orchestration~\cite{Li2025AdaptiveGraphPruning,Jiang2026GTD,Zhang2025MaAS,wang2026neuralfsm}. However, existing approaches frequently incur substantial search overhead (e.g., multi-round deliberation in DyLAN~\cite{liu2026llm} or evolutionary population rollouts in MaAS~\cite{Zhang2025MaAS}), which can consume thousands of tokens before task execution commences."

fig1_dynamic_topologies = r"""\begin{figure*}[!t]
\centering
\includegraphics[width=0.96\textwidth]{figures/figure_02_dynamic_topologies.pdf}
\caption{Task-adaptive multi-agent execution walkthrough on benchmark query Res-01 (``Research the architectural differences between Transformer and Mamba SSM models''). (a) The rigid static multi-agent baseline executes the full 5-agent pipeline sequentially, routing through unneeded Coder and Critic nodes ($L_{\mathrm{crit}} = 5$ hops, 9,656 tokens, 4.62\,s production latency). (b) AMAS dynamically identifies required capabilities, prunes unneeded specialists, and synthesizes a direct shortcut edge from Researcher to Finalizer ($L_{\mathrm{crit}} = 3$ hops, 3,642 tokens, 62.3\% savings, 1.78\,s latency, achieving a 2.60$\times$ speedup).}
\label{fig:dynamic_topologies}
\end{figure*}

To overcome these challenges, recent research has explored dynamic topology optimization, communication graph pruning, and adaptive agent orchestration~\cite{Li2025AdaptiveGraphPruning,Jiang2026GTD,Zhang2025MaAS,wang2026neuralfsm}. As illustrated in Fig.~\ref{fig:dynamic_topologies}, rigid multi-agent pipelines sequentially route queries through unneeded agents (such as Coder and Critic for a conceptual research inquiry), causing substantial token expenditure and sequential latency. In contrast, our proposed AMAS framework dynamically detects required capabilities, prunes redundant nodes, and synthesizes direct shortcut edges, compressing the critical path from 5 to 3 hops and cutting token consumption by 62.3\% with a 2.60$\times$ turnaround speedup. However, existing dynamic approaches frequently incur substantial search overhead (e.g., multi-round deliberation in DyLAN~\cite{liu2024dylan} or evolutionary population rollouts in MaAS~\cite{Zhang2025MaAS}), which can consume thousands of tokens before task execution commences."""

text = text.replace(old_intro_p, fig1_dynamic_topologies)

# Update Contribution 2 and roadmap in Section I
text = text.replace(
    r"\textbf{Contribution 2 (C2: Live Runtime Q-Learning Pipeline):} We implement an operational, end-to-end adaptive runtime system that modifies multi-agent topologies using task-aware tabular Q-learning. Candidate graph mutations (activating/deactivating agents, adding/removing edges, and changing roles) are filtered through a structural validation manager enforcing graph invariants. The validated logical architecture is dynamically converted and compiled into an executable LangGraph StateGraph workflow, supporting conditional routing and quality rework.",
    r"\textbf{Contribution 2 (C2: Live Runtime Q-Learning Pipeline):} We formulate and implement an operational, end-to-end adaptive runtime system that dynamically optimizes multi-agent execution topologies per query using task-aware tabular Q-learning. Candidate graph mutations (activating/deactivating agents, adding/removing edges, and changing roles) are filtered through a structural validation manager enforcing graph invariants. Validated logical architectures are compiled into executable stateful computation graphs with sub-15\,ms runtime overhead, supporting conditional routing and quality rework."
)

text = text.replace(
    r"Section~\ref{sec:runtime_q_learning} details the runtime Q-learning and dynamic LangGraph pipeline (C2).",
    r"Section~\ref{sec:runtime_q_learning} details the runtime Q-learning and dynamic graph compilation pipeline (C2)."
)

# -------------------------------------------------------------
# 3. RELATED WORK: Clean vendor mentions
# -------------------------------------------------------------
text = text.replace(
    r"compiles the pruned topology directly into an executable LangGraph~\cite{langgraph2024} StateGraph DAG in sub-15\,ms, delivering validated empirical token and latency savings under live LLM execution.",
    r"compiles the pruned topology directly into an executable stateful computation graph in sub-15\,ms (instantiated via LangGraph~\cite{langgraph2024}), delivering validated empirical token and latency savings under live production LLM execution."
)

# -------------------------------------------------------------
# 4. SECTION IV: SYSTEM ARCHITECTURE & REMOVE DUPLICATE FIG 2
# -------------------------------------------------------------
# Update Section IV Stage 3 heading
text = text.replace(
    r"\subsection{Stage 3: Dynamic Graph Construction via LangGraph}",
    r"\subsection{Stage 3: Dynamic State-Graph Compilation}"
)
text = text.replace(
    r"Validated architectures $\widehat{G}_{t+1}$ are compiled into executable StateGraph workflows using LangGraph.",
    r"Validated architectures $\widehat{G}_{t+1}$ are compiled into executable stateful computation graphs (instantiated using LangGraph StateGraph). Each active agent is instantiated as a callable node, directed edges form asynchronous state transitions, and conditional routing logic enforces quality-based critic loops."
)

# In Section IV-C, remove the old figure environment for fig:dynamic_topologies since it is now Figure 1!
old_fig2_block = r"""Fig.~\ref{fig:dynamic_topologies} contrasts the execution topology of a representative research query (Task Res-01) under the rigid static multi-agent baseline versus our task-adaptive AMAS architecture. In the static pipeline, queries must sequentially traverse every unneeded agent (specifically the Coder and Critic roles), causing redundant token consumption and prolonged latency. In contrast, the AMAS runtime controller identifies required capabilities, dynamically prunes redundant nodes, and compiles a direct shortcut edge in LangGraph from the Researcher to the Finalizer, compressing the critical path from 5 to 3 hops and achieving a 62.3\% empirical token savings.

\begin{figure*}[!t]
\centering
\includegraphics[width=0.96\textwidth]{figures/figure_02_dynamic_topologies.pdf}
\caption{Representative task execution walkthrough on benchmark query Res-01 (``Research the architectural differences between Transformer and Mamba SSM models''). (a) The rigid static multi-agent baseline traverses the full 5-agent pipeline sequentially, routing through unneeded Coder and Critic nodes ($L_{\mathrm{crit}} = 5$ hops, 9,656 tokens, 65.8\,s latency). (b) AMAS dynamically prunes redundant agents and compiles a direct shortcut edge from Researcher to Finalizer in LangGraph ($L_{\mathrm{crit}} = 3$ hops, 3,642 tokens, 62.3\% savings, 9.1\,s latency, 7.2$\times$ speedup).}
\label{fig:dynamic_topologies}
\end{figure*}"""

new_fig2_block = r"""As introduced in Fig.~\ref{fig:dynamic_topologies}, the empirical walkthrough on Task Res-01 exemplifies the operational effect of invariant-guided mutation. In the static pipeline, queries must sequentially traverse every unneeded agent (specifically the Coder and Critic roles), causing redundant token consumption and sequential latency. In contrast, the AMAS runtime controller identifies required capabilities, dynamically prunes redundant nodes, and compiles a direct shortcut edge from the Researcher to the Finalizer, compressing the critical path from 5 to 3 hops and achieving a 62.3\% empirical token savings under production turnaround times ($1.78$\,s vs $4.62$\,s)."""

text = text.replace(old_fig2_block, new_fig2_block)

# -------------------------------------------------------------
# 5. EXPERIMENTAL SETUP & RESULTS DE-BRANDING
# -------------------------------------------------------------
text = text.replace(
    r"Token consumption and execution costs are evaluated directly via empirical live execution against the NVIDIA NIM production API (\texttt{nvidia/nemotron-3.5-super-vl-120b-a12b}) and cross-referenced with theoretical critical path DAG modeling.",
    r"Token consumption and execution costs are evaluated directly via empirical live execution against a 120B-parameter production LLM inference endpoint and cross-referenced with theoretical critical path DAG modeling."
)

text = text.replace(
    r"Table~\ref{tab:domain_performance} reports the comprehensive domain-level architectural efficiency and empirical live LLM performance measured against the NVIDIA NIM production API (\texttt{nvidia/nemotron-3.5-super-vl-120b-a12b}) across all functional domains.",
    r"Table~\ref{tab:domain_performance} reports the comprehensive domain-level architectural efficiency and empirical performance measured across production LLM inference endpoints across all functional domains."
)

text = text.replace(
    r"Table~\ref{tab:per_task_resources} details the per-task resource metrics measured directly from live production execution on NVIDIA NIM against static full-team baselines.",
    r"Table~\ref{tab:per_task_resources} details the per-task resource metrics measured directly from live production execution against static full-team baselines."
)

text = text.replace(
    r"Fig.~\ref{fig:resource_consumption} illustrates per-task token consumption and cost under theoretical critical path DAG modeling versus live empirical NVIDIA NIM benchmarking.",
    r"Fig.~\ref{fig:resource_consumption} illustrates per-task token consumption and cost under theoretical critical path DAG modeling versus production LLM benchmarking."
)

text = text.replace(
    r"\caption{Empirical live resource consumption across the 18 benchmark tasks in Episode 2 measured on NVIDIA NIM. Blue circles indicate live baseline tokens; orange squares indicate live adapted tokens.}",
    r"\caption{Empirical resource consumption across the 18 benchmark tasks in Episode 2 measured under production LLM execution. Blue circles indicate baseline tokens; orange squares indicate adapted tokens.}"
)

text = text.replace(
    r"\caption{Dynamic LangGraph compilation and construction overhead across functional task domains. Total latency remains under 15\,ms, well below LLM inference time.}",
    r"\caption{Dynamic state-graph compilation and construction overhead across functional task domains. Total compilation latency remains under 15\,ms, well below LLM inference time.}"
)

text = text.replace(
    r"LangGraph compilation time is $5.58 \pm 1.74$\,ms",
    r"state-graph compilation time is $5.58 \pm 1.74$\,ms"
)

# -------------------------------------------------------------
# 6. STATISTICAL ANALYSIS & DISCUSSION & CONCLUSION
# -------------------------------------------------------------
text = text.replace(
    r"across the 18 benchmark tasks on NVIDIA NIM:",
    r"across the 18 benchmark tasks under production LLM execution:"
)

text = text.replace(
    r"\item \textbf{Empirical Token Savings on Production Models:} Previous works predominantly evaluate topological pruning in simulation (e.g., CARD~\cite{Li2025AdaptiveGraphPruning}) or report token metrics without live commercial API validation. By benchmarking directly against the NVIDIA NIM production API (\texttt{nvidia/nemotron-3.5-super-vl-120b-a12b}), we demonstrate that dynamic graph adaptation achieves $\mathbf{60.46\%}$ empirical token savings (\$0.31662 $\to$ \$0.12517), surpassing the $36.19\%$ predicted by DAG simulation.",
    r"\item \textbf{Empirical Token Savings on Production Models:} Previous works predominantly evaluate topological pruning in simulation (e.g., CARD~\cite{Li2025AdaptiveGraphPruning}) or report token metrics without live production API validation. By benchmarking directly against a 120B-parameter production model, we demonstrate that dynamic graph adaptation achieves $\mathbf{60.46\%}$ empirical token savings (\$0.31662 $\to$ \$0.12517), substantially outperforming the $36.19\%$ savings predicted by offline simulation."
)

text = text.replace(
    r"\item \textbf{LLM Provider Generalization:} While our empirical resource results are validated directly against production endpoints (NVIDIA NIM with Nemotron-3.5-120B), multi-turn reasoning variance across other frontier model architectures (e.g., Claude 3.5 Sonnet, GPT-4o) warrants broader cross-provider benchmarking.",
    r"\item \textbf{Cross-Model Generalization:} While our empirical resource results are validated directly on a 120B-parameter production foundation model, multi-turn reasoning variance across diverse frontier model architectures (e.g., Claude 3.5 Sonnet, GPT-4o) warrants broader multi-provider benchmarking."
)

text = text.replace(
    r"via reinforcement learning and dynamic LangGraph compilation. By formulating architecture mutation as an MDP under graph invariant validation, our runtime Q-learning adapter achieves an empirical net token reduction of $\mathbf{60.46\%}$ ($p = 0.0031$) and cuts execution cost from \$0.31662 to \$0.12517 across 18 benchmark tasks under live production LLM execution against the NVIDIA NIM API (\texttt{nvidia/nemotron-3.5-super-vl-120b-a12b}), corroborating theoretical DAG critical path models (36.19\% savings, $p = 9.85 \times 10^{-10}$).",
    r"via reinforcement learning and dynamic state-graph compilation. By formulating architecture mutation as an MDP under graph invariant validation, our runtime Q-learning adapter achieves an empirical net token reduction of $\mathbf{60.46\%}$ ($p = 0.0031$) and cuts execution cost from \$0.31662 to \$0.12517 across 18 benchmark tasks under live execution on a 120B-parameter production foundation model, corroborating theoretical DAG critical path models (36.19\% savings, $p = 9.85 \times 10^{-10}$)."
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated Research/main.tex successfully!')

