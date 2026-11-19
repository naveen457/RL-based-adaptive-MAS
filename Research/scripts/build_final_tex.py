import re

with open('main_revised.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# =========================================================================
# 1. ABSTRACT REFINEMENT (Clear distinction of 16/18 complete vs 94.44% macro)
# =========================================================================
old_abstract = r"""In this paper, we propose AMAS (Adaptive Multi-Agent System), an end-to-end framework that dynamically optimizes multi-agent execution topologies per query through reinforcement learning and validated state-graph compilation. Formalizing architecture adaptation as a Markov Decision Process (MDP), we present two clearly demarcated contributions: (1) a real-time tabular Q-learning controller paired with a structural graph invariant manager that dynamically compiles topologies into executable DAGs in $<15$\,ms (C2); and (2) an empirical investigation of cross-task neural policy transfer under discrete action constraints (C1). Evaluated across 18 benchmark tasks in 6 functional domains on a 120B parameter production LLM, AMAS achieves an empirical net token reduction of $\mathbf{60.46\%}$ ($p=0.0031$), slashing inference costs from \$0.31662 to \$0.12517 (corroborating theoretical DAG token models of 36.19\%). It prunes 50.0\% of available agents relative to a 6-agent master roster (40.0\% relative to a 5-agent baseline) and compresses the critical path from 5.00 to 2.89 hops ($-42.2\%$, $t=19.00, p = 6.91 \times 10^{-13}$), while preserving 100\% capability coverage on 16 of 18 tasks (macro mean 94.44\%). In Episode~2, 72.2\% of queries converge to sub-millisecond Q-table lookups with zero token search overhead."""

new_abstract = r"""In this paper, we propose AMAS (Adaptive Multi-Agent System), an end-to-end framework that dynamically optimizes multi-agent execution topologies per query through reinforcement learning and validated state-graph compilation. Formalizing adaptation as a Markov Decision Process (MDP), we present two demarcated contributions: (1) an operational tabular Q-learning controller paired with a structural invariant validator dynamically compiling topologies into executable DAGs in $<15$\,ms (C2); and (2) an empirical investigation of cross-task neural policy transfer under discrete action constraints (C1). Evaluated across 18 benchmark tasks in 6 domains on a 120B production LLM, AMAS achieves an empirical token reduction of $\mathbf{60.46\%}$ ($p=0.0031$), cutting inference costs from \$0.31662 to \$0.12517 (corroborating theoretical DAG token models of 36.19\%). It prunes 50.0\% of available agents relative to a 6-agent master roster (40.0\% relative to a 5-agent baseline) and compresses the critical path from 5.00 to 2.89 hops ($-42.2\%$, $t=19.00, p = 6.91 \times 10^{-13}$). Crucially, AMAS achieves complete ($100\%$) capability coverage on 16 of 18 benchmark tasks and $94.44\% \pm 17.15\%$ macro capability coverage overall (with minor drops isolated to early exploration in composite queries). In Episode~2, 72.2\% of queries resolve via sub-millisecond Q-table lookups with zero token search overhead."""

assert old_abstract in text, "old_abstract not found"
text = text.replace(old_abstract, new_abstract)

# =========================================================================
# 2. INTRODUCTION TIGHTENING
# =========================================================================
old_intro_penalties = r"""This structural over-provisioning introduces three severe operational penalties:
\begin{enumerate}[leftmargin=*]
    \item \textbf{Token Consumption and Financial Cost:} Invoking specialized agents with irrelevant contexts yields superfluous context windows and redundant conversational exchanges, drastically multiplying token usage and API expenditures.
    \item \textbf{Sequential Latency and Bottlenecks:} Dense pipelines require sequential turn-taking. Latency scales with pipeline depth rather than intrinsic query complexity~\cite{Li2025AdaptiveGraphPruning}.
    \item \textbf{Noise Propagation and Hallucination:} Superfluous intermediate agent interactions pollute the global context, increasing epistemic drift and hallucination risks~\cite{xiao2026agentdropout}.
\end{enumerate}"""

new_intro_penalties = r"""This structural over-provisioning introduces three critical operational penalties: (1)~\textbf{Token & Cost Inflation:} Invoking irrelevant specialists consumes superfluous context windows, drastically multiplying token usage and API costs; (2)~\textbf{Sequential Latency:} Dense pipelines enforce sequential turn-taking where latency scales with pipeline depth rather than intrinsic task complexity~\cite{Li2025AdaptiveGraphPruning}; and (3)~\textbf{Context Pollution:} Superfluous intermediate interactions dilute global context, elevating epistemic drift and hallucination risks~\cite{xiao2026agentdropout}."""

assert old_intro_penalties in text, "old_intro_penalties not found"
text = text.replace(old_intro_penalties, new_intro_penalties)

old_intro_contributions = r"""This work makes two clearly demarcated contributions:
\begin{enumerate}[leftmargin=*]
    \item \textbf{Contribution 1 (C1: Neural Cross-Task Transfer Analysis):} We formalize and evaluate a task-conditioned neural policy trained across synthetic task environments. Using continuous state and task embeddings, the policy predicts architecture mutations. We evaluate cross-task transfer on held-out tasks, providing an honest analysis of discrete action space limitations and stability trade-offs. This pathway operates on a deterministic compatibility proxy and does not drive live runtime execution.
    \item \textbf{Contribution 2 (C2: Live Runtime Q-Learning Pipeline):} We formulate and implement an operational, end-to-end adaptive runtime system that dynamically optimizes multi-agent execution topologies per query using task-aware tabular Q-learning. Candidate graph mutations are filtered through a structural validation manager enforcing graph invariants. Validated logical architectures are compiled into executable stateful computation graphs with sub-15\,ms runtime overhead ($11.78 \pm 3.64$\,ms), supporting conditional routing and quality rework.
\end{enumerate}"""

new_intro_contributions = r"""This work provides two clearly demarcated contributions:
\begin{enumerate}[leftmargin=*]
    \item \textbf{Contribution 1 (C1: Neural Transfer Analysis):} We formalize and evaluate an offline task-conditioned neural policy using continuous state/task embeddings to predict mutations across synthetic environments. Evaluated on held-out tasks under a deterministic compatibility proxy, this pathway rigorously documents discrete action space boundaries ($\Delta Q = 0.0000$) and summarization stability without driving live execution.
    \item \textbf{Contribution 2 (C2: Live Runtime Q-Learning Pipeline):} We implement an operational runtime system that dynamically tailors execution DAGs per query via tabular Q-learning, filtering mutations through structural invariant validation. Validated graphs are compiled into executable LangGraph stateful workflows with low adaptation overhead ($11.78 \pm 3.64$\,ms), achieving 60.46\% live token savings.
\end{enumerate}"""

assert old_intro_contributions in text, "old_intro_contributions not found"
text = text.replace(old_intro_contributions, new_intro_contributions)

# =========================================================================
# 3. RELATED WORK TIGHTENING
# =========================================================================
old_rw1 = r"""Foundational architectures such as MetaGPT~\cite{hong2024metagpt}, AutoGen~\cite{wu2023autogen}, and ChatDev~\cite{qian2024chatdev} demonstrated that structuring agent interactions via Standard Operating Procedures (SOPs) and predefined conversational roles significantly improves reasoning stability. However, these foundational systems enforce static execution pipelines: every user query activates an identical monolithic pipeline sequentially. Recently, budget-aware multi-agent systems such as BAMAS~\cite{Yang2026BAMAS}, evolving orchestration frameworks~\cite{dang2025multiagent}, and decentralized coordination in AgentNet~\cite{yang2025agentnet} have demonstrated that dynamic topology structuring substantially mitigates execution cost. Similarly, AgentDropout~\cite{xiao2026agentdropout} explored deactivating non-essential agents during multi-turn interactions, while DyLAN~\cite{liu2024dylan} introduced an unsupervised peer-rating mechanism to filter out low-contributing agents between communication layers. However, DyLAN incurs an $\mathcal{O}(N^2)$ prompt scoring overhead per communication round. In contrast, AMAS employs sub-millisecond tabular Q-learning with zero prompt evaluation overhead."""

new_rw1 = r"""Foundational systems such as MetaGPT~\cite{hong2024metagpt}, AutoGen~\cite{wu2023autogen}, and ChatDev~\cite{qian2024chatdev} established that structuring interactions through specialized roles and Standard Operating Procedures (SOPs) improves reasoning consistency. However, they enforce static pipelines where every query triggers the same monolithic chain. Recent works have explored dynamic coordination: BAMAS~\cite{Yang2026BAMAS} structured budget-aware agent sets; evolving orchestration~\cite{dang2025multiagent} and AgentNet~\cite{yang2025agentnet} explored evolutionary collaboration; AgentDropout~\cite{xiao2026agentdropout} pruned non-essential agents; and DyLAN~\cite{liu2024dylan} applied peer-ratings to deactivate inactive agents. Nonetheless, DyLAN incurs an $\mathcal{O}(N^2)$ prompt evaluation cost per round. In contrast, AMAS achieves dynamic topology selection via tabular Q-learning with sub-millisecond lookup and zero prompt deliberation overhead."""

assert old_rw1 in text, "old_rw1 not found"
text = text.replace(old_rw1, new_rw1)

old_rw2 = r"""Transforming multi-agent execution from rigid chains into task-adaptive computation graphs represents an active research frontier. GPTSwarm~\cite{zhuge2024gptswarm} conceptualized LLM ensembles as differentiable computation graphs, using reinforcement learning to optimize edge weights while retaining a fully connected topology. AFlow~\cite{zhang2024aflow} automated workflow synthesis by framing agentic code generation as MCTS, demonstrating that task-adapted workflows surpass static architectures. In parallel, Multi-Agent Architecture Search (MaAS)~\cite{Zhang2025MaAS} explored role and edge allocation via evolutionary supernets, while Adaptive Graph Pruning (CARD/AGP)~\cite{Li2025AdaptiveGraphPruning} employed contextual bandits to prune communication matrices, compressing critical paths by 44\% ($5.0 \to 2.8$ hops). Recent advances include G-Designer~\cite{zhang2025gdesigner} (variational graph autoencoders), AgentConductor~\cite{wang2026agentconductor} (evolutionary coding graphs), NeuralFSM~\cite{wang2026neuralfsm} (finite-state policies), Graph Diffusion Models~\cite{Jiang2026GTD}, Agent Q-Mix~\cite{jiang2026agentqmix}, and MetaAgent-X~\cite{Zhang2026MetaAgentX}. AMAS advances this literature by coupling sub-millisecond tabular Q-learning with strict topological invariant verification, compiling pruned DAGs into executable state-graphs via LangGraph~\cite{langgraph2024}."""

new_rw2 = r"""Graph-based workflow optimization is an active research frontier. GPTSwarm~\cite{zhuge2024gptswarm} optimized edge weights via policy gradients over fully connected graphs. AFlow~\cite{zhang2024aflow} synthesized workflows using MCTS, trading query-time search overhead for code quality. MaAS~\cite{Zhang2025MaAS} searched topologies via evolutionary supernets, while Adaptive Graph Pruning (CARD)~\cite{Li2025AdaptiveGraphPruning} pruned communication matrices using contextual bandits, shortening critical paths by 44\% ($5.0 \to 2.8$ hops). Other recent advances include G-Designer~\cite{zhang2025gdesigner}, AgentConductor~\cite{wang2026agentconductor}, NeuralFSM~\cite{wang2026neuralfsm}, Graph Diffusion Topologies~\cite{Jiang2026GTD}, Agent Q-Mix~\cite{jiang2026agentqmix}, and MetaAgent-X~\cite{Zhang2026MetaAgentX}. AMAS differs fundamentally by combining tabular Q-learning with structural invariant verification, compiling minimal DAGs into executable stateful graphs in $<15$\,ms without search deliberation."""

assert old_rw2 in text, "old_rw2 not found"
text = text.replace(old_rw2, new_rw2)

# =========================================================================
# 4. INSERT THE NEW MAIN AMAS SYSTEM ARCHITECTURE DIAGRAM IN SECTION IV
# =========================================================================
# Figure 2 definition
arch_fig_latex = r"""\begin{figure*}[!t]
\centering
\includegraphics[width=0.96\textwidth,height=4.6cm,keepaspectratio]{figures/amas_system_architecture.pdf}
\vspace{-1mm}
\caption{Overall AMAS System Architecture and Operational Pipeline. The framework unifies two decoupled research streams fed by a shared task-context representation: (1)~\textbf{Contribution C2 (Live Runtime Pipeline):} Incoming queries undergo context extraction by the Planner ($\mathbf{c}_t$). The Q-Learning Controller selects discrete graph mutation operators ($O_1 \dots O_5$), supported by an LLM fallback for unvisited states. Candidate mutations must satisfy structural invariant validation $\mathcal{V}(G)$ (entry/exit preservation, specialist reachability, DAG acyclicity, and capability floors); invalid mutations are rejected with penalty $P_{\mathrm{reject}}$ and looped back for re-sampling. Validated topologies are assembled by the Dynamic Graph Builder and compiled into stateful LangGraph execution workflows ($11.78 \pm 3.64$\,ms total overhead). Only active specialists execute, streaming tokens and latency telemetry into a composite reward calculator that updates tabular Q-values via closed-loop TD learning. (2)~\textbf{Contribution C1 (Offline Experimental Path):} A separate research exploration investigating cross-task transfer via continuous dual encoders and a shared MLP policy evaluated on held-out tasks under a deterministic proxy, strictly decoupled from runtime execution.}
\label{fig:system_architecture}
\vspace{-2mm}
\end{figure*}"""

# Insert Figure 2 right at the start of Section IV
pos_sec4 = text.find(r'\section{The AMAS Runtime Framework (C2)}')
assert pos_sec4 != -1, "Section IV header not found"

# Check where Section IV text starts
sec4_intro_old = r"""The operational runtime core of AMAS (Contribution C2) dynamically maps query contexts to compiled state graphs across four stages:"""

sec4_intro_new = r"""As illustrated in the end-to-end architecture schematic in Fig.~\ref{fig:system_architecture}, the operational runtime core of AMAS (Contribution C2) dynamically maps incoming queries to validated, compiled execution graphs across four modular stages:"""

assert sec4_intro_old in text, "sec4_intro_old not found"
text = text.replace(sec4_intro_old, arch_fig_latex + '\n\n' + sec4_intro_new)

# =========================================================================
# 5. FIX SECTION VII-D & DISCUSSION WORDING
# =========================================================================
# Fix "Zero-overhead compilation" -> "Low-overhead dynamic compilation"
old_viid_wording = r"""Zero-overhead compilation is critical for live deployment. As shown in Fig.~\ref{fig:runtime_overhead}, dynamic adaptation latency averages $\mathbf{11.78 \pm 3.64}$\,ms"""
new_viid_wording = r"""Low-overhead dynamic compilation is critical for live deployment. As shown in Fig.~\ref{fig:runtime_overhead}, dynamic adaptation latency averages $\mathbf{11.78 \pm 3.64}$\,ms"""
if old_viid_wording in text:
    text = text.replace(old_viid_wording, new_viid_wording)

# Fix claim of "guarantees 100% compilation safety" -> "structural validation strictly enforces defined graph invariants"
old_disc_wording = r"""By confining live routing to tabular Q-learning with structural invariant verification, AMAS guarantees sub-millisecond dispatch and 100\% compilation safety."""
new_disc_wording = r"""Confining live routing to tabular Q-learning with structural invariant verification enables sub-millisecond dispatch while strictly enforcing defined graph invariants prior to state-graph compilation."""
if old_disc_wording in text:
    text = text.replace(old_disc_wording, new_disc_wording)

# Fix Figure references:
# In main_revised, fig:pareto_frontier was Fig. 2, fig:runtime_overhead was Fig. 3.
# Now with fig:system_architecture as Fig. 2:
# Pareto is Fig. 3, Runtime Overhead is Fig. 4.
# LaTeX \ref{fig:pareto_frontier} automatically handles numbers, but check hardcoded "Fig. 2" or "Fig. 3" in text!
text = text.replace('Fig.~\\ref{fig:pareto_frontier} illustrates', 'Fig.~\\ref{fig:pareto_frontier} illustrates')
text = text.replace('As shown in Fig.~\\ref{fig:runtime_overhead}', 'As shown in Fig.~\\ref{fig:runtime_overhead}')

# =========================================================================
# 6. WRITE OUT TO main_final.tex
# =========================================================================
with open('main_final.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print("Created main_final.tex successfully!")
