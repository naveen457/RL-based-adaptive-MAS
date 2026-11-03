path = 'Research/main.tex'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# -----------------------------------------------------------------
# 1. INTRODUCTION PARAGRAPH 1: Cite foundational MAS first [1, 2, 3]
# -----------------------------------------------------------------
old_p1 = r"To overcome the reasoning and context limitations of single-agent prompting, multi-agent systems (MAS) have emerged as the dominant architecture for handling sophisticated decomposition tasks~\cite{hong2024metagpt,lee2026survey,cao2025survey}. In these collaborative architectures, heterogeneous agents assume specialized functional roles---such as planning, information research, code synthesis, critique, and result finalization---communicating through structured control flows."

new_p1 = r"To overcome the reasoning and context limitations of single-agent prompting, collaborative multi-agent systems (MAS) have emerged as the dominant paradigm for handling complex decomposition tasks, establishing standard operating procedures across specialized roles~\cite{hong2024metagpt,wu2023autogen,qian2024chatdev}. In these collaborative architectures, heterogeneous agents assume specialized functional roles---such as planning, information research, code synthesis, critique, and result finalization---communicating through structured execution topologies."

text = text.replace(old_p1, new_p1)

# -----------------------------------------------------------------
# 2. INTRODUCTION PARAGRAPH 2: Cite direct dynamic baselines [4, 5, 6, 7, 8, 9]
# -----------------------------------------------------------------
old_intro_dyn = r"""To overcome these challenges, recent research has explored dynamic topology optimization, communication graph pruning, and adaptive agent orchestration~\cite{Li2025AdaptiveGraphPruning,Jiang2026GTD,Zhang2025MaAS,wang2026neuralfsm}. As illustrated in Fig.~\ref{fig:dynamic_topologies}, rigid multi-agent pipelines sequentially route queries through unneeded agents (such as Coder and Critic for a conceptual research inquiry), causing substantial token expenditure and sequential latency. In contrast, our proposed AMAS framework dynamically detects required capabilities, prunes redundant nodes, and synthesizes direct shortcut edges, compressing the critical path from 5 to 3 hops and cutting token consumption by 62.3\% with a 2.60$\times$ turnaround speedup. However, existing dynamic approaches frequently incur substantial search overhead (e.g., multi-round deliberation in DyLAN~\cite{liu2024dylan} or evolutionary population rollouts in MaAS~\cite{Zhang2025MaAS}), which can consume thousands of tokens before task execution commences."""

new_intro_dyn = r"""To overcome these challenges, recent research has pioneered dynamic topology optimization, communication graph pruning, and adaptive agent selection~\cite{liu2024dylan,zhuge2024gptswarm,zhang2024aflow,Zhang2025MaAS,Li2025AdaptiveGraphPruning,xiao2026agentdropout}. As illustrated in Fig.~\ref{fig:dynamic_topologies}, rigid multi-agent pipelines sequentially route queries through unneeded agents (such as Coder and Critic for a conceptual research inquiry), causing substantial token expenditure and sequential latency. In contrast, our proposed AMAS framework dynamically detects required capabilities, prunes redundant nodes, and synthesizes direct shortcut edges, compressing the critical path from 5 to 3 hops and cutting token consumption by 62.3\% with a 2.60$\times$ turnaround speedup under production turnaround times ($1.78$\,s vs $4.62$\,s). However, existing dynamic approaches frequently incur substantial search overhead: DyLAN~\cite{liu2024dylan} requires iterative peer-scoring rounds, AFlow~\cite{zhang2024aflow} relies on 15--30 Monte Carlo tree search rollouts, and MaAS~\cite{Zhang2025MaAS} demands offline evolutionary supernet generations, all of which consume excessive tokens or prevent instant online routing."""

text = text.replace(old_intro_dyn, new_intro_dyn)

# -----------------------------------------------------------------
# 3. SECTION II: RELATED WORK (STRUCTURED IN PRIORITY ORDER)
# -----------------------------------------------------------------
old_sec2_start = r"""\section{Related Work}
\label{sec:related_work}

\subsection{Static vs. Dynamic LLM-Based Multi-Agent Systems}
Multi-agent collaboration frameworks coordinate ensembles of specialized LLMs to tackle sophisticated problem decomposition. Foundational architectures such as MetaGPT~\cite{hong2024metagpt}, AutoGen~\cite{wu2023autogen}, and ChatDev~\cite{qian2024chatdev} demonstrated that structuring interactions via Standard Operating Procedures (SOPs) and predefined conversational roles significantly reduces reasoning ambiguity. However, these prevailing systems operate under strict \textit{structural rigidity}: every incoming user query activates the identical monolithic multi-agent pipeline sequentially, regardless of problem complexity. Recent surveys emphasize that static multi-agent pipelines incur severe computational redundancy, inflated inference costs, and latency bottlenecks~\cite{lee2026survey,cao2025survey,sun2024llmmarl}.

To mitigate agent redundancy, AgentDropout~\cite{xiao2026agentdropout} proposed deactivating non-essential agents during multi-turn interactions. Similarly, DyLAN~\cite{liu2024dylan} introduced an unsupervised peer-rating mechanism to filter out low-contributing agents between communication layers, reporting 25--40\% agent pruning and 15--25\% token reductions. However, DyLAN requires all agents to iteratively score their peers at each round, generating an $\mathcal{O}(N^2)$ LLM prompt overhead that negates token efficiency on simple or fast queries."""

new_sec2_start = r"""\section{Related Work}
\label{sec:related_work}

\subsection{Static vs. Dynamic Multi-Agent Architectures}
Multi-agent collaboration frameworks coordinate ensembles of specialized LLMs to tackle sophisticated problem decomposition. Foundational architectures such as MetaGPT~\cite{hong2024metagpt}, AutoGen~\cite{wu2023autogen}, and ChatDev~\cite{qian2024chatdev} demonstrated that structuring agent interactions via Standard Operating Procedures (SOPs) and predefined conversational roles significantly improves reasoning stability. However, these foundational systems operate under strict \textit{structural rigidity}: every user query activates an identical monolithic pipeline sequentially. Recently, budget-aware multi-agent systems such as BAMAS~\cite{Yang2026BAMAS} and evolving orchestration frameworks~\cite{dang2025multiagent} have demonstrated that dynamic topology structuring substantially mitigates execution cost. Similarly, AgentDropout~\cite{xiao2026agentdropout} explored deactivating non-essential agents during multi-turn interactions, while DyLAN~\cite{liu2024dylan} introduced an unsupervised peer-rating mechanism to filter out low-contributing agents between communication layers. However, DyLAN incurs an $\mathcal{O}(N^2)$ prompt scoring overhead per communication round, undermining efficiency on straightforward queries."""

text = text.replace(old_sec2_start, new_sec2_start)

# Section II-B: Graph Architecture Optimization, Pruning, and Evolution
old_sec2_b = r"""\subsection{Graph Architecture Search, Optimization, and Pruning}
Transforming multi-agent execution from rigid chains into task-adaptive computation graphs represents an active research frontier. GPTSwarm~\cite{zhuge2024gptswarm} conceptualized LLM ensembles as differentiable computation graphs, using reinforcement learning and policy gradients to optimize edge routing weights. However, GPTSwarm retains a fully connected topological substrate where all agents remain active, focusing on task accuracy (+12.4\% on GAIA) rather than sequential critical-path compression or agent pruning.

AFlow~\cite{zhang2024aflow} investigated automated workflow synthesis by framing agentic code generation as Monte Carlo Tree Search (MCTS), demonstrating that task-adapted workflows surpass static architectures (+5.7\% accuracy). Nevertheless, AFlow requires 15--30 exploratory rollout evaluations per problem, resulting in substantial search latency and token expenditure that prevent instant online query execution. Multi-Agent Architecture Search (MaAS)~\cite{Zhang2025MaAS} introduced agentic supernets to explore role assignments and edge topologies using multi-objective evolutionary algorithms, reporting 18--32\% token savings. Yet, evolutionary search necessitates multi-generation evaluations offline, yielding static architectures per task class rather than dynamic per-query adaptation.

In parallel, Adaptive Graph Pruning (CARD/AGP)~\cite{Li2025AdaptiveGraphPruning} employed contextual bandits to dynamically prune agent communication matrices, compressing critical execution paths by 44\% ($5.0 \to 2.8$ hops). Our proposed AMAS architecture directly connects with and extends these foundations. Rather than incurring multi-round peer deliberation (DyLAN), iterative MCTS rollouts (AFlow), or offline evolutionary generations (MaAS), AMAS couples instant tabular Q-learning action selection ($<1$\,ms decision time, 0 search tokens) with strict invariant validation and compiles the pruned topology directly into an executable stateful computation graph in sub-15\,ms (instantiated via LangGraph~\cite{langgraph2024}), delivering validated empirical token and latency savings under live production LLM execution."""

new_sec2_b = r"""\subsection{Graph Architecture Search, Optimization, and Pruning}
Transforming multi-agent execution from rigid chains into task-adaptive computation graphs represents an active research frontier. GPTSwarm~\cite{zhuge2024gptswarm} conceptualized LLM ensembles as differentiable computation graphs, using reinforcement learning and policy gradients to optimize edge routing weights while retaining a fully connected topology. AFlow~\cite{zhang2024aflow} automated workflow synthesis by framing agentic code generation as Monte Carlo Tree Search (MCTS), demonstrating that task-adapted workflows surpass static architectures. In parallel, Multi-Agent Architecture Search (MaAS)~\cite{Zhang2025MaAS} introduced agentic supernets to explore role assignments and edge topologies using multi-objective evolutionary algorithms, while Adaptive Graph Pruning (CARD/AGP)~\cite{Li2025AdaptiveGraphPruning} employed contextual bandits to prune communication matrices, compressing critical paths by 44\% ($5.0 \to 2.8$ hops).

Recent advancements have further investigated dynamic topology generation and structural adaptation. AgentConductor~\cite{wang2026agentconductor} explored topology evolution for competition-level code generation. NeuralFSM~\cite{wang2026neuralfsm} introduced adaptive coordination via learned finite-state execution policies, while Jiang et al.~\cite{Jiang2026GTD} investigated graph diffusion models to dynamically generate communication topologies. Concurrently, Agent Q-Mix~\cite{jiang2026agentqmix} applied multi-agent value decomposition to action selection, and MetaAgent-X~\cite{Zhang2026MetaAgentX} developed automatic multi-agent systems via end-to-end RL. AMAS directly connects with and advances these works by coupling sub-millisecond tabular Q-learning with strict topological invariant verification, compiling pruned DAGs into executable state-graphs (instantiated via LangGraph~\cite{langgraph2024}) with sub-15\,ms overhead."""

text = text.replace(old_sec2_b, new_sec2_b)

# Section II-C: LLM Multi-Agent Reinforcement Learning & Resilience
# and Section II-D: Meta-RL & Transfer
old_sec2_c = r"""\subsection{Meta-Reinforcement Learning and Cross-Task Transfer}
Meta-RL investigates learning adaptation strategies across task distributions~\cite{Kashikar2025MetaRL,Huang2024AdaptiveMARL}. Contextual meta-MARL has been explored to coordinate heterogeneous agents under varying team attributes~\cite{Huang2024AdaptiveMARL}. Fu et al.~\cite{Fu2026MOFA} investigated mixtures of orthogonal experts for rapid policy transfer. Our experimental neural pathway (C1) explores whether task-conditioned neural policies trained across synthetic task distributions can transfer architectural priors to held-out tasks. We explicitly distinguish this from canonical gradient meta-learning algorithms such as MAML or RL$^2$."""

new_sec2_c = r"""\subsection{Reinforcement Learning for LLM Multi-Agent Collaboration}
Integrating reinforcement learning into LLM multi-agent coordination has recently emerged as a vital methodology. Liu et al.~\cite{liu2026llm} established collaborative multi-agent reinforcement learning formulations for LLMs. Zhao et al.~\cite{Zhao2026StrongerMAS} introduced Stronger-MAS, exploring multi-agent RL for collaborative LLM alignment. To address multi-agent deliberation and reasoning overhead, ReMA~\cite{Wan2025ReMA} developed meta-thinking via MARL, while Yang and Thomason~\cite{Yang2026LearningToDeliberate} investigated meta-policy collaboration for deliberation. In addition, ResMAS~\cite{Zhou2026ResMAS} formulated resilience optimization against cascading agent failures, and Rjoub~\cite{rjoub2026aamc} proposed adaptive agentic meta-controllers for intelligent SLM/LLM orchestration.

\subsection{Meta-Reinforcement Learning and Cross-Task Transfer}
Meta-RL investigates learning reusable adaptation strategies across task distributions~\cite{Huang2024AdaptiveMARL,Kashikar2025MetaRL}. Contextual meta-MARL coordinates heterogeneous agents under varying team attributes~\cite{Huang2024AdaptiveMARL}, while Kashikar~\cite{Kashikar2025MetaRL} explored emergent communication in zero-shot coordination. Fu et al.~\cite{Fu2026MOFA} investigated mixtures of orthogonal experts (MOFA) for rapid transfer, and Zhang et al.~\cite{zhang2025bilevel} formulated bi-level knowledge transfer for multi-task MARL. Self-play meta-RL has also been analyzed in decentralized multi-agent settings~\cite{Mali2026SelfPlay,li2020adaptive}. Comprehensive literature surveys~\cite{Cao2025TNNLSSurvey,sun2024llmmarl,cao2025survey,lee2026survey} underscore the urgency of addressing multi-agent token redundancy and structural rigidity. In this investigation, our experimental neural pathway (C1) examines whether task-conditioned neural policies trained across synthetic task environments can transfer architectural priors to held-out tasks."""

text = text.replace(old_sec2_c, new_sec2_c)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated main.tex with prioritized citations successfully!')

