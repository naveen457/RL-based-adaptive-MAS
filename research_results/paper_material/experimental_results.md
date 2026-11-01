# Experimental Results Section Draft (Academic Paper Manuscript)

## IV. Experimental Results

We evaluate our proposed adaptive multi-agent system framework across two experimental pathways: (1) cross-task neural policy adaptation on held-out tasks, and (2) runtime dynamic multi-agent architecture optimization evaluated on an 18-task cross-domain benchmark suite.

### A. Contribution 1: Neural Cross-Task Policy Adaptation

To evaluate the generalization capacity of task-conditioned neural adaptation, we train a shared policy net across a training distribution $\mathcal{T}_{\text{train}}$ and evaluate fast adaptation on held-out tasks $\tau_{\text{heldout}} \in \mathcal{T}_{\text{test}}$. We benchmark the shared meta-policy against three baselines: (i) a fresh scratch policy initialized with random weights, (ii) a deterministic heuristic baseline, and (iii) a random uniform policy. Each condition is evaluated across 5 random seeds (`[42, 101, 202, 303, 404]`) over 2 fine-tuning adaptation steps (Table \ref{tab:neural_adaptation_baselines}).

#### 1) Adaptation Trajectories and Action Space Constraints
As depicted in Fig. \ref{fig:neural_adaptation_trajectory}(a), on research-oriented tasks (`task-research-002` and `task-research-001`), the initial perturbation sets the `researcher` agent's role to `"analysis"`, dropping capability coverage from $1.0000$ to $0.6000$ (and $0.0000$ respectively). Across 2 fine-tuning steps, the shared meta-policy, fresh policy, and random policy all exhibit zero adaptation gain ($\Delta = 0.0000 \pm 0.0000$). 

A rigorous audit of the discrete action mapper reveals that the environment was configured with $\text{role\_options} = [\text{"analysis"}, \text{"planning"}]$. Consequently, the capability `"research"` is structurally absent from the candidate action set $\mathcal{A}$. Among the 34 valid discrete actions, 31 leave the score at $0.6000$, and 3 destructive actions reduce the score to $0.0000$. Because all viable actions produce identical TD-targets ($y = 0.6000$), weight gradients shift the Q-value baseline but cannot break ties or overcome the structural $0.6000$ performance ceiling.

#### 2) Generalization Stability vs. Destructive Exploration
On `task-summarization-001` (Fig. \ref{fig:heldout_generalization}), the pre-trained shared meta-policy demonstrated superior stability over scratch initialization. While the fresh policy suffered destructive actions in 1 of 5 seeds ($0.4800 \pm 0.2400$, destructive rate $20.0\%$), the shared policy consistently preserved the vital synthesis agent across all seeds ($0.6000 \pm 0.0000$, destructive rate $0.0\%$).

Conversely, on `task-coding-001` (Fig. \ref{fig:neural_adaptation_trajectory}(b)), where the initial topology already satisfied requirements ($1.0000$), unconstrained $\epsilon$-greedy exploratory updates caused the shared policy to sample destructive agent deactivations, degrading performance to $0.6000$ ($\Delta = -0.4000$). This finding underscores the necessity of capability-preserving action masks during neural fine-tuning.

---

### B. Contribution 2: Dynamic Multi-Agent Architecture Adaptation

#### 1) Topological Compression and Capability Preservation
Table \ref{tab:benchmark_convergence} details convergence metrics across the 18 benchmark tasks in Episode 2. The dynamic architecture adapter pruned the full-team 5-agent baseline down to a mean of $2.94 \pm 0.73$ active agents, achieving an average Agent Pruning Rate (APR) of $40.74\% \pm 12.10\%$. 

As illustrated in Fig. \ref{fig:capability_vs_agents}, 100\% capability coverage is preserved across 15 of 18 tasks with only 2 to 4 active agents. For greeting tasks (`greet-001` to `greet-003`), the system pruned 3 redundant specialists ($66.7\%$ APR), deploying a minimal planner-finalizer topology ($L_{\text{crit}} = 2$). For math, coding, research, and web search domains, the policy dynamically retained only domain specialists, achieving a 33.3% to 50.0% APR without coverage loss.

#### 2) Token Consumption and Negative Savings Transparency
Across the 18 tasks in Episode 2, total baseline consumption of $63,000$ tokens was reduced to $40,200$ adapted tokens, representing a **$36.19\%$ net token reduction** ($t = 6.48, p = 4.4 \times 10^{-6}$, paired Student's t-test). Cumulative consumption across all 36 executions over 2 episodes yielded $80,400$ adapted tokens vs. $126,000$ baseline tokens (**$36.19\%$ cumulative savings**).

Crucially, our unclamped accounting preserves negative savings (Fig. \ref{fig:token_efficiency_comparison}). On complex cross-domain tasks requiring multi-agent verification (`cross_domain-001` and `cross_domain-002`), the system expanded coordination topologies, resulting in negative savings ($-5.7\%$ on chess platform, $-24.3\%$ on smart contract audit). This demonstrates that our framework optimizes architectures according to task demand rather than enforcing naive agent reduction.

#### 3) Critical Path Latency Compression
As shown in Fig. \ref{fig:critical_path_compression}, serial static execution requires 5.0 sequential hops entry-to-exit. By activating only required agents and optimizing communication edges, AMAS compressed the critical path length to a mean of $2.89 \pm 0.47$ hops—a **$42.2\%$ reduction in latency proxy overhead**.

#### 4) Pareto Net Utility Optimization
Fig. \ref{fig:architecture_utility_sweetspot} demonstrates that topologies operating within a structural complexity range of 5.0 to 7.0 achieve peak Pareto net utilities ($U_{\text{pareto}} \in [0.770, 0.930]$). In contrast, the static baseline incurs an over-engineering penalty ($U_{\text{pareto}} \approx 0.500$). The mean net utility of the adapted topologies reached $0.798 \pm 0.207$, representing a $+59.6\%$ improvement over the static baseline.

---

### C. Dynamic Graph Compilation Overhead

Dynamic multi-agent reconfiguration is frequently criticized for incurring excessive compilation latency. We measured graph construction time ($T_{\text{build}}$) and LangGraph compilation time ($T_{\text{comp}}$) with microsecond precision across all tasks (Fig. \ref{fig:dynamic_graph_overhead}). 

$T_{\text{build}}$ averaged $0.052 \pm 0.021$ ms, while $T_{\text{comp}}$ averaged $0.029 \pm 0.012$ ms. The total adaptation pipeline overhead averaged **$0.081$ ms**, introducing strictly sub-millisecond overhead. Build and compilation success rates were $100.0\%$ (36/36 runs), with zero cyclic routing violations.

---

### D. Reinforcement Learning Behavior

As shown in Fig. \ref{fig:rl_action_decomposition}(a), by Episode 2, the tabular Q-policy achieved an **Active RL Policy Hit Rate of $72.22\%$** (13 / 18 tasks), providing instant zero-LLM adaptations ($0$ search tokens, $<1$ ms latency). Only 5 tasks ($27.78\%$) required fallback to the multi-agent LLM reasoning adapter due to novel state transitions. Fig. \ref{fig:rl_action_decomposition}(b) depicts the distribution of accepted graph mutations, confirming balanced topology tuning across agent activations, role adjustments, and communication edge additions.

