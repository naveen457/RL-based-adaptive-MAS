# Discussion Section Draft (Academic Paper Manuscript)

## V. Discussion and Comparative Analysis

### A. Synthesis with Published Multi-Agent Architecture Research

Our findings establish important points of convergence and divergence with recent literature (Table \ref{tab:literature_comparison}):

1. **Eliminating Search Token Tax:** Frameworks such as AFlow \cite{aflow2024} and MaAS \cite{maas2024} rely on heavy rollout search mechanisms (15 to 30 rollouts per discovered graph), introducing a substantial upfront "search token tax." In contrast, AMAS employs an **active tabular Q-learning selector**: once a task-architecture state is reinforced, subsequent encounters achieve an **instant zero-LLM policy hit** ($0$ tokens overhead, $<1$ ms latency), reaching a 72.2% hit rate by Episode 2.
2. **Critical Path Latency vs. DyLAN Pruning:** DyLAN \cite{dylan2024} demonstrates that agent pruning (25–40% APR) reduces communication noise during multi-round voting. However, DyLAN maintains a rigid layered topology. Our work extends pruning into a dynamic DAG compilation setting, simultaneously achieving a **40.7% APR** and compressing the critical path from $5.0 \to 2.89$ hops (a 42.2% latency reduction matching principles in CARD \cite{card2024}).
3. **Executable Workflow Grounding:** Theoretical graph models (e.g., GPTSwarm \cite{gptswarm2024}) operate on abstract adjacency matrices. In our implementation, every adapted architecture is validated against cycle constraints and compiled into an executable **LangGraph StateGraph DAG** with sub-millisecond overhead (0.081 ms total build and compilation time).

---

### B. Trade-offs in Task-Adaptive Pruning: The Cross-Domain Challenge

Our results highlight an essential trade-off in multi-agent optimization: **aggressive pruning can lead to under-engineering on cross-domain tasks**. While greeting and math tasks benefit from aggressive pruning (66.7% and 50.0% APR), tasks with compound demands (such as `cross_domain-001`: Multiplayer Chess Platform) require concurrent coding, math, and verification capabilities. 

In Episode 2, under-engineering occurred on 2 cross-domain tasks where the Q-policy selected minimal topologies, dropping capability coverage to 33.3% and 66.7%. This observation proves that architecture optimization must incorporate strict capability constraints; simple cost-minimization objectives risk pruning specialist nodes required for edge cases.

---

### C. Honest Reflection on Neural Policy Action Space Expressiveness

The empirical evaluation of Contribution 1 revealed an invariant adaptation gain ($\Delta = 0.0000$) across 4 of 5 held-out tasks. Rather than obscuring this result, we identified the exact structural root cause: **discrete action alphabet restriction**. 

In `MetaNeuralTrainer`, the environment mapped single-step role changes restricted to `["analysis", "planning"]`. When evaluating held-out tasks requiring `"research"` or `"coding"`, the optimal corrective action was structurally absent from $\mathcal{A}$. Because all 31 non-destructive candidate actions yielded identical reward ($0.6000$), TD-target updates shifted Q-values uniformly without altering relative action rankings. Furthermore, unconstrained $\epsilon$-greedy exploration on coding tasks resulted in negative adaptation gains ($-0.4000$) due to destructive agent deactivations.

This finding provides a critical lesson for the multi-agent systems community: **neural architecture adaptation requires an action vocabulary that fully spans the capability taxonomy of the task distribution, coupled with capability-preserving action masks during gradient fine-tuning.**

---

### D. Limitations and Threats to Validity

1. **Proxy-Based Cost Metrics:** Primary benchmark token metrics were computed using validated DAG critical-path formulas ($650\text{ tok/agent} + 150\text{ tok/edge}$). While these provide a reproducible, deterministic proxy for topological efficiency, they do not capture non-deterministic prompt length variations inherent in live LLM execution.
2. **Tabular vs. Neural Separation:** In the current prototype, runtime architecture selection is governed by tabular Q-learning, while neural adaptation operates on an experimental held-out task framework. Unifying these into a hierarchical neural-tabular controller represents a compelling avenue for future work.
3. **Task Distribution Scale:** The benchmark suite comprises 18 diverse tasks across 6 categories. While representative of core functional capabilities, evaluating on larger public benchmarks (e.g., GAIA, SWE-bench) under live LLM endpoints remains essential for production validation.

