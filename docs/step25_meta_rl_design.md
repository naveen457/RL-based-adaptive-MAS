# Step 25 — Meta-RL Method Selection and Design

**Status:** Research and design only. No advanced Meta-RL implementation yet.

**Scope:** Select and justify a modern Meta-RL formulation for *adaptive LLM-MAS architecture*, to inform Step 26 implementation.

---

## 1. Problem framing

### Research question
How can a learned meta-policy rapidly adapt the **architecture** of an LLM-based multi-agent system to an unseen task, including:

- active agents
- agent roles
- communication edges/topology

### What is being adapted
The **system architecture**, not:
- a better individual agent policy
- the next message
- a fixed communication protocol
- prompt-level agent behavior

### What we already have
- `MASArchitecture` state representation (agents, roles, active flags, edges)
- `ArchitectureAction` space (activate/deactivate, add/remove edge, change role)
- `MetaTask` distribution with task category, required capabilities, difficulty, context features
- `MetaEnvironment` / `MASArchitectureEnv` with structural reward + optional task-performance reward
- Baselines: architecture-only Q-learning, task-aware Q-learning, multi-task Q-learning, transfer/adaptation baseline, multi-seed generalization evaluation

---

## 2. Candidate methods considered

### 2.1 Gradient-based Meta-RL (MAML / Reptile / meta-gradients)
- **What is adapted:** policy parameters via inner-loop gradient steps
- **Task representation:** task context or task ID, sometimes inferred
- **Test-time adaptation:** usually requires gradient updates at adaptation time
- **Discrete actions:** possible with policy gradients, but more complex
- **Fit to architecture adaptation:** moderate; can optimize architecture policy, but inner-loop update cost and stability are concerns
- **Complexity:** high
- **Prototype suitability:** lower

### 2.2 Recurrent / in-context Meta-RL (RL^2 style)
- **What is adapted:** a recurrent policy’s hidden state effectively encodes task experience
- **Task representation:** implicit in interaction history
- **Test-time adaptation:** fast, via recurrence/memory rather than explicit gradient updates
- **Discrete actions:** natural fit for discrete policy outputs
- **Fit to architecture adaptation:** strong; architecture adaptation is sequential and partially observable across steps, so recurrence can accumulate evidence about the task from the reward/structure trajectory
- **Complexity:** moderate
- **Prototype suitability:** high

### 2.3 Contextual / latent-variable Meta-RL (PEARL style)
- **What is adapted:** inference over latent task variables that condition the policy
- **Task representation:** learned latent posterior from support data
- **Test-time adaptation:** fast inference of latent context from a small number of trajectories
- **Discrete actions:** possible
- **Fit to architecture adaptation:** strong conceptually because tasks differ mainly by required capabilities/structure; a latent code can capture “what kind of architecture this task needs”
- **Complexity:** moderate-to-high
- **Prototype suitability:** moderate

### 2.4 Task-conditioned / context-augmented RL
- **What is adapted:** a policy conditioned on explicit task features/context
- **Task representation:** explicit task context vector/dict
- **Test-time adaptation:** no gradient update needed if context is provided; can also be combined with fast context inference
- **Discrete actions:** natural
- **Fit to architecture adaptation:** strong and simple; our `MetaTask` already provides explicit context features
- **Complexity:** low-to-moderate
- **Prototype suitability:** very high

### 2.5 Hypernetwork / meta-linear adapter approaches
- **What is adapted:** policy weights generated from task embedding
- **Task representation:** task embedding
- **Test-time adaptation:** fast conditioning, sometimes with adaptation
- **Discrete actions:** possible
- **Fit:** moderate; more common in continuous control or sequence models
- **Prototype suitability:** moderate

### 2.6 Meta-learning for multi-agent coordination / decomposing coordination
- **What is adapted:** coordination policies, communication structure, or role assignment strategies
- **Task representation:** task/vector or learned embedding
- **Test-time adaptation:** varies
- **Fit:** relevant direction, but much of the literature targets message/policy-level coordination rather than explicit architecture topology adaptation
- **Prototype suitability:** moderate; may inform evaluation design more than core method

### 2.7 LLM-based routing / agent selection / architecture generation
- **What is adapted:** LLM decides agents, roles, topology, or workflow structure
- **Task representation:** natural language task description + context
- **Test-time adaptation:** zero/few-shot via prompting or lightweight fine-tuning
- **Discrete actions:** natural
- **Fit:** highly relevant to LLM-MAS, but this project is explicitly scoped around *learned* Meta-RL, not LLM-as-meta-controller
- **Prototype suitability:** high for applied systems, lower for the Meta-RL research angle here

---

## 3. Comparison summary

Using the requested 10 criteria:

| Method | Adapted | Task representation | Test-time gradient? | Discrete actions | Works with current state | Fast unseen adaptation | Computational complexity | Implementation complexity | Capstone suitability | Clarity vs baselines |
|---|---|---|---|---|---|---|---|---|---|---|
| MAML/Reptile/meta-gradients | policy params | task context/id | Usually yes | Possible, harder | Yes | Yes, but at cost | High | High | Lower | Good for gradient-based story |
| RL^2 recurrent Meta-RL | recurrent policy state | implicit in history | No explicit gradient needed | Natural | Yes | Very fast in-context | Moderate | Moderate | High | Strong; clear improvement story |
| PEARL-style latent context | latent task variable + policy | learned latent posterior | Fast inference | Possible | Yes | Fast via latent inference | Moderate-high | Moderate-high | Moderate | Strong; clean separation of task inference and control |
| Task-conditioned/context RL | policy conditioned on task context | explicit task features | No (if context given) | Natural | Yes (we already have task context) | Fast if context provided | Low-moderate | Low-moderate | Very high | Very clear; builds directly on Step 17/21 |
| Hypernetwork/meta-linear | generated policy weights | task embedding | Often fast conditioning | Possible | Yes | Fast | Moderate | Moderate | Moderate | Moderate |
| Multi-agent coordination Meta-RL | coordination/routing/roles | task/vector/embedding | Varies | Varies | Partial | Varies | Moderate-high | High | Moderate | Relevant, but less aligned to explicit architecture topology |
| LLM routing/architecture generation | LLM output | natural language | No gradient needed usually | Natural | Yes (text + structured context) | Fast | Variable | Moderate | High (systems), lower for Meta-RL research | High applied relevance, but different research framing |

---

## 4. Selected method

**Primary selected direction:** **Task-conditioned Meta-RL with a learned context/latent task representation**, with a strong secondary emphasis on **recurrent/in-context adaptation** for fast test-time behavior.

In plain terms:
- Learn a **meta-policy** that maps `(architecture state, task context/latent task representation)` to an `ArchitectureAction`.
- During **meta-training**, expose the policy to many tasks from a task distribution and train it to adapt its behavior across tasks.
- At **test time**, for an unseen task, either:
  - use explicit task context if available, and/or
  - infer a latent task representation from a small amount of recent interaction, then continue acting rapidly.

This direction is chosen because it directly matches the project’s existing infrastructure and research question:

1. **It adapts architecture, not agent internals.**
   - Action space is already `ArchitectureAction`.
   - State already captures active agents, roles, and topology.

2. **It uses explicit task information already present in the project.**
   - `MetaTask` category, required capabilities, difficulty, and context features are available.
   - This makes the method explainable and testable, which matters for a student/research prototype.

3. **It supports rapid adaptation to unseen tasks.**
   - A task-conditioned policy can specialize behavior per task class.
   - A latent/in-context extension allows fast adaptation even when task context must be inferred from interaction.

4. **It is compatible with discrete architecture actions.**
   - The policy can output a distribution over discrete architecture actions.

5. **It is easier to compare against existing baselines.**
   - We can compare:
     - architecture-only Q-learning
     - task-aware Q-learning
     - transfer/shared Q-table baseline
     - multi-seed generalization
     - against a learned task-conditioned/meta policy

6. **It is practical for a capstone/research prototype.**
   - It avoids requiring test-time gradient updates if we use explicit task conditioning.
   - It can be extended later with latent inference or recurrence without changing the whole formulation.

---

## 5. Why it fits “Meta-RL for adaptive LLM-MAS architecture”

The key distinction is that the **policy’s decision target is the MAS architecture**, not the agents’ internal generation behavior.

That makes this problem similar to:
- **configurable/compositional systems control**, where the controller selects structure
- **task-dependent routing/selection**, where the right subgraph of agents/roles/edges depends on the task
- **fast structural adaptation**, where a new task may need a different active subgraph, role mapping, and connectivity pattern

Task-conditioned Meta-RL fits because:
- Different tasks plausibly require different architecture configurations.
- The project already has a task distribution and task features.
- The same underlying agents can be reconfigured structurally rather than retrained per task.
- The adaptation target is a small, finite, structured action space, which is easier to prototype than full continuous control.

Recurrent/in-context elements fit because:
- Architecture adaptation proceeds step by step.
- Early transitions provide information about whether the current architecture suits the task.
- A recurrent/meta policy can accumulate that evidence and adjust subsequent architectural decisions.

---

## 6. Key papers / references

Foundational and representative references to consult before implementation:

- **MAML:** Finn, Abbeel, and Levine, *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks*, ICML 2017.
- **RL^2 / learning to reinforce:** Stadie, Rajeswaran, and Abbeel, *Some Theoretical Observations on Model-Agnostic Meta-Learning*, and related In-Context Meta-RL work; see also Rakelly et al. for latent context contrasts.
- **PEARL:** Rakelly, Zhou, Finn, Jia, and Levine, *Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables*, ICML 2019.
- **Task-conditioned / context-based Meta-RL:**_prior work on conditioning policies on task embeddings/context variables; broadly represented in Meta-RL surveys and contextual RL literature.
- **Meta-RL surveys:** recent survey papers on Meta-Reinforcement Learning for a landscape view and terminology.
- **Multi-agent / coordination meta-learning:** literature on learning to coordinate, learn communication structure, or learn role assignment where available; used mainly for motivation and evaluation framing.
- **Architecture/topology adaptation for multi-agent systems:** related work on dynamic software architectures, learnable communication structures, and adaptive MAS design.

Use these as a starting point. The exact citation set for Step 26 should be finalized during implementation readiness, with preference for primary sources and strong venues.

---

## 7. Proposed meta-learning formulation

### Overview
We formulate Meta-RL for adaptive MAS architecture as learning a policy that can **rapidly select architecture actions appropriate to a new task**, given:
- the current architecture state,
- task information (explicit context and/or inferred latent context),
- and optionally short interaction history.

### Meta-training task distribution
- Use `MetaTaskDistribution` as the meta-training distribution.
- Tasks should span multiple categories and difficulty levels.
- For stronger Meta-RL, ensure the distribution contains enough diversity that fast adaptation is meaningful: tasks should differ in required capabilities and suitable topology/role patterns.

### Support/adaptation data
- A small set of interactions on the new task used to infer task properties or condition the policy.
- Could be:
  - explicit task context from `MetaTask`, and/or
  - a short “support” trajectory of recent architecture states, actions, and rewards.

### Query/evaluation data
- The remaining episodes/steps on the same unseen task used to measure adaptation performance.
- Measured by structural/task-performance reward and task-success proxy, plus adaptation speed.

### Meta-state representation
A practical meta-state could include:
- architecture observation: active agents, roles, edges, agent count
- task context: category, required capabilities, difficulty, context features
- optional short interaction history: recent rewards, recent structural deltas, number of invalid transitions, current version
- optional latent task code if using PEARL-like inference

This leverages existing state encoding while adding task-conditioning.

### Architecture action representation
Keep the existing `ArchitectureAction` design:
- activate_agent
- deactivate_agent
- add_edge
- remove_edge
- change_role

For a learned policy, the action space should be represented as a discrete set of currently valid architecture actions, similar in spirit to the current mapper-based action space. Invalid actions should be masked or penalized.

### Reward
Use the existing reward design:
- structural reward delta from `ArchitectureEvaluator`
- optional task-performance reward delta from `TaskPerformanceEvaluator`
- combined reward with configurable task-performance weight
- invalid transition penalty

This keeps Step 21 integration and avoids inventing a new reward scheme.

### Adaptation procedure
Two compatible variants:

1. **Explicit task-conditioned meta-policy**
   - At test time, condition the policy on the new task’s context and act.
   - No gradient update required.

2. **Latent/in-context Meta-RL extension**
   - Infer a latent task representation from a small support segment.
   - Condition the policy on that latent representation.
   - Continue acting, possibly updating the latent representation as more evidence arrives.

For Step 26, start with the explicit task-conditioned variant and treat latent/in-context adaptation as the clearest upgrade path.

---

## 8. Train/test protocol

### Meta-training
- Sample many tasks from the distribution.
- For each task, run a short interaction/episode sequence.
- Train the meta-policy to perform well across tasks, either:
  - by conditioning on task context, or
  - by learning to infer and use latent task context.

### Adaptation evaluation
- Hold out unseen tasks disjoint from meta-training tasks.
- For each unseen task:
  - allow a short adaptation/support phase if latent/in-context variant is used,
  - then measure adaptation performance over a fixed query budget.

### Baselines
Compare against:
- architecture-only Q-learning
- task-aware Q-learning
- multi-task shared Q-table
- transfer baseline (train on source tasks, adapt on unseen tasks)
- multi-seed generalization results

### Controls
- Same task distribution, same initial architecture, same max steps, same reward definition where applicable.
- Disjoint train/test task sets.
- Deterministic evaluations where possible.

---

## 9. Evaluation metrics

At least:
- final structural score
- final task-success score proxy
- mean reward per episode
- adaptation speed: reward vs episodes/steps
- number of valid vs invalid transitions
- final architecture version and structural changes
- per-task and aggregate results
- descriptive statistics across seeds/tasks
- sample-efficiency proxy where meaningful

Also report:
- whether adaptation uses explicit task context or inferred context
- how much support data was used
- whether results are positive, zero, or negative relative to baselines

Do not claim superiority unless the experiment shows it.

---

## 10. Expected research contribution

A plausible contribution is:
- a practical Meta-RL formulation for **adaptive architecture control in LLM-based MAS**,
- with a concrete task distribution and architecture action space,
- showing whether task-conditioned/meta policies can adapt architecture faster or more effectively than non-meta baselines on unseen tasks.

This is most credible if the contribution is framed as:
- a structured Meta-RL problem definition for architecture adaptation,
- plus an empirical comparison under controlled conditions,
- rather than a claim of a universal Meta-RL solution.

---

## 11. Likely limitations

- The current reward is structural/task-performance proxy, not real LLM execution outcomes.
- Task distribution is small and synthetic for now.
- Discrete architecture action space may limit the expressiveness of certain Meta-RL methods.
- Meta-RL can be sensitive to hyperparameters and task diversity.
- Fast adaptation claims require strong experimental controls and multiple seeds/tasks.
- Latent/in-context approaches add implementation and debugging complexity.
- The gap between architecture compatibility proxies and real LLM-MAS performance remains an important caveat.

---

## 12. Decision for Step 26

Step 25 chooses a **task-conditioned Meta-RL approach**, with latent/in-context adaptation as the preferred extension path, as the most appropriate modern Meta-RL direction for this project.

Step 26 should therefore:
- implement a learned task-conditioned architecture policy,
- integrate it with existing `MetaTask` context and architecture state,
- keep the existing discrete architecture action space and reward design,
- and compare against the existing baselines under controlled meta-train/test splits,
- without prematurely claiming full Meta-RL results.

---

## 13. What Step 25 does NOT do

- Implement any Meta-RL algorithm.
- Introduce PPO, DQN, neural networks, MAML, Reptile, meta-gradients, or learned embeddings in code.
- Add LLM/API calls.
- Add persistent memory.
- Modify `app/graph/workflow.py` or `app/agents/*`.
- Modify existing RL implementations unless necessary for documentation.

---

*End of Step 25 design document.*
