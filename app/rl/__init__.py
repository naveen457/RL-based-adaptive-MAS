"""
RL policy, controller, and multi-task/meta-training layer for the adaptive MAS research project.

This layer provides:
* A policy interface that selects actions from observations and valid action IDs.
* A deterministic baseline policy for testing.
* An RL controller/environment runner that executes episodes.
* Trajectory/transition representations for recording experience.
* A baseline multi-task / meta-training foundation for task-aware Q-learning.

This layer deliberately does NOT implement:
* PPO, DQN, policy gradients, or neural networks.
* Gradient-based meta-learning (MAML, Reptile, etc.).
* Dynamic LangGraph rebuilding.
* Persistent memory.
* Any LLM calls.
"""
