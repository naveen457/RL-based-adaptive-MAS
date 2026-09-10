"""
RL policy and controller layer for the adaptive MAS research project.

This layer provides:
* A policy interface that selects actions from observations and valid action IDs.
* A deterministic baseline policy for testing.
* An RL controller/environment runner that executes episodes.
* Trajectory/transition representations for recording experience.

This step deliberately does NOT implement:
* RL algorithm, training loop, or Meta-RL controller.
* PPO, DQN, policy gradients, or neural networks.
* Dynamic LangGraph rebuilding.
* Persistent memory.
* Any LLM calls.
"""
