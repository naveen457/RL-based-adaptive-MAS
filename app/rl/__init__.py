"""
RL environment layer for the adaptive MAS research project.

This layer provides a Gymnasium-compatible environment around the existing
architecture adaptation mechanism in ``app/architecture``.

This step deliberately does NOT implement:
* RL algorithm, policy, training loop, or Meta-RL controller.
* Dynamic LangGraph rebuilding.
* Persistent memory.
* Any LLM calls.
"""
