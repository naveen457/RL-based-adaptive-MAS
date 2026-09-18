"""Unit tests for Active RL-Driven Architecture Action Selection (Step 2).

Tests that:
1. QLearningPolicy actively selects optimal architecture actions for learned states.
2. High-confidence states adapt instantly with is_instant=True and zero LLM calls.
3. Unseen states or exploratory turns fall back to the LLM adapter.
4. Action IDs map cleanly to valid ArchitectureAction objects.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from app.agents.planner import PlannerOutput
from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.manager import ArchitectureManager
from app.architecture.models import MASArchitecture
from app.rl.action_space import ArchitectureActionMapper
from app.rl.active_selector import RLArchitectureSelector, RLSelectionResult
from app.rl.q_learning import QLearningPolicy
from app.rl.state import ArchitectureStateEncoder


class DummyLLMAdapter:
    """Mock LLM adapter tracking whether it was invoked."""

    def __init__(self) -> None:
        self.invoked_count = 0

    def adapt_from_planner_output(self, planner_output: PlannerOutput) -> Any:
        self.invoked_count += 1

        class DummyResult:
            planner_output = {"task": "dummy"}
            architecture_decision = {
                "decision": "apply_actions",
                "reasoning": "LLM recommended deactivating coder",
                "actions": [{"action_type": "deactivate_agent", "agent_id": "coder"}],
            }
            parsed_actions = [{"action_type": "deactivate_agent", "agent_id": "coder"}]
            valid_actions = [{"action_type": "deactivate_agent", "agent_id": "coder"}]
            rejected_actions = []

        return DummyResult()


def test_active_rl_exploits_learned_state_instantly() -> None:
    """When a state has learned positive Q-values, RL policy selects it without LLM calls."""
    policy = QLearningPolicy(task_aware=True, epsilon=0.0)  # Pure exploitation
    selector = RLArchitectureSelector(policy=policy, epsilon=0.0, min_q_threshold=0.0)

    manager = ArchitectureManager.create_default_architecture()
    arch = manager.to_architecture_model()
    plan = PlannerOutput(
        task_understanding="Research task",
        required_capabilities=["research"],
        requires_research=True,
    )

    # Encode state
    state_key = selector.get_state_key(arch, plan)
    mapper = ArchitectureActionMapper.from_manager(manager)

    # Find the action ID for deactivating coder
    target_action = ArchitectureAction(action_type=ActionType.DEACTIVATE_AGENT, agent_id="coder")
    target_id = mapper.encode(target_action)

    # Pre-train the policy with high reward for this action
    policy.q_table.set(state_key, target_id, 2.5)

    llm_adapter = DummyLLMAdapter()
    result = selector.select(arch, plan, llm_adapter=llm_adapter)

    # Verify active RL selection
    assert result.decision_source == "rl_policy"
    assert result.is_instant is True
    assert result.confidence == 2.5
    assert len(result.valid_actions) == 1
    assert result.valid_actions[0]["agent_id"] == "coder"
    assert result.valid_actions[0]["action_type"] == "deactivate_agent"
    # Verify zero LLM calls were made!
    assert llm_adapter.invoked_count == 0


def test_active_rl_falls_back_to_llm_on_unseen_state() -> None:
    """When a state is unvisited, selector consults the LLM adapter."""
    policy = QLearningPolicy(task_aware=True, epsilon=0.0)
    selector = RLArchitectureSelector(policy=policy, epsilon=0.0)

    manager = ArchitectureManager.create_default_architecture()
    arch = manager.to_architecture_model()
    plan = PlannerOutput(
        task_understanding="Unseen task",
        required_capabilities=["coding"],
        requires_coding=True,
    )

    llm_adapter = DummyLLMAdapter()
    result = selector.select(arch, plan, llm_adapter=llm_adapter)

    assert result.decision_source == "llm_adapter"
    assert result.is_instant is False
    assert llm_adapter.invoked_count == 1


def test_epsilon_exploration_forces_exploration() -> None:
    """With epsilon=1.0, selector explores rather than exploits."""
    policy = QLearningPolicy(task_aware=True, epsilon=1.0)
    selector = RLArchitectureSelector(policy=policy, epsilon=1.0)

    manager = ArchitectureManager.create_default_architecture()
    arch = manager.to_architecture_model()
    plan = PlannerOutput(
        task_understanding="Explore task",
        required_capabilities=[],
    )
    state_key = selector.get_state_key(arch, plan)
    policy.q_table.set(state_key, 0, 5.0)  # Artificially high Q

    llm_adapter = DummyLLMAdapter()
    result = selector.select(arch, plan, llm_adapter=llm_adapter)

    # Should consult adapter/explore because epsilon=1.0
    assert result.decision_source in {"exploration", "llm_adapter"}
    assert result.is_instant is False
