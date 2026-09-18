"""Active RL-driven architecture action selector.

Closes the RL loop by allowing QLearningPolicy to actively select or guide
architecture actions (a ~ pi(s)) using an epsilon-greedy policy controller,
triggering instant adaptations for high-confidence learned states without
requiring an LLM reasoning call.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from app.agents.planner import PlannerOutput
from app.architecture.actions import ArchitectureAction
from app.architecture.manager import ArchitectureManager
from app.architecture.models import MASArchitecture
from app.rl.action_space import ArchitectureActionMapper
from app.rl.meta_task import MetaTaskContext
from app.rl.q_learning import QLearningPolicy
from app.rl.state import ArchitectureStateEncoder


class RLSelectionResult(BaseModel):
    """Result of active RL action selection for architecture adaptation."""

    decision_source: Literal["rl_policy", "llm_adapter", "exploration", "default"]
    decision: Dict[str, Any]
    selected_actions: List[Dict[str, Any]] = Field(default_factory=list)
    valid_actions: List[Dict[str, Any]] = Field(default_factory=list)
    rejected_actions: List[Dict[str, Any]] = Field(default_factory=list)
    action_ids: List[int] = Field(default_factory=list)
    q_values: Dict[int, float] = Field(default_factory=dict)
    confidence: float = 0.0
    is_instant: bool = False
    latency_saved_ms: float = 0.0
    token_cost_saved: int = 0
    state_key: Optional[Any] = None

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class RLArchitectureSelector:
    """Active epsilon-greedy policy controller for architecture adaptation."""

    def __init__(
        self,
        policy: Optional[QLearningPolicy] = None,
        *,
        epsilon: Optional[float] = None,
        min_q_threshold: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.policy = policy or QLearningPolicy(task_aware=True, epsilon=0.1)
        self.epsilon = epsilon if epsilon is not None else self.policy.epsilon
        self.min_q_threshold = min_q_threshold
        self.rng = rng or random.Random()

    def build_task_context(self, planner_output: PlannerOutput) -> MetaTaskContext:
        """Derive a deterministic MetaTaskContext from PlannerOutput."""
        req_caps = list(planner_output.required_capabilities or [])
        tools_needed = list(planner_output.tools_needed or [])
        if "web_search" in tools_needed and "web_search" not in req_caps:
            req_caps.append("web_search")
        if getattr(planner_output, "requires_research", False) and "research" not in req_caps:
            req_caps.append("research")
        if getattr(planner_output, "requires_coding", False) and "coding" not in req_caps:
            req_caps.append("coding")
        if getattr(planner_output, "requires_verification", False) and "verification" not in req_caps:
            req_caps.append("verification")

        if "web_search" in req_caps or "tool_use" in req_caps:
            cat = "tool_use"
        elif "coding" in req_caps:
            cat = "coding"
        elif "research" in req_caps:
            cat = "research"
        else:
            cat = "general"

        complexity_map = {"trivial": 1, "low": 1, "simple": 1, "medium": 2, "high": 3, "complex": 4}
        comp_str = str(getattr(planner_output, "estimated_complexity", "medium")).lower()
        difficulty = complexity_map.get(comp_str, 2)

        return MetaTaskContext(
            task_category=cat,
            required_capabilities=sorted(req_caps),
            difficulty=difficulty,
            context_features={
                "requires_research": getattr(planner_output, "requires_research", False),
                "requires_coding": getattr(planner_output, "requires_coding", False),
                "requires_verification": getattr(planner_output, "requires_verification", False),
                "requires_tools": getattr(planner_output, "requires_tools", False),
            },
        )

    def get_state_key(
        self,
        architecture: MASArchitecture,
        planner_output: PlannerOutput,
    ) -> Tuple:
        """Compute the deterministic task-aware state key."""
        encoder = ArchitectureStateEncoder(architecture)
        obs = encoder.encode()
        task_ctx = self.build_task_context(planner_output)
        return self.policy.state_encoder.encode(obs, task_ctx)

    def select(
        self,
        current_architecture: MASArchitecture,
        planner_output: PlannerOutput,
        llm_adapter: Optional[Any] = None,
    ) -> RLSelectionResult:
        """Select architecture action(s) using epsilon-greedy active RL.

        With probability 1 - epsilon:
          If learned Q-values exist with Q(s, a) > min_q_threshold:
            Select optimal action a* = argmax Q(s, a).
            Decode and apply instantly with ZERO LLM calls.
        With probability epsilon (or if state is unvisited / uncertain):
          Consult the LLM adapter (if available) to propose candidate actions,
          or explore candidate actions from the mapper.
        """
        task_ctx = self.build_task_context(planner_output)
        encoder = ArchitectureStateEncoder(current_architecture)
        obs = encoder.encode()
        state_key = self.policy.state_encoder.encode(obs, task_ctx)

        # Build action mapper for the current architecture
        manager = ArchitectureManager(current_architecture)
        mapper = ArchitectureActionMapper.from_manager(manager)
        valid_action_ids = list(range(mapper.action_count))

        # Query learned Q-values
        state_actions = self.policy.q_table.get_state_actions(state_key)

        # Determine if we can exploit learned Q-table
        can_exploit = False
        best_action_id: Optional[int] = None
        best_q = -float("inf")

        if state_actions:
            for aid in valid_action_ids:
                q_val = state_actions.get(aid, self.policy.q_table.default_value)
                if q_val > best_q:
                    best_q = q_val
                    best_action_id = aid
            if best_action_id is not None and best_q > self.min_q_threshold:
                can_exploit = True

        roll = self.rng.random()

        # Branch 1: Exploit learned optimal action (Instant zero-LLM adaptation)
        if can_exploit and roll >= self.epsilon and best_action_id is not None:
            chosen_action = mapper.decode(best_action_id)
            serialized_action = chosen_action.serialize()
            return RLSelectionResult(
                decision_source="rl_policy",
                decision={
                    "decision": "apply_actions",
                    "reasoning": f"Active RL Q-Learning Policy selected optimal action (Q={best_q:.4f}, confidence=high)",
                    "actions": [serialized_action],
                },
                selected_actions=[serialized_action],
                valid_actions=[serialized_action],
                action_ids=[best_action_id],
                q_values=state_actions,
                confidence=best_q,
                is_instant=True,
                latency_saved_ms=1200.0,
                token_cost_saved=650,
                state_key=state_key,
            )

        # Branch 2: Consult LLM adapter (Exploration / Unseen state)
        if llm_adapter is not None:
            llm_result = llm_adapter.adapt_from_planner_output(planner_output)
            valid_acts = llm_result.valid_actions
            action_ids: List[int] = []
            for act_dict in valid_acts:
                try:
                    act_obj = ArchitectureAction.model_validate(act_dict)
                    aid = mapper.encode(act_obj)
                    action_ids.append(aid)
                except Exception:
                    # Action might be outside current single-step mapper (e.g. multi-step action)
                    pass

            source: Literal["rl_policy", "llm_adapter", "exploration", "default"] = (
                "exploration" if can_exploit else "llm_adapter"
            )
            return RLSelectionResult(
                decision_source=source,
                decision=llm_result.architecture_decision,
                selected_actions=llm_result.parsed_actions,
                valid_actions=valid_acts,
                rejected_actions=[r.serialize() if hasattr(r, "serialize") else dict(r) for r in llm_result.rejected_actions],
                action_ids=action_ids,
                q_values=state_actions,
                confidence=0.0,
                is_instant=False,
                latency_saved_ms=0.0,
                token_cost_saved=0,
                state_key=state_key,
            )

        # Branch 3: Random exploration fallback when no LLM adapter
        if valid_action_ids:
            rand_id = self.rng.choice(valid_action_ids)
            chosen_action = mapper.decode(rand_id)
            serialized_action = chosen_action.serialize()
            return RLSelectionResult(
                decision_source="exploration",
                decision={
                    "decision": "apply_actions",
                    "reasoning": f"Exploratory random action (epsilon={self.epsilon:.2f})",
                    "actions": [serialized_action],
                },
                selected_actions=[serialized_action],
                valid_actions=[serialized_action],
                action_ids=[rand_id],
                q_values=state_actions,
                confidence=0.0,
                is_instant=True,
                latency_saved_ms=1200.0,
                token_cost_saved=650,
                state_key=state_key,
            )

        # Default no-op
        return RLSelectionResult(
            decision_source="default",
            decision={"decision": "no_change", "reasoning": "No valid actions", "actions": []},
            state_key=state_key,
        )
