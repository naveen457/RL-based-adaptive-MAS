"""LLM-driven architecture recommendation and controlled application.

The LLM is treated as a recommender only. Every proposed action is parsed into
an existing ``ArchitectureAction`` and applied through the existing adaptive
workflow integration layer.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, Tuple

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.manager import ArchitectureManager
from app.architecture.models import MASArchitecture
from app.agents.planner import PlannerOutput
from app.config.settings import settings
from app.graph.adaptive_integration import (
    AdaptiveWorkflowAdapter,
    WorkflowExecutionPlan,
)


class LLMClient(Protocol):
    """Minimal client contract used by the adapter and deterministic tests."""

    def invoke(self, messages: Sequence[Dict[str, str]]) -> Any: ...


class ArchitectureRecommendation(BaseModel):
    """Structured recommendation envelope returned by an LLM."""

    reasoning: str = Field(default="")
    actions: List[Dict[str, Any]] = Field(default_factory=list)

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class PlannerArchitectureDecision(BaseModel):
    """Typed architecture decision produced from an existing PlannerOutput."""

    decision: Literal["apply_actions", "no_change"] = "apply_actions"
    reasoning: str = Field(default="")
    actions: List[Dict[str, Any]] = Field(default_factory=list)

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class RejectedAction(BaseModel):
    """An action rejected during parsing or architecture validation."""

    action: Dict[str, Any] = Field(default_factory=dict)
    reason: str

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class LLMAdaptationResult(BaseModel):
    """Serializable result of one LLM recommendation/application run."""

    task: str
    initial_architecture: Dict[str, Any]
    llm_recommendation: Dict[str, Any]
    parsed_actions: List[Dict[str, Any]] = Field(default_factory=list)
    valid_actions: List[Dict[str, Any]] = Field(default_factory=list)
    rejected_actions: List[RejectedAction] = Field(default_factory=list)
    final_architecture: Dict[str, Any]
    workflow_plan: WorkflowExecutionPlan

    @property
    def langgraph_compatible(self) -> bool:
        return self.workflow_plan.compatibility.compatible

    def serialize(self) -> Dict[str, Any]:
        data = self.model_dump(mode="json", exclude_none=True)
        data["langgraph_compatible"] = self.langgraph_compatible
        return data

    def to_dict(self) -> Dict[str, Any]:
        return self.serialize()


class PlannerArchitectureAdaptationResult(LLMAdaptationResult):
    """Serializable result for PlannerOutput-driven adaptation."""

    planner_output: Dict[str, Any]
    architecture_decision: Dict[str, Any]


SYSTEM_PROMPT = """You recommend architecture changes for an existing multi-agent system.
Return only one JSON object with this shape:
{"reasoning": "short explanation", "actions": [{"action_type": "...", ...}]}
Use only these action types: activate_agent, deactivate_agent, add_edge,
remove_edge, change_role. Use fields compatible with ArchitectureAction.
Do not invent agents, roles, or fields. The host application validates every
action before applying it.
"""

PLANNER_ARCHITECTURE_SYSTEM_PROMPT = """You convert an existing PlannerOutput into a typed
architecture decision for an existing multi-agent architecture.

Return only one JSON object with this shape:
{"decision":"apply_actions|no_change", "reasoning":"short explanation",
 "actions":[{"action_type":"...", ...}]}

Adapt the architecture towards its minimal sufficient topology (Pareto-optimal efficiency):
- If planner_output indicates requires_tools is true, or selected_agents contains "tool_executor", or tools_needed is non-empty, or capabilities include "web_search"/"tool_use":
  * You MUST activate tool_executor: {"action_type": "activate_agent", "agent_id": "tool_executor"}
  * You MUST deactivate unneeded specialist agents to prevent OVER_ENGINEERED waste:
    - If requires_research is false: {"action_type": "deactivate_agent", "agent_id": "researcher"}
    - If requires_coding is false: {"action_type": "deactivate_agent", "agent_id": "coder"}
    - If requires_verification is false: {"action_type": "deactivate_agent", "agent_id": "critic"}
- If requires_tools is false and tools/web_search are not needed, deactivate tool_executor if currently active.
- If requires_research is false and in-depth synthesis is not needed, deactivate researcher if currently active.
- If requires_coding is false and coding is not needed, deactivate coder if currently active.
- If requires_verification is false and verification is not needed, deactivate critic if currently active.
- If requires_research or research capability is needed, retain or activate researcher.
- If requires_coding or coding capability is needed, retain or activate coder.
- If requires_verification is needed, retain or activate critic.
- Always retain planner and finalizer.

You may activate any agent listed in available_registry_agents (e.g. tool_executor).
Use only these action types: activate_agent, deactivate_agent, add_edge, remove_edge, change_role.
Every action is validated by the host application.
For no_change, return decision="no_change" and an empty actions list.
"""


class LLMArchitectureAdapter:
    """Recommend and safely apply architecture actions for a user task."""

    def __init__(
        self,
        client: LLMClient,
        *,
        workflow_adapter: Optional[AdaptiveWorkflowAdapter] = None,
    ) -> None:
        self.client = client
        self.workflow_adapter = workflow_adapter or AdaptiveWorkflowAdapter()

    @classmethod
    def from_settings(
        cls,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "LLMArchitectureAdapter":
        """Create a live adapter from the existing environment-backed settings."""

        resolved_model = model if model is not None else settings.model
        resolved_base_url = base_url if base_url is not None else settings.base_url
        resolved_api_key = settings.api_key if api_key is None else api_key
        if not resolved_api_key:
            raise ValueError(
                "LLM API key is not configured. Set NVIDIA_API_KEY in the environment or .env file."
            )
        if not resolved_model:
            raise ValueError(
                "LLM model is not configured. Set NVIDIA_MODEL in the environment or .env file."
            )

        client = ChatOpenAI(
            model=resolved_model,
            openai_api_key=resolved_api_key,
            openai_api_base=resolved_base_url,
            temperature=0.0,
            max_tokens=settings.max_tokens,
        )
        return cls(client)

    def build_prompt(
        self,
        task_description: str,
        architecture: Optional[MASArchitecture] = None,
    ) -> List[Dict[str, str]]:
        """Build the task and architecture prompt without exposing credentials."""

        current = architecture or self.workflow_adapter.current_architecture()
        user_payload = {
            "task": task_description,
            "current_architecture": current.serialize(),
        }
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
        ]

    def recommend(
        self,
        task_description: str,
        architecture: Optional[MASArchitecture] = None,
    ) -> ArchitectureRecommendation:
        """Ask the client for a structured recommendation and parse its output."""

        response = self.client.invoke(self.build_prompt(task_description, architecture))
        return self.parse_response(response)

    def build_planner_prompt(
        self,
        planner_output: PlannerOutput,
        architecture: Optional[MASArchitecture] = None,
    ) -> List[Dict[str, str]]:
        """Build the planner-driven architecture prompt."""

        current = architecture or self.workflow_adapter.current_architecture()
        try:
            from app.agents.registry import default_registry
            registry_agents = [
                {
                    "agent_id": a.agent_id,
                    "role": a.role,
                    "description": a.description,
                    "capabilities": a.capabilities,
                    "tools": [t.tool_name for t in a.tools],
                }
                for a in default_registry.list_agents()
            ]
        except Exception:
            registry_agents = []

        user_payload = {
            "planner_output": planner_output.model_dump(mode="json"),
            "current_architecture": current.serialize(),
            "available_registry_agents": registry_agents,
        }
        return [
            {"role": "system", "content": PLANNER_ARCHITECTURE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
        ]

    def recommend_from_planner_output(
        self,
        planner_output: PlannerOutput,
        architecture: Optional[MASArchitecture] = None,
    ) -> PlannerArchitectureDecision:
        """Ask the existing LLM client for a decision based on PlannerOutput."""
        import time

        prompt = self.build_planner_prompt(planner_output, architecture)
        response = None
        for attempt in range(3):
            try:
                response = self.client.invoke(prompt)
                break
            except Exception as e:
                err_str = str(e).lower()
                if ("rate_limit" in err_str or "429" in err_str or "tokens per minute" in err_str) and attempt < 2:
                    print("  [Rate Limit] Replenishing tokens, waiting 5s...")
                    time.sleep(5)
                    continue
                raise

        payload = self._extract_payload(response)
        return self._normalize_planner_decision(payload)

    def adapt_from_planner_output(
        self,
        planner_output: PlannerOutput,
    ) -> PlannerArchitectureAdaptationResult:
        """Convert PlannerOutput into a validated, atomically applied decision."""

        initial = self.workflow_adapter.current_architecture()
        decision = self.recommend_from_planner_output(planner_output, initial)
        parsed_actions: List[ArchitectureAction] = []
        rejected: List[RejectedAction] = []

        for raw_action in decision.actions:
            try:
                parsed_actions.append(ArchitectureAction.model_validate(raw_action))
            except (TypeError, ValueError) as exc:
                rejected.append(
                    RejectedAction(action=dict(raw_action), reason=str(exc))
                )

        # Deterministic capability synchronization:
        # Guarantee that if PlannerOutput explicitly mandates tool_executor, researcher, coder, or critic,
        # the corresponding activation action is present so LLM completions never drop required capabilities.
        active_ids = set(initial.active_agent_ids)
        already_activated = {a.agent_id for a in parsed_actions if a.action_type == ActionType.ACTIVATE_AGENT}

        needs_tools = (
            planner_output.requires_tools
            or "tool_executor" in (planner_output.selected_agents or [])
            or bool(planner_output.tools_needed)
            or bool(set(planner_output.required_capabilities or []) & {"tool_use", "web_search", "external_api", "tools"})
        )
        if needs_tools and "tool_executor" not in active_ids and "tool_executor" not in already_activated:
            parsed_actions.append(
                ArchitectureAction(action_type=ActionType.ACTIVATE_AGENT, agent_id="tool_executor")
            )

        valid_actions: List[Dict[str, Any]] = []
        if not rejected:
            trial = AdaptiveWorkflowAdapter(manager=ArchitectureManager(initial))
            for action in parsed_actions:
                success, _, error = trial.try_apply_action(action)
                if not success:
                    rejected.append(
                        RejectedAction(action=action.serialize(), reason=error)
                    )

        if not rejected:
            for action in parsed_actions:
                success, _, error = self.workflow_adapter.try_apply_action(action)
                if success:
                    valid_actions.append(action.serialize())
                else:
                    rejected.append(
                        RejectedAction(action=action.serialize(), reason=error)
                    )

        final = self.workflow_adapter.current_architecture()
        plan = self.workflow_adapter.produce_workflow_config(final)
        return PlannerArchitectureAdaptationResult(
            task=planner_output.task_understanding,
            initial_architecture=initial.serialize(),
            llm_recommendation=decision.serialize(),
            parsed_actions=[action.serialize() for action in parsed_actions],
            valid_actions=valid_actions,
            rejected_actions=rejected,
            final_architecture=final.serialize(),
            workflow_plan=plan,
            planner_output=planner_output.model_dump(mode="json"),
            architecture_decision=decision.serialize(),
        )

    def reassess_from_context(
        self,
        task: str,
        context: str,
        current_architecture: Optional[MASArchitecture] = None,
    ) -> PlannerArchitectureAdaptationResult:
        """Reassess architecture during execution based on task and intermediate context."""
        initial = current_architecture or self.workflow_adapter.current_architecture()
        full_context = f"{task}\n\nContext:\n{context}" if context else task

        user_payload = {
            "task": full_context,
            "current_architecture": initial.serialize(),
        }
        prompt = [
            {"role": "system", "content": PLANNER_ARCHITECTURE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
        ]
        response = self.client.invoke(prompt)
        payload = self._extract_payload(response)
        decision = self._normalize_planner_decision(payload)

        parsed_actions: List[ArchitectureAction] = []
        rejected: List[RejectedAction] = []

        for raw_action in decision.actions:
            try:
                parsed_actions.append(ArchitectureAction.model_validate(raw_action))
            except (TypeError, ValueError) as exc:
                rejected.append(
                    RejectedAction(action=dict(raw_action), reason=str(exc))
                )

        valid_actions: List[Dict[str, Any]] = []
        if not rejected:
            trial = AdaptiveWorkflowAdapter(manager=ArchitectureManager(initial))
            for action in parsed_actions:
                success, _, error = trial.try_apply_action(action)
                if not success:
                    rejected.append(
                        RejectedAction(action=action.serialize(), reason=error)
                    )

        if not rejected:
            for action in parsed_actions:
                success, _, error = self.workflow_adapter.try_apply_action(action)
                if success:
                    valid_actions.append(action.serialize())
                else:
                    rejected.append(
                        RejectedAction(action=action.serialize(), reason=error)
                    )

        final = self.workflow_adapter.current_architecture()
        plan = self.workflow_adapter.produce_workflow_config(final)
        return PlannerArchitectureAdaptationResult(
            task=task,
            initial_architecture=initial.serialize(),
            llm_recommendation=decision.serialize(),
            parsed_actions=[action.serialize() for action in parsed_actions],
            valid_actions=valid_actions,
            rejected_actions=rejected,
            final_architecture=final.serialize(),
            workflow_plan=plan,
            planner_output={},
            architecture_decision=decision.serialize(),
        )

    def parse_response(self, response: Any) -> ArchitectureRecommendation:
        """Parse a JSON/object response into the recommendation envelope."""

        payload = self._extract_payload(response)
        return ArchitectureRecommendation.model_validate(payload)

    @staticmethod
    def _extract_payload(response: Any) -> Dict[str, Any]:
        if isinstance(response, BaseModel):
            payload: Any = response.model_dump(mode="json")
        elif isinstance(response, dict):
            payload = response
        else:
            content = getattr(response, "content", response)
            if isinstance(content, dict):
                payload = content
            else:
                if isinstance(content, list):
                    content = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    )
                payload = LLMArchitectureAdapter._load_json(str(content))
        if isinstance(payload, str):
            payload = LLMArchitectureAdapter._load_json(payload)
        if not isinstance(payload, dict):
            raise ValueError("LLM response must be a JSON object")
        return payload

    @staticmethod
    def _normalize_planner_decision(
        payload: Dict[str, Any],
    ) -> PlannerArchitectureDecision:
        actions = list(payload.get("actions") or [])
        decision = payload.get("decision", "apply_actions")
        no_change_action = any(
            isinstance(action, dict)
            and str(action.get("action_type", "")).lower() == "no_change"
            for action in actions
        )
        if no_change_action:
            actions = [
                action
                for action in actions
                if str(action.get("action_type", "")).lower() != "no_change"
            ]
            decision = "no_change"
        if decision not in {"apply_actions", "no_change"}:
            raise ValueError("architecture decision must be apply_actions or no_change")
        if decision == "no_change" and actions:
            raise ValueError("no_change decisions must not contain actions")
        return PlannerArchitectureDecision(
            decision=decision,
            reasoning=str(payload.get("reasoning", "")),
            actions=actions,
        )

    def apply_recommendation(
        self,
        task_description: str,
        recommendation: ArchitectureRecommendation,
    ) -> LLMAdaptationResult:
        """Parse, validate, apply, and convert a recommendation safely."""

        initial = self.workflow_adapter.current_architecture()
        parsed_actions: List[ArchitectureAction] = []
        rejected: List[RejectedAction] = []
        for raw_action in recommendation.actions:
            try:
                parsed_actions.append(ArchitectureAction.model_validate(raw_action))
            except (TypeError, ValueError) as exc:
                rejected.append(
                    RejectedAction(action=dict(raw_action), reason=str(exc))
                )

        valid_actions: List[Dict[str, Any]] = []
        for action in parsed_actions:
            success, _, error = self.workflow_adapter.try_apply_action(action)
            if success:
                valid_actions.append(action.serialize())
            else:
                rejected.append(RejectedAction(action=action.serialize(), reason=error))

        final = self.workflow_adapter.current_architecture()
        plan = self.workflow_adapter.produce_workflow_config(final)
        return LLMAdaptationResult(
            task=task_description,
            initial_architecture=initial.serialize(),
            llm_recommendation=recommendation.serialize(),
            parsed_actions=[action.serialize() for action in parsed_actions],
            valid_actions=valid_actions,
            rejected_actions=rejected,
            final_architecture=final.serialize(),
            workflow_plan=plan,
        )

    def adapt(self, task_description: str) -> LLMAdaptationResult:
        """Run recommendation, controlled application, and compatibility conversion."""

        recommendation = self.recommend(
            task_description,
            self.workflow_adapter.current_architecture(),
        )
        return self.apply_recommendation(task_description, recommendation)

    def run_basic_flow(self, task_description: str) -> Tuple[LLMAdaptationResult, Any]:
        """Run adaptation, then execute the existing protected MAS workflow."""

        result = self.adapt(task_description)
        state = self.workflow_adapter.run_baseline_workflow(task_description)
        return result, state

    @staticmethod
    def _load_json(content: str) -> Dict[str, Any]:
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        candidate = fenced.group(1) if fenced else content.strip()
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response must be a JSON object")
        return parsed


def run_live_llm_smoke(
    task: str = "Research a technical topic and synthesize the findings.",
) -> LLMAdaptationResult:
    """Run one live recommendation/application smoke test."""

    return LLMArchitectureAdapter.from_settings().adapt(task)


def print_live_llm_smoke(
    task: str = "Research a technical topic and synthesize the findings.",
) -> None:
    """Print a credential-free live smoke summary."""

    result = run_live_llm_smoke(task)
    print("LLM_ADAPTATION_SMOKE_COMPLETED")
    print(f"task={result.task}")
    print(f"initial_architecture={result.initial_architecture}")
    print(f"llm_recommendation={result.llm_recommendation}")
    print(f"parsed_actions={result.parsed_actions}")
    print(f"valid_actions={result.valid_actions}")
    print(f"rejected_actions={[item.serialize() for item in result.rejected_actions]}")
    print(f"final_architecture={result.final_architecture}")
    print(f"langgraph_compatible={result.langgraph_compatible}")


if __name__ == "__main__":
    print_live_llm_smoke()
