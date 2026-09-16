"""Tests for runtime / during-execution architecture adaptation.

Verifies that the AMAS architecture can reassess and reconfigure itself
DURING execution using compiled LangGraph execution for both v0 and v1.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from app.agents.coder import CoderOutput
from app.agents.critic import CriticOutput
from app.agents.finalizer import FinalizerOutput
from app.agents.planner import PlannerOutput
from app.agents.researcher import ResearcherOutput
from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.llm_adapter import LLMArchitectureAdapter, LLMClient
from app.architecture.manager import ArchitectureManager
from app.graph.adaptive_integration import AdaptiveWorkflowAdapter
from app.runtime.llm_execution import ExistingLLMAgentExecutor
from app.runtime.orchestrator import AdaptiveRuntimeOrchestrator


class _FakeResponse:
    def __init__(self, content: Any) -> None:
        self.content = content


class SequentialDecisionLLM(LLMClient):
    """LLM client that returns different decisions for initial adaptation vs mid-execution reassessment."""

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.recorded_prompts: List[Any] = []
        self.model = self

    def invoke(self, messages: Any) -> Any:
        self.recorded_prompts.append(messages)
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return _FakeResponse(resp)
        return _FakeResponse({"decision": "no_change", "reasoning": "no more actions", "actions": []})


class FakePlanner:
    def __init__(self, output: PlannerOutput, client: Any) -> None:
        self.output = output
        self.model = client

    def plan(self, task: str) -> PlannerOutput:
        return self.output


def _make_agent_executor() -> ExistingLLMAgentExecutor:
    class _Researcher:
        def research(self, task: str) -> ResearcherOutput:
            return ResearcherOutput(
                research_question=task,
                findings=[
                    "Palindrome algorithms surveyed (two-pointer, expand-around-center).",
                    "A Java implementation with edge-case verification is appropriate and required.",
                ],
                sources_or_evidence=["Algorithm literature"],
            )

    class _Coder:
        def code(self, task: str) -> CoderOutput:
            return CoderOutput(
                approach="Two-pointer palindrome verification algorithm.",
                code="public class PalindromeChecker { public static boolean isPalindrome(String s) { return true; } }",
                explanation="Scans from both ends comparing characters.",
            )

    class _Critic:
        def review(self, original_task: str, output_to_review: str) -> CriticOutput:
            return CriticOutput(
                overall_assessment="Implementation correctly handles basic and edge cases.",
                verification_status="correct",
            )

    class _Finalizer:
        def finalize(self, original_task: str, supporting_info: str) -> FinalizerOutput:
            return FinalizerOutput(
                final_answer="Comprehensive palindrome solution with verified Java implementation.",
                key_points=[supporting_info],
                limitations=[],
            )

    return ExistingLLMAgentExecutor(
        researcher_factory=_Researcher,
        coder_factory=_Coder,
        critic_factory=_Critic,
        finalizer_factory=_Finalizer,
    )


def test_during_execution_architecture_adaptation_lifecycle() -> None:
    """Demonstrate runtime adaptation:
    v0: planner -> researcher -> (reassessment)
    v1: coder -> critic -> finalizer
    Logical path: planner -> researcher -> coder -> critic -> finalizer
    """
    task = "Research palindrome algorithms and, if appropriate, implement a Java solution and verify it."

    initial_plan = PlannerOutput(
        task_understanding=task,
        required_capabilities=["research", "information_synthesis"],
        steps=["Research palindrome algorithms", "Determine if coding is appropriate"],
        requires_research=True,
        requires_coding=False,
        requires_verification=False,
        estimated_complexity="medium",
    )

    client = SequentialDecisionLLM([
        {
            "decision": "apply_actions",
            "reasoning": "Initial plan requires research.",
            "actions": [{"action_type": "activate_agent", "agent_id": "researcher"}],
        },
        {
            "decision": "apply_actions",
            "reasoning": "Research findings confirm implementation and verification are needed.",
            "actions": [
                {"action_type": "activate_agent", "agent_id": "coder"},
                {"action_type": "activate_agent", "agent_id": "critic"},
            ],
        },
    ])

    manager = ArchitectureManager.create_default_architecture()
    manager.apply_action(
        ArchitectureAction(
            action_type=ActionType.DEACTIVATE_AGENT,
            agent_id="coder",
        )
    )
    manager.apply_action(
        ArchitectureAction(
            action_type=ActionType.DEACTIVATE_AGENT,
            agent_id="critic",
        )
    )
    workflow_adapter = AdaptiveWorkflowAdapter(manager=manager)
    architecture_adapter = LLMArchitectureAdapter(
        client,
        workflow_adapter=workflow_adapter,
    )
    orchestrator = AdaptiveRuntimeOrchestrator(
        planner=FakePlanner(initial_plan, client),
        architecture_adapter=architecture_adapter,
        agent_executor=_make_agent_executor(),
    )

    result = orchestrator.run_dynamic(task, runtime_adaptation=True)

    # 1. Execution Path
    assert result.actual_execution_path == [
        "planner",
        "researcher",
        "coder",
        "critic",
        "finalizer",
    ]
    assert result.triggering_agent == "researcher"
    assert result.execution_path_before == ["planner", "researcher"]
    assert result.execution_path_after == ["coder", "critic", "finalizer"]

    # 2. Architecture versions
    assert len(result.architecture_versions) == 2
    assert result.architecture_versions[0] < result.architecture_versions[1]
    assert result.architecture_version == result.architecture_versions[1]

    # 3. Before/After architecture
    assert result.architecture_before_reassessment is not None
    assert result.architecture_after_reassessment is not None
    assert result.architecture_before_reassessment != result.architecture_after_reassessment

    # 4. Context that triggered reassessment
    assert result.context_that_triggered_reassessment is not None
    assert "Palindrome algorithms surveyed" in result.context_that_triggered_reassessment

    # 5. Reassessment actions
    assert len(result.reassessment_actions_proposed) == 2
    assert len(result.reassessment_actions_accepted) == 2
    accepted_types = {(a["action_type"], a["agent_id"]) for a in result.reassessment_actions_accepted}
    assert ("activate_agent", "coder") in accepted_types
    assert ("activate_agent", "critic") in accepted_types

    # 6. Graph before and after
    assert result.graph_before is not None
    assert result.graph_after is not None
    assert "coder" not in result.graph_before["graph_nodes"]
    assert "coder" in result.graph_after["graph_nodes"]
    assert "critic" in result.graph_after["graph_nodes"]

    # 7. Trace event stream verification
    trace_events = [e.event for e in result.execution_trace]

    assert "architecture.v0.created" in trace_events
    assert "graph.v0.compiled" in trace_events
    assert "agent.planner.started" in trace_events
    assert "agent.planner.completed" in trace_events
    assert "agent.researcher.started" in trace_events
    assert "agent.researcher.completed" in trace_events
    assert "architecture.reassessment.started" in trace_events
    assert "architecture.reassessment.completed" in trace_events
    assert "architecture.v1.created" in trace_events
    assert "graph.v1.compiled" in trace_events
    assert "agent.coder.started" in trace_events
    assert "agent.coder.completed" in trace_events
    assert "agent.critic.started" in trace_events
    assert "agent.critic.completed" in trace_events
    assert "agent.finalizer.started" in trace_events
    assert "agent.finalizer.completed" in trace_events
    assert "workflow.completed" in trace_events

    # Verify event ordering
    assert trace_events.index("architecture.v0.created") < trace_events.index("graph.v0.compiled")
    assert trace_events.index("graph.v0.compiled") < trace_events.index("agent.researcher.started")
    assert trace_events.index("agent.researcher.completed") < trace_events.index("architecture.reassessment.started")
    assert trace_events.index("architecture.reassessment.started") < trace_events.index("architecture.reassessment.completed")
    assert trace_events.index("architecture.reassessment.completed") < trace_events.index("architecture.v1.created")
    assert trace_events.index("architecture.v1.created") < trace_events.index("graph.v1.compiled")
    assert trace_events.index("graph.v1.compiled") < trace_events.index("agent.coder.started")
    assert trace_events.index("agent.coder.completed") < trace_events.index("agent.critic.started")
    assert trace_events.index("agent.critic.completed") < trace_events.index("agent.finalizer.started")
    assert trace_events.index("agent.finalizer.completed") < trace_events.index("workflow.completed")

    # 8. Final response produced
    assert result.final_response is not None
    assert result.final_response.get("final_answer") == (
        "Comprehensive palindrome solution with verified Java implementation."
    )


def test_during_execution_no_change_scenario() -> None:
    """When intermediate context does not require new agents, reassessment leaves architecture intact."""
    task = "Research palindrome algorithms and, if appropriate, implement a Java solution."

    initial_plan = PlannerOutput(
        task_understanding=task,
        required_capabilities=["research"],
        steps=["Research"],
        requires_research=True,
        requires_coding=False,
        requires_verification=False,
        estimated_complexity="low",
    )

    client = SequentialDecisionLLM([
        {
            "decision": "apply_actions",
            "reasoning": "Initial research.",
            "actions": [{"action_type": "activate_agent", "agent_id": "researcher"}],
        },
        {
            "decision": "no_change",
            "reasoning": "Research is sufficient; no coding needed.",
            "actions": [],
        },
    ])

    manager = ArchitectureManager.create_default_architecture()
    workflow_adapter = AdaptiveWorkflowAdapter(manager=manager)
    architecture_adapter = LLMArchitectureAdapter(
        client,
        workflow_adapter=workflow_adapter,
    )
    orchestrator = AdaptiveRuntimeOrchestrator(
        planner=FakePlanner(initial_plan, client),
        architecture_adapter=architecture_adapter,
        agent_executor=_make_agent_executor(),
    )

    result = orchestrator.run_dynamic(task, runtime_adaptation=True)

    assert result.actual_execution_path == ["planner", "researcher", "finalizer"]
    assert result.triggering_agent == "researcher"
    assert result.execution_path_before == ["planner", "researcher"]
    assert result.execution_path_after == ["finalizer"]
    assert len(result.architecture_versions) == 1
    assert "coder" not in result.graph_after["graph_nodes"]
    assert "critic" not in result.graph_after["graph_nodes"]


def test_during_execution_reassessment_failure_safely_continues() -> None:
    """When reassessment encounters an error, runtime safely continues with the current architecture."""
    task = "Research palindrome algorithms and, if appropriate, implement a Java solution."

    initial_plan = PlannerOutput(
        task_understanding=task,
        required_capabilities=["research"],
        steps=["Research"],
        requires_research=True,
        requires_coding=False,
        requires_verification=False,
        estimated_complexity="low",
    )

    class ErrorDecisionLLM(LLMClient):
        def __init__(self) -> None:
            self.call_count = 0
            self.model = self

        def invoke(self, messages: Any) -> Any:
            self.call_count += 1
            if self.call_count == 1:
                return _FakeResponse(
                    {
                        "decision": "apply_actions",
                        "reasoning": "Initial research.",
                        "actions": [{"action_type": "activate_agent", "agent_id": "researcher"}],
                    }
                )
            raise RuntimeError("LLM reassessment service error")

    client = ErrorDecisionLLM()
    manager = ArchitectureManager.create_default_architecture()
    workflow_adapter = AdaptiveWorkflowAdapter(manager=manager)
    architecture_adapter = LLMArchitectureAdapter(
        client,
        workflow_adapter=workflow_adapter,
    )
    orchestrator = AdaptiveRuntimeOrchestrator(
        planner=FakePlanner(initial_plan, client),
        architecture_adapter=architecture_adapter,
        agent_executor=_make_agent_executor(),
    )

    result = orchestrator.run_dynamic(task, runtime_adaptation=True)

    assert result.reassessment_error == "LLM reassessment service error"
    assert result.actual_execution_path == ["planner", "researcher", "finalizer"]
    assert result.final_response is not None

