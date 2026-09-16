from app.agents.coder import CoderOutput
from app.agents.critic import CriticOutput
from app.agents.finalizer import FinalizerOutput
from app.agents.planner import PlannerOutput
from app.agents.researcher import ResearcherOutput
from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.llm_adapter import LLMArchitectureAdapter
from app.architecture.manager import ArchitectureManager
from app.graph.adaptive_integration import AdaptiveWorkflowAdapter
from app.runtime.llm_execution import ExistingLLMAgentExecutor
from app.runtime.orchestrator import AdaptiveRuntimeOrchestrator


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeDecisionLLM:
    def __init__(self, response):
        self.response = response

    def invoke(self, messages):
        return FakeResponse(self.response)


class FakePlanner:
    def __init__(self, output, client):
        self.output = output
        self.model = client

    def plan(self, task):
        return self.output


class RecordingResearcher:
    calls = []

    def research(self, task):
        self.calls.append(task)
        return ResearcherOutput(
            research_question=task,
            findings=["real researcher output"],
            sources_or_evidence=["test source"],
        )


class RecordingCoder:
    calls = []

    def code(self, task):
        self.calls.append(task)
        return CoderOutput(
            approach="real approach",
            code="public class Solution {}",
            explanation="real coder output",
        )


class RecordingCritic:
    calls = []

    def review(self, original_task, output_to_review):
        self.calls.append((original_task, output_to_review))
        return CriticOutput(
            overall_assessment="real review",
            verification_status="correct",
        )


class RecordingFinalizer:
    calls = []

    def finalize(self, original_task, supporting_info):
        self.calls.append((original_task, supporting_info))
        return FinalizerOutput(
            final_answer="real final answer",
            key_points=["derived from agent outputs"],
            limitations=[],
        )


def _plan(*, coding=False, research=False, verification=False):
    return PlannerOutput(
        task_understanding="Complete the task.",
        required_capabilities=[],
        steps=["Complete"],
        requires_coding=coding,
        requires_research=research,
        requires_verification=verification,
        estimated_complexity="medium",
    )


def _executor():
    RecordingResearcher.calls = []
    RecordingCoder.calls = []
    RecordingCritic.calls = []
    RecordingFinalizer.calls = []
    return ExistingLLMAgentExecutor(
        researcher_factory=RecordingResearcher,
        coder_factory=RecordingCoder,
        critic_factory=RecordingCritic,
        finalizer_factory=RecordingFinalizer,
    )


def _runtime(plan, decision, *, inactive_agent=None):
    manager = ArchitectureManager.create_default_architecture()
    if inactive_agent:
        manager.apply_action(
            ArchitectureAction(
                action_type=ActionType.DEACTIVATE_AGENT,
                agent_id=inactive_agent,
            )
        )
    workflow = AdaptiveWorkflowAdapter(manager=manager)
    client = FakeDecisionLLM(decision)
    adapter = LLMArchitectureAdapter(client, workflow_adapter=workflow)
    return AdaptiveRuntimeOrchestrator(
        planner=FakePlanner(plan, client),
        architecture_adapter=adapter,
        agent_executor=_executor(),
    )


def test_coding_runtime_calls_real_coder_and_downstream_agents() -> None:
    task = "Write a Java solution and verify edge cases."
    runtime = _runtime(
        _plan(coding=True, verification=True),
        {
            "decision": "apply_actions",
            "reasoning": "Activate coder.",
            "actions": [{"action_type": "activate_agent", "agent_id": "coder"}],
        },
        inactive_agent="coder",
    )

    result = runtime.run(task)

    assert RecordingCoder.calls == [task]
    assert RecordingCritic.calls[0][0] == task
    assert "public class Solution {}" in RecordingCritic.calls[0][1]
    assert RecordingFinalizer.calls[0][0] == task
    assert "real review" in RecordingFinalizer.calls[0][1]
    assert result.final_response["final_answer"] == "real final answer"
    coder_record = next(
        record for record in result.execution_trace if record.agent_id == "coder"
    )
    assert coder_record.output["code"] == "public class Solution {}"


def test_research_runtime_calls_real_researcher_and_passes_output_forward() -> None:
    task = "Research a technical topic and synthesize it."
    runtime = _runtime(
        _plan(research=True),
        {
            "decision": "apply_actions",
            "reasoning": "Activate researcher.",
            "actions": [{"action_type": "activate_agent", "agent_id": "researcher"}],
        },
        inactive_agent="researcher",
    )

    result = runtime.run(task)

    assert RecordingResearcher.calls == [task]
    assert "real researcher output" in RecordingFinalizer.calls[0][1]
    assert result.final_response["final_answer"] == "real final answer"


def test_simple_task_does_not_construct_specialized_agents() -> None:
    task = "Say hello."
    runtime = _runtime(
        _plan(),
        {"decision": "no_change", "reasoning": "No specialization.", "actions": []},
    )

    result = runtime.run(task)

    assert RecordingCoder.calls == []
    assert RecordingResearcher.calls == []
    assert RecordingCritic.calls == []
    assert RecordingFinalizer.calls == [(task, "(no intermediate outputs)")]
    assert result.agents_actually_invoked == ["planner", "finalizer"]


def test_inactive_agent_is_not_used_even_if_task_requests_it() -> None:
    task = "Write code."
    runtime = _runtime(
        _plan(coding=True),
        {"decision": "no_change", "reasoning": "No activation.", "actions": []},
        inactive_agent="coder",
    )

    result = runtime.run(task)

    assert RecordingCoder.calls == []
    assert "coder" not in result.agents_actually_invoked
    assert result.final_response["final_answer"] == "real final answer"
