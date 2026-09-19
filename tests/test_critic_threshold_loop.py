import pytest
from unittest.mock import MagicMock

from app.agents.critic import CriticOutput
from app.agents.planner import PlannerOutput
from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.manager import ArchitectureManager
from app.config.settings import settings
from app.graph.dynamic_builder import DynamicGraphBuilder


def test_critic_high_score_skips_loop():
    """When Critic quality_score >= threshold (e.g. 0.85 >= 0.75), graph proceeds directly to finalizer with 0 retries."""
    mgr = ArchitectureManager.create_default_architecture()
    mgr.apply_action(ArchitectureAction(action_type=ActionType.ACTIVATE_AGENT, agent_id="tool_executor"))
    mgr.apply_action(ArchitectureAction(action_type=ActionType.DEACTIVATE_AGENT, agent_id="researcher"))
    mgr.apply_action(ArchitectureAction(action_type=ActionType.DEACTIVATE_AGENT, agent_id="coder"))
    arch = mgr.get_architecture()

    planner_out = PlannerOutput(
        task_understanding="get current time",
        required_capabilities=["tool_use"],
        requires_tools=True,
        requires_verification=True,
        tools_needed=["get_current_date"],
    )

    call_counts = {"planner": 0, "tool_executor": 0, "critic": 0, "finalizer": 0}

    def planner_h(state):
        call_counts["planner"] += 1
        return {"planner_output": planner_out}

    def tool_h(state):
        call_counts["tool_executor"] += 1
        return {"tool_output": [{"tool_name": "get_current_date", "result": "2026-09-19"}]}

    def critic_h(state):
        call_counts["critic"] += 1
        critic_out = CriticOutput(
            overall_assessment="Complete and accurate.",
            verification_status="correct",
            quality_score=0.90,  # >= 0.75 threshold
        )
        return {"critic_output": critic_out}

    def finalizer_h(state):
        call_counts["finalizer"] += 1
        return {"final_answer": "Task complete."}

    handlers = {
        "planner": planner_h,
        "tool_executor": tool_h,
        "critic": critic_h,
        "finalizer": finalizer_h,
    }

    builder = DynamicGraphBuilder()
    build_res = builder.build(arch, planner_out, handlers)
    compiled = build_res.compiled_graph

    init_state = {
        "original_task": "get current time",
        "_feedback_iterations": 0,
    }
    result = compiled.invoke(init_state)

    assert call_counts["planner"] == 1
    assert call_counts["tool_executor"] == 1
    assert call_counts["critic"] == 1
    assert call_counts["finalizer"] == 1
    assert result.get("_feedback_iterations", 0) == 0


def test_critic_low_score_triggers_retry_loop():
    """When Critic quality_score < threshold (e.g. 0.40 < 0.75), graph loops back to tool_executor for follow-up."""
    mgr = ArchitectureManager.create_default_architecture()
    mgr.apply_action(ArchitectureAction(action_type=ActionType.ACTIVATE_AGENT, agent_id="tool_executor"))
    mgr.apply_action(ArchitectureAction(action_type=ActionType.DEACTIVATE_AGENT, agent_id="researcher"))
    mgr.apply_action(ArchitectureAction(action_type=ActionType.DEACTIVATE_AGENT, agent_id="coder"))
    arch = mgr.get_architecture()

    planner_out = PlannerOutput(
        task_understanding="get time and modi bio",
        required_capabilities=["tool_use", "web_search"],
        requires_tools=True,
        requires_verification=True,
        tools_needed=["get_current_date"],
    )

    critic_call_count = 0
    tool_call_count = 0

    def planner_h(state):
        return {"planner_output": planner_out}

    def tool_h(state):
        nonlocal tool_call_count
        tool_call_count += 1
        return {"tool_output": [f"tool result pass {tool_call_count}"]}

    def critic_h(state):
        nonlocal critic_call_count
        critic_call_count += 1
        if critic_call_count == 1:
            # First pass: fail because missing bio
            critic_out = CriticOutput(
                overall_assessment="Missing bio data for Narendra Modi.",
                verification_status="partially_correct",
                quality_score=0.40,  # Below 0.75 -> triggers retry
                retry_target_node="tool_executor",
                suggested_tool_calls=[{"tool_name": "web_search", "arguments": {"query": "Narendra Modi bio"}}],
            )
        else:
            # Second pass: pass!
            critic_out = CriticOutput(
                overall_assessment="All requirements now satisfied.",
                verification_status="correct",
                quality_score=0.92,
            )
        return {"critic_output": critic_out}

    def finalizer_h(state):
        return {"final_answer": "Complete final answer."}

    handlers = {
        "planner": planner_h,
        "tool_executor": tool_h,
        "critic": critic_h,
        "finalizer": finalizer_h,
    }

    builder = DynamicGraphBuilder()
    build_res = builder.build(arch, planner_out, handlers)
    compiled = build_res.compiled_graph

    init_state = {
        "original_task": "get time and modi bio",
        "_feedback_iterations": 0,
    }
    result = compiled.invoke(init_state)

    assert tool_call_count == 2
    assert critic_call_count == 2
    assert result.get("_feedback_iterations") == 1
    assert "final_answer" in result


def test_critic_max_retries_safeguard_prevents_infinite_loop():
    """Even if Critic always gives quality_score < threshold, max_retries = 1 forces termination to finalizer."""
    mgr = ArchitectureManager.create_default_architecture()
    mgr.apply_action(ArchitectureAction(action_type=ActionType.ACTIVATE_AGENT, agent_id="tool_executor"))
    mgr.apply_action(ArchitectureAction(action_type=ActionType.DEACTIVATE_AGENT, agent_id="researcher"))
    mgr.apply_action(ArchitectureAction(action_type=ActionType.DEACTIVATE_AGENT, agent_id="coder"))
    arch = mgr.get_architecture()

    planner_out = PlannerOutput(
        task_understanding="hard task",
        required_capabilities=["tool_use"],
        requires_tools=True,
        requires_verification=True,
        tools_needed=["get_current_date"],
    )

    critic_call_count = 0
    tool_call_count = 0
    finalizer_called = False

    def planner_h(state):
        return {"planner_output": planner_out}

    def tool_h(state):
        nonlocal tool_call_count
        tool_call_count += 1
        return {"tool_output": ["result"]}

    def critic_h(state):
        nonlocal critic_call_count
        critic_call_count += 1
        critic_out = CriticOutput(
            overall_assessment="Still not satisfied.",
            verification_status="incorrect",
            quality_score=0.20,  # Always below threshold
            retry_target_node="tool_executor",
        )
        return {"critic_output": critic_out}

    def finalizer_h(state):
        nonlocal finalizer_called
        finalizer_called = True
        return {"final_answer": "Final output despite low score."}

    handlers = {
        "planner": planner_h,
        "tool_executor": tool_h,
        "critic": critic_h,
        "finalizer": finalizer_h,
    }

    builder = DynamicGraphBuilder()
    build_res = builder.build(arch, planner_out, handlers)
    compiled = build_res.compiled_graph

    init_state = {
        "original_task": "hard task",
        "_feedback_iterations": 0,
    }
    result = compiled.invoke(init_state)

    assert tool_call_count == 2
    assert critic_call_count == 2
    assert finalizer_called is True
    assert result.get("_feedback_iterations") == 2
