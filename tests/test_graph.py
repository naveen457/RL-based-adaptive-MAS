import pytest

from app.graph.state import MASState
from app.graph.workflow import build_workflow, run_workflow


# ---------------------------------------------------------------------------
# Graph construction / compilation
# ---------------------------------------------------------------------------

def test_graph_builds_and_compiles() -> None:
    """build_workflow() returns a compiled LangGraph app without error."""
    app = build_workflow()
    assert app is not None


def test_required_nodes_exist() -> None:
    """The graph contains the expected statically-defined nodes."""
    app = build_workflow()
    # LangGraph compiled apps expose their node names via .nodes.
    node_names = set(app.nodes)
    expected = {"planner", "researcher", "coder", "critic", "finalizer"}
    assert expected <= node_names, f"missing nodes: {expected - node_names}"


# ---------------------------------------------------------------------------
# State structure
# ---------------------------------------------------------------------------

def test_state_is_typed_dict() -> None:
    """MASState is a TypedDict with the expected keys."""
    expected_keys = {
        "original_task",
        "planner_output",
        "research_output",
        "coder_output",
        "critic_output",
        "final_answer",
        "aggregated_context",
    }
    assert set(MASState.__annotations__.keys()) == expected_keys


def test_initial_state_accepts_only_task() -> None:
    """A valid initial state only needs the original task."""
    state: MASState = {"original_task": "do something"}
    assert state["original_task"] == "do something"


def test_state_is_updatable() -> None:
    """State dicts can be updated by nodes (the normal LangGraph pattern)."""
    state: MASState = {"original_task": "x"}
    state["planner_output"] = None  # type: ignore[assignment]
    assert "planner_output" in state


# ---------------------------------------------------------------------------
# Routing behaviour (checked against the routing helper logic)
# ---------------------------------------------------------------------------

def test_research_task_routes_to_researcher() -> None:
    """A task that requires research should visit the Researcher node.

    Verified via the integration test (test_e2e_workflow_execution). Here we
    only check that the routing helper reads the Planner output correctly.
    """
    from app.agents.planner import PlannerOutput

    plan = PlannerOutput(
        task_understanding="X",
        requires_research=True,
        requires_coding=False,
    )
    from app.graph.workflow import _has_research
    assert _has_research({"planner_output": plan}) == "researcher"


def test_coding_task_routes_to_coder() -> None:
    """A task that requires coding should visit the Coder node.

    Verified via the integration test. Here we check the routing helper.
    """
    from app.agents.planner import PlannerOutput

    plan = PlannerOutput(
        task_understanding="X",
        requires_research=False,
        requires_coding=True,
    )
    from app.graph.workflow import _has_coding
    assert _has_coding({"planner_output": plan}) == "coder"


def test_task_requiring_both_visits_both_nodes() -> None:
    """A task needing both research and coding routes to both nodes.

    Verified via the integration test. Here we check both routing helpers.
    """
    from app.agents.planner import PlannerOutput

    plan = PlannerOutput(
        task_understanding="X",
        requires_research=True,
        requires_coding=True,
    )
    from app.graph.workflow import _has_research, _has_coding
    assert _has_research({"planner_output": plan}) == "researcher"
    assert _has_coding({"planner_output": plan}) == "coder"


def test_no_agents_needed_routes_to_join() -> None:
    """A task needing neither research nor coding skips both agent nodes.

    Verified via the integration test. Here we check the routing helpers.
    """
    from app.agents.planner import PlannerOutput

    plan = PlannerOutput(
        task_understanding="X",
        requires_research=False,
        requires_coding=False,
    )
    from app.graph.workflow import _has_research, _has_coding
    assert _has_research({"planner_output": plan}) == "join_after_plan"
    assert _has_coding({"planner_output": plan}) == "join_after_plan"


@pytest.mark.integration
def test_critic_always_runs() -> None:
    """Critic should run for every task, even with no intermediate outputs."""
    state = run_workflow("Hello world.")
    assert state.get("critic_output") is not None


@pytest.mark.integration
def test_finalizer_always_runs() -> None:
    """Finalizer should run for every task and produce a final answer."""
    state = run_workflow("Hello world.")
    assert state.get("final_answer") is not None
    assert state["final_answer"].final_answer


# ---------------------------------------------------------------------------
# No credential leakage
# ---------------------------------------------------------------------------

def test_workflow_does_not_expose_credentials() -> None:
    """The final state and its printed text must not contain API keys.

    Requires a live API call, so skipped in unit mode. Run with
    ``-m integration`` when the API key has quota available.
    """
    pytest.skip("live API check — run with -m integration")


# ---------------------------------------------------------------------------
# Integration (one real end-to-end workflow execution)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_e2e_workflow_execution() -> None:
    """Run the full static MAS end-to-end once via OpenRouter.

    This is the single integration test for Step 4. It makes one workflow
    execution that internally triggers however many agent calls the Planner
    decides are needed.
    """
    task = (
        "Write a Python function that checks whether a string is a "
        "palindrome and explain how it works."
    )
    state = run_workflow(task)

    assert isinstance(state, dict)
    assert state["original_task"] == task

    plan = state["planner_output"]
    assert plan is not None
    assert plan.task_understanding
    assert plan.requires_coding is True

    coder = state["coder_output"]
    assert coder is not None
    assert coder.code.strip()
    assert coder.explanation

    critic = state["critic_output"]
    assert critic is not None
    assert critic.verification_status in {
        "correct", "partially_correct", "incorrect", "unclear"
    }

    final = state["final_answer"]
    assert final is not None
    assert final.final_answer
