from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agents.critic import Critic, create_critic_review
from app.agents.coder import Coder, create_code_solution
from app.agents.finalizer import Finalizer, create_final_answer
from app.agents.planner import Planner, create_plan
from app.agents.researcher import Researcher, create_research_report
from app.config.settings import settings
from app.graph.state import MASState


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def _planner_node(state: MASState) -> dict:
    """Run the Planner on the original task and write planner_output."""
    planner = Planner.from_settings()
    plan = planner.plan(state["original_task"])
    return {"planner_output": plan}


def _researcher_node(state: MASState) -> dict:
    """Run the Researcher when the Planner indicated research is required."""
    researcher = Researcher.from_settings()
    research = researcher.research(state["original_task"])
    return {"research_output": research}


def _coder_node(state: MASState) -> dict:
    """Run the Coder when the Planner indicated coding is required."""
    coder = Coder.from_settings()
    code = coder.code(state["original_task"])
    return {"coder_output": code}


def _critic_node(state: MASState) -> dict:
    """Run the Critic over the available intermediate outputs.

    The Critic reviews the original task plus whatever the Researcher and/or
    Coder produced. If neither ran, it still reviews the task against an empty
    context so the graph always reaches the Finalizer.
    """
    critic = Critic.from_settings()

    plan = state.get("planner_output")
    if plan and not plan.requires_verification:
        return {"critic_output": None}  # skip critic when not needed

    context_parts: list[str] = []
    if state.get("research_output"):
        context_parts.append(
            "Research findings:\n"
            + "\n".join(state["research_output"].findings or [])
        )
    if state.get("coder_output"):
        context_parts.append(
            "Code implementation:\n"
            + state["coder_output"].code
        )

    aggregated = "\n\n".join(context_parts) if context_parts else "(no intermediate outputs)"

    review = critic.review(
        original_task=state["original_task"],
        output_to_review=aggregated,
    )
    return {"critic_output": review}


def _join_after_plan_node(state: MASState) -> dict:
    """No-op join point after Researcher and/or Coder.

    This node exists only so the graph has a registered node to route both
    conditional branches through before proceeding to the Critic.
    """
    return {}


def _finalizer_node(state: MASState) -> dict:
    """Synthesize the final answer from the available intermediate outputs."""
    finalizer = Finalizer.from_settings()

    context_parts: list[str] = []
    if state.get("research_output"):
        context_parts.append(
            "Research findings:\n"
            + "\n".join(state["research_output"].findings or [])
        )
    if state.get("coder_output"):
        context_parts.append(
            "Code implementation:\n"
            + state["coder_output"].code
            + "\n\nExplanation:\n"
            + state["coder_output"].explanation
        )
    if state.get("critic_output"):
        context_parts.append(
            "Critic review:\n"
            + state["critic_output"].overall_assessment
            + "\nIssues:\n"
            + "\n".join(state["critic_output"].issues or [])
            + "\nCorrections:\n"
            + "\n".join(state["critic_output"].corrections or [])
        )

    supporting = "\n\n".join(context_parts) if context_parts else "(no intermediate outputs)"

    final = finalizer.finalize(
        original_task=state["original_task"],
        supporting_info=supporting,
    )
    return {"final_answer": final, "aggregated_context": supporting}


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _has_research(state: MASState) -> str:
    plan = state.get("planner_output")
    if plan and getattr(plan, "requires_research", False):
        return "researcher"
    return "join_after_plan"


def _has_coding(state: MASState) -> str:
    plan = state.get("planner_output")
    if plan and getattr(plan, "requires_coding", False):
        return "coder"
    return "join_after_plan"



def _needs_verification(state: MASState) -> str:
    """Route to Critic only when the Planner flagged verification."""
    plan = state.get("planner_output")
    if plan and getattr(plan, "requires_verification", False):
        return "critic"
    return "finalizer"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_workflow():
    """Construct and compile the static multi-agent workflow.

    The topology is fixed (static baseline). The Planner's structured output
    only controls *which* of the Researcher and Coder nodes are visited before
    the Critic and Finalizer run.
    """
    graph = StateGraph(MASState)

    # Register all nodes (the entry/final nodes are registered explicitly too).
    graph.add_node("planner", _planner_node)
    graph.add_node("researcher", _researcher_node)
    graph.add_node("coder", _coder_node)
    graph.add_node("critic", _critic_node)
    graph.add_node("finalizer", _finalizer_node)
    graph.add_node("join_after_plan", _join_after_plan_node)

    # Entry -> Planner
    graph.set_entry_point("planner")

    # Planner -> Researcher (conditional)
    graph.add_conditional_edges(
        "planner",
        _has_research,
        {"researcher": "researcher", "join_after_plan": "join_after_plan"},
    )

    # Planner -> Coder (conditional). Both conditional edges may be active
    # simultaneously, giving LangGraph a parallel branch from the Planner.
    graph.add_conditional_edges(
        "planner",
        _has_coding,
        {"coder": "coder", "join_after_plan": "join_after_plan"},
    )

    # Researcher -> join point
    graph.add_edge("researcher", "join_after_plan")
    # Coder -> join point
    graph.add_edge("coder", "join_after_plan")

    # Join point -> Critic (when verification is required) -> Finalizer -> END
    graph.add_conditional_edges(
        "join_after_plan",
        _needs_verification,
        {"critic": "critic", "finalizer": "finalizer"},
    )
    graph.add_edge("critic", "finalizer")
    graph.add_edge("finalizer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Clean execution API
# ---------------------------------------------------------------------------

def run_workflow(task: str) -> MASState:
    """Run the static MAS on *task* and return the final state.

    This is the clean public API. Callers do not need to know anything about
    LangGraph internals.
    """
    app = build_workflow()
    return app.invoke({"original_task": task})
