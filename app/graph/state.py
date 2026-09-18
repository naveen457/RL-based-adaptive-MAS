from typing import Annotated, Any, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.agents.critic import CriticOutput
from app.agents.coder import CoderOutput
from app.agents.finalizer import FinalizerOutput
from app.agents.planner import PlannerOutput
from app.agents.researcher import ResearcherOutput


class MASState(TypedDict, total=False):
    """Typed state flowing through the static multi-agent workflow.

    ``total=False`` because not every field is populated for every task
    (e.g. a task that needs no research will leave ``research_output`` unset).
    """

    # --- Inputs ---------------------------------------------------------
    original_task: str

    # --- Planner -------------------------------------------------------
    planner_output: PlannerOutput

    # --- Per-agent outputs (optional, populated when the node runs) -----
    research_output: ResearcherOutput
    coder_output: CoderOutput
    critic_output: CriticOutput

    # --- Final result --------------------------------------------------
    final_answer: FinalizerOutput

    # --- Convenience aggregated text for the Finalizer -----------------
    # Built by the graph from the available agent outputs so the Finalizer
    # receives a single readable block rather than raw state keys.
    aggregated_context: str


class ExtendedMASState(MASState, total=False):
    """Extended state supporting tool calling, dynamic node extensions, and conversation threading."""

    # Threading and message queueing (default thread: 'thread-1')
    messages: Annotated[Sequence[BaseMessage], add_messages]
    thread_id: str

    tool_output: Any
    tool_calls: list[dict[str, Any]]
    _feedback_iterations: int


