"""Dynamic LangGraph construction from an adapted architecture."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from app.agents.planner import PlannerOutput
from app.architecture.models import MASArchitecture
from app.graph.state import ExtendedMASState, MASState

NodeHandler = Callable[[MASState], Dict[str, Any]]


class DynamicGraphMetadata(BaseModel):
    """Inspectable metadata for one dynamically compiled graph."""

    graph_nodes: list[str]
    graph_edges: list[dict[str, str]]
    active_agents: list[str]
    architecture_version: int
    architecture_actions: list[dict[str, Any]] = Field(default_factory=list)
    compiled: bool = False
    graph_representation: str = ""

    def serialize(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class DynamicGraphBuildResult(BaseModel):
    """Metadata returned with a compiled graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: DynamicGraphMetadata
    compiled_graph: Any = None


class DynamicGraphBuilder:
    """Translate adapted architecture state into a compiled LangGraph graph."""

    def build(
        self,
        architecture: MASArchitecture,
        planner_output: PlannerOutput,
        node_handlers: Mapping[str, NodeHandler],
        *,
        architecture_version: int = 0,
        architecture_actions: list[dict[str, Any]] | None = None,
        entry_point: Optional[str] = None,
        excluded_nodes: Optional[Set[str]] = None,
        include_finalizer: bool = True,
        checkpointer: Optional[Any] = None,
    ) -> DynamicGraphBuildResult:
        active = set(architecture.active_agent_ids)
        required = {"planner"}
        if include_finalizer:
            required.add("finalizer")
        if planner_output.requires_research:
            required.add("researcher")
        if planner_output.requires_coding:
            required.add("coder")
        if planner_output.requires_verification:
            required.add("critic")
        has_tool_demand = (
            "tool_executor" in active
            or getattr(planner_output, "requires_tools", False)
            or "tool_executor" in (getattr(planner_output, "selected_agents", []) or [])
            or bool(getattr(planner_output, "tools_needed", []))
            or bool(set(getattr(planner_output, "required_capabilities", []) or []) & {"tool_use", "web_search", "external_api", "tools"})
            or any(
                e.source == "tool_executor" or e.target == "tool_executor"
                for e in architecture.communication_edges
            )
        )
        if has_tool_demand:
            required.add("tool_executor")

        raw_nodes = set(active & required)
        if has_tool_demand:
            raw_nodes.add("tool_executor")
        if excluded_nodes:
            raw_nodes = raw_nodes - excluded_nodes

        graph_nodes = sorted(raw_nodes)
        actual_entry = entry_point or "planner"
        if actual_entry not in graph_nodes:
            graph_nodes.insert(0, actual_entry)
        if not graph_nodes:
            raise ValueError("dynamic graph requires at least one active node")

        configured_edges = [
            {"source": edge.source, "target": edge.target}
            for edge in architecture.communication_edges
            if edge.source in graph_nodes
            and edge.target in graph_nodes
            and not (
                edge.source == "planner"
                and edge.target == "finalizer"
                and (planner_output.requires_research or planner_output.requires_coding or "tool_executor" in graph_nodes)
            )
        ]
        if "critic" not in graph_nodes and "finalizer" in graph_nodes:
            for edge in architecture.communication_edges:
                if edge.target == "critic" and edge.source in graph_nodes:
                    bypass_edge = {"source": edge.source, "target": "finalizer"}
                    if bypass_edge not in configured_edges:
                        configured_edges.append(bypass_edge)

        if "tool_executor" in graph_nodes:
            if "planner" in graph_nodes and not any(e["source"] == "planner" and e["target"] == "tool_executor" for e in configured_edges):
                configured_edges.append({"source": "planner", "target": "tool_executor"})
            has_outgoing = any(e["source"] == "tool_executor" for e in configured_edges)
            if not has_outgoing and "finalizer" in graph_nodes:
                target_node = "critic" if "critic" in graph_nodes else "finalizer"
                configured_edges.append({"source": "tool_executor", "target": target_node})

        graph_edges = sorted(
            configured_edges,
            key=lambda edge: (edge["source"], edge["target"]),
        )

        missing_handlers = [node for node in graph_nodes if node not in node_handlers]
        if missing_handlers:
            raise ValueError(
                f"missing handlers for dynamic graph nodes: {missing_handlers}"
            )

        # Detect feedback loops (e.g. finalizer -> planner) to prevent infinite recursion
        feedback_edges = {
            (edge["source"], edge["target"])
            for edge in graph_edges
            if edge["source"] == "finalizer" and edge["target"] in graph_nodes
        }
        feedback_sources = {src for src, _ in feedback_edges}

        wrapped_handlers = dict(node_handlers)
        for src_node in feedback_sources:
            if src_node in wrapped_handlers:
                orig_handler = wrapped_handlers[src_node]
                def _wrap(fn):
                    def _wrapped(state: Any) -> Dict[str, Any]:
                        res = fn(state)
                        curr_count = state.get("_feedback_iterations", 0) if isinstance(state, dict) else 0
                        if isinstance(res, dict):
                            res["_feedback_iterations"] = curr_count + 1
                        return res
                    return _wrapped
                wrapped_handlers[src_node] = _wrap(orig_handler)

        graph = StateGraph(ExtendedMASState)
        for node in graph_nodes:
            graph.add_node(node, wrapped_handlers[node])
        graph.set_entry_point(actual_entry)

        outgoing = {edge["source"] for edge in graph_edges}
        for edge in graph_edges:
            src = edge["source"]
            tgt = edge["target"]
            if (src, tgt) in feedback_edges:
                # Conditional router: allows 1 feedback loop (when count <= 1), then terminates cleanly at END
                def _make_feedback_router(target_node: str):
                    def _router(state: Any) -> str:
                        count = state.get("_feedback_iterations", 0) if isinstance(state, dict) else 0
                        if count > 1:
                            return END
                        return target_node
                    return _router

                graph.add_conditional_edges(
                    src,
                    _make_feedback_router(tgt),
                    {tgt: tgt, END: END},
                )
            else:
                graph.add_edge(src, tgt)

        for node in graph_nodes:
            if node not in outgoing:
                graph.add_edge(node, END)

        compiled_graph = (
            graph.compile(checkpointer=checkpointer)
            if checkpointer is not None
            else graph.compile()
        )
        representation = compiled_graph.get_graph().draw_mermaid()
        metadata = DynamicGraphMetadata(
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            active_agents=graph_nodes,
            architecture_version=architecture_version,
            architecture_actions=architecture_actions or [],
            compiled=True,
            graph_representation=representation,
        )
        return DynamicGraphBuildResult(
            metadata=metadata,
            compiled_graph=compiled_graph,
        )
