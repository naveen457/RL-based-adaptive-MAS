"""Dynamic LangGraph construction from an adapted architecture."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from app.agents.planner import PlannerOutput
from app.architecture.models import MASArchitecture
from app.graph.state import MASState

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

        raw_nodes = active & required
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
                and (planner_output.requires_research or planner_output.requires_coding)
            )
        ]
        if "critic" not in graph_nodes and "finalizer" in graph_nodes:
            for edge in architecture.communication_edges:
                if edge.target == "critic" and edge.source in graph_nodes:
                    bypass_edge = {"source": edge.source, "target": "finalizer"}
                    if bypass_edge not in configured_edges:
                        configured_edges.append(bypass_edge)

        graph_edges = sorted(
            configured_edges,
            key=lambda edge: (edge["source"], edge["target"]),
        )

        missing_handlers = [node for node in graph_nodes if node not in node_handlers]
        if missing_handlers:
            raise ValueError(
                f"missing handlers for dynamic graph nodes: {missing_handlers}"
            )

        graph = StateGraph(MASState)
        for node in graph_nodes:
            graph.add_node(node, node_handlers[node])
        graph.set_entry_point(actual_entry)

        outgoing = {edge["source"] for edge in graph_edges}
        for edge in graph_edges:
            graph.add_edge(edge["source"], edge["target"])
        for node in graph_nodes:
            if node not in outgoing:
                graph.add_edge(node, END)

        compiled_graph = graph.compile()
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
