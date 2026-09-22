"""Dynamic LangGraph construction from an adapted architecture."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from app.agents.planner import PlannerOutput
from app.architecture.models import MASArchitecture
from app.config.settings import settings
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

    def _repr_mimebundle_(self, **kwargs: Any) -> dict[str, Any]:
        """Mime bundle used by Jupyter/IPython to display the graph image automatically."""
        if self.compiled_graph is not None and hasattr(self.compiled_graph, "_repr_mimebundle_"):
            try:
                return self.compiled_graph._repr_mimebundle_(**kwargs)
            except Exception:
                pass
        return {"text/plain": repr(self)}

    def _repr_png_(self) -> bytes | None:
        """PNG representation for Jupyter/IPython."""
        if self.compiled_graph is not None and hasattr(self.compiled_graph, "get_graph"):
            try:
                return self.compiled_graph.get_graph().draw_mermaid_png()
            except Exception:
                return None
        return None

    def get_graph(self):
        """Pass-through to compiled_graph.get_graph()."""
        if self.compiled_graph is not None and hasattr(self.compiled_graph, "get_graph"):
            return self.compiled_graph.get_graph()
        raise AttributeError("compiled_graph has no get_graph method")

    def draw_mermaid_png(self, output_file_path: Optional[str] = None, **kwargs) -> bytes:
        """Draw and optionally save the compiled graph as a Mermaid PNG."""
        if self.compiled_graph is not None and hasattr(self.compiled_graph, "get_graph"):
            return self.compiled_graph.get_graph().draw_mermaid_png(
                output_file_path=output_file_path, **kwargs
            )
        raise AttributeError("compiled_graph has no draw_mermaid_png method")

    def save_png(self, output_file_path: str = "runs/graph.png") -> str:
        """Save the compiled graph to a PNG image file and return its path."""
        from pathlib import Path
        p = Path(output_file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.draw_mermaid_png(output_file_path=str(p))
        return str(p.resolve())


class DynamicGraphBuilder:
    """Translate adapted architecture state into a compiled LangGraph graph."""

    def build(
        self,
        architecture: MASArchitecture,
        planner_output: PlannerOutput,
        node_handlers: Dict[str, NodeHandler],
        *,
        architecture_version: int = 0,
        architecture_actions: list[dict[str, Any]] | None = None,
        entry_point: Optional[str] = None,
        excluded_nodes: Optional[Set[str]] = None,
        include_finalizer: bool = True,
        checkpointer: Optional[Any] = None,
        save_image_path: Optional[str] = None,
    ) -> DynamicGraphBuildResult:
        active = set(architecture.active_agent_ids)
        required = {"planner"}
        if include_finalizer:
            required.add("finalizer")
        if planner_output.requires_coding:
            required.add("coder")
        if planner_output.requires_verification:
            required.add("critic")
        if getattr(planner_output, "requires_research", False) and "researcher" in active:
            required.add("researcher")
        has_tool_demand = (
            "tool_executor" in active
            or getattr(planner_output, "requires_tools", False)
            or (getattr(planner_output, "requires_research", False) and "researcher" not in active)
            or "tool_executor" in (getattr(planner_output, "selected_agents", []) or [])
            or "tool_executor" in (getattr(planner_output, "recommended_agents", []) or [])
            or bool(getattr(planner_output, "tools_needed", []))
            or bool(set(getattr(planner_output, "required_capabilities", []) or []) & {"tool_use", "web_search", "external_api", "tools"})
            or (bool(set(getattr(planner_output, "required_capabilities", []) or []) & {"research"}) and "researcher" not in active)
            or any(
                e.source == "tool_executor" or e.target == "tool_executor"
                for e in architecture.communication_edges
            )
        )
        if has_tool_demand and ("tool_executor" in active or "tool_executor" in node_handlers):
            required.add("tool_executor")

        raw_nodes = set(active & required)
        if has_tool_demand and ("tool_executor" in active or "tool_executor" in node_handlers):
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
                and (
                    "coder" in graph_nodes
                    or "tool_executor" in graph_nodes
                    or "critic" in graph_nodes
                )
            )
            and not (
                edge.source == "tool_executor"
                and edge.target == "finalizer"
                and "critic" in graph_nodes
            )
        ]
        if "critic" not in graph_nodes and "finalizer" in graph_nodes:
            for edge in architecture.communication_edges:
                if edge.target == "critic" and edge.source in graph_nodes:
                    bypass_edge = {"source": edge.source, "target": "finalizer"}
                    if bypass_edge not in configured_edges:
                        configured_edges.append(bypass_edge)

        # Ensure workers have valid incoming edges from planner and outgoing edges downstream
        worker_nodes = [n for n in graph_nodes if n not in ("planner", "finalizer", "critic")]
        for w in worker_nodes:
            if "planner" in graph_nodes and not any(e["source"] == "planner" and e["target"] == w for e in configured_edges):
                configured_edges.append({"source": "planner", "target": w})
            has_outgoing = any(e["source"] == w for e in configured_edges)
            if not has_outgoing and "finalizer" in graph_nodes:
                target_node = "critic" if "critic" in graph_nodes else "finalizer"
                configured_edges.append({"source": w, "target": target_node})

        # If critic is in graph_nodes, ensure critic connects to finalizer
        if "critic" in graph_nodes and "finalizer" in graph_nodes:
            if not any(e["source"] == "critic" and e["target"] == "finalizer" for e in configured_edges):
                configured_edges.append({"source": "critic", "target": "finalizer"})

        # If no worker nodes and no critic, ensure planner connects directly to finalizer
        if "planner" in graph_nodes and not worker_nodes and "critic" not in graph_nodes and "finalizer" in graph_nodes:
            if not any(e["source"] == "planner" and e["target"] == "finalizer" for e in configured_edges):
                configured_edges.append({"source": "planner", "target": "finalizer"})

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

        # Critic threshold feedback loop setup
        critic_retry_candidates = [
            n for n in ["tool_executor", "coder", "researcher"] if n in graph_nodes
        ]
        has_critic_loop = (
            "critic" in graph_nodes
            and len(critic_retry_candidates) > 0
        )
        threshold = getattr(settings, "critic_quality_threshold", 0.75)
        max_retries = getattr(settings, "critic_max_retries", 1)

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

        if has_critic_loop and "critic" in wrapped_handlers:
            orig_critic = wrapped_handlers["critic"]
            def _wrap_critic_handler(fn):
                def _wrapped(state: Any) -> Dict[str, Any]:
                    res = fn(state)
                    count = state.get("_feedback_iterations", 0) if isinstance(state, dict) else 0
                    critic_out = res.get("critic_output") if isinstance(res, dict) else None
                    score = getattr(critic_out, "quality_score", None) if critic_out else None
                    if score is None and isinstance(critic_out, dict):
                        score = critic_out.get("quality_score", 0.85)
                    elif score is None:
                        score = 0.85

                    if isinstance(res, dict):
                        if score < threshold:
                            res["_feedback_iterations"] = count + 1
                        else:
                            res["_feedback_iterations"] = count
                    return res
                return _wrapped
            wrapped_handlers["critic"] = _wrap_critic_handler(orig_critic)

        graph = StateGraph(ExtendedMASState)
        for node in graph_nodes:
            graph.add_node(node, wrapped_handlers[node])
        graph.set_entry_point(actual_entry)

        outgoing = {edge["source"] for edge in graph_edges}

        if has_critic_loop:
            default_target = "tool_executor" if "tool_executor" in critic_retry_candidates else critic_retry_candidates[0]
            forward_target = "finalizer" if "finalizer" in graph_nodes else END

            def _critic_router(state: Any) -> str:
                count = state.get("_feedback_iterations", 0) if isinstance(state, dict) else 0
                if count > max_retries:
                    return forward_target

                critic_out = state.get("critic_output") if isinstance(state, dict) else None
                if not critic_out:
                    return forward_target

                score = getattr(critic_out, "quality_score", None)
                if score is None and isinstance(critic_out, dict):
                    score = critic_out.get("quality_score", 0.85)
                elif score is None:
                    score = 0.85

                if score >= threshold:
                    return forward_target

                target = getattr(critic_out, "retry_target_node", None)
                if not target and isinstance(critic_out, dict):
                    target = critic_out.get("retry_target_node")

                if target and target in critic_retry_candidates:
                    return target
                return default_target

            destination_map = {tgt: tgt for tgt in critic_retry_candidates}
            if "finalizer" in graph_nodes:
                destination_map["finalizer"] = "finalizer"
            destination_map[END] = END

            graph.add_conditional_edges(
                "critic",
                _critic_router,
                destination_map,
            )
            outgoing.add("critic")

        for edge in graph_edges:
            src = edge["source"]
            tgt = edge["target"]
            if has_critic_loop and src == "critic" and tgt in ("finalizer", END):
                continue
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
        if save_image_path:
            try:
                from pathlib import Path
                p = Path(save_image_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                compiled_graph.get_graph().draw_mermaid_png(output_file_path=str(p))
            except Exception:
                pass

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
