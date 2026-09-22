"""Architecture and Workflow Graph Visualizer.

Provides terminal ASCII and Mermaid diagram visualizers for adapted MAS architectures.
Can be rendered in terminals, logs, and Jupyter / IPynb notebooks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from langgraph.graph import END, START, StateGraph
from app.architecture.models import MASArchitecture
from app.graph.state import ExtendedMASState, MASState


def _extract_arch_data(
    architecture: MASArchitecture | Dict[str, Any],
) -> tuple[str, list[str], set[str], list[tuple[str, str]], dict[str, str]]:
    """Extract architecture id, agents, active agents, edges, and role map."""
    if isinstance(architecture, MASArchitecture):
        arch_id = architecture.architecture_id
        all_agents = [a.agent_id for a in architecture.agents]
        active_agents = set(architecture.active_agent_ids)
        edges = [(e.source, e.target) for e in architecture.communication_edges]
        role_map = architecture.role_map
    else:
        arch_id = architecture.get("architecture_id", "dynamic-mas")
        agents_raw = architecture.get("agents", [])
        all_agents = [a.get("agent_id") if isinstance(a, dict) else a.agent_id for a in agents_raw]
        active_raw = architecture.get("active_agent_ids")
        if active_raw is not None:
            active_agents = set(active_raw)
        else:
            active_agents = {
                a.get("agent_id") for a in agents_raw if isinstance(a, dict) and a.get("active", True)
            }
        edges_raw = architecture.get("communication_edges", [])
        edges = [
            (e.get("source"), e.get("target")) if isinstance(e, dict) else (e.source, e.target)
            for e in edges_raw
        ]
        role_map = {
            a.get("agent_id"): a.get("role", "") for a in agents_raw if isinstance(a, dict)
        }

    try:
        from app.agents.registry import default_registry
        reg_ids = default_registry.list_agent_ids()
        for rid in reg_ids:
            if rid not in all_agents:
                all_agents.append(rid)
            if rid not in role_map:
                spec = default_registry.get(rid)
                if spec:
                    role_map[rid] = spec.role
    except Exception:
        pass

    return arch_id, all_agents, active_agents, edges, role_map


def render_langgraph_ascii(architecture: MASArchitecture | Dict[str, Any]) -> str:
    """Render a dynamic LangGraph ASCII graph of the active architecture."""
    _, _, active_agents, edges, _ = _extract_arch_data(architecture)

    active_edges = [(s, t) for s, t in edges if s in active_agents and t in active_agents]

    try:
        sg = StateGraph(MASState)
        for agent_id in active_agents:
            sg.add_node(agent_id, lambda s: s)

        # Wire entry point
        if "planner" in active_agents:
            sg.add_edge(START, "planner")
        elif active_agents:
            first_node = sorted(list(active_agents))[0]
            sg.add_edge(START, first_node)

        # Wire internal active communication edges
        for src, dst in active_edges:
            sg.add_edge(src, dst)

        # Wire exit point
        if "finalizer" in active_agents:
            sg.add_edge("finalizer", END)
        elif active_agents:
            # Nodes with no outbound edges connect to END
            sources = {s for s, _ in active_edges}
            terminals = active_agents - sources
            for term in terminals:
                sg.add_edge(term, END)

        compiled = sg.compile()
        return compiled.get_graph().draw_ascii()
    except Exception:
        # Fallback to structured ASCII DAG
        return render_dag_ascii(architecture)


def render_dag_ascii(architecture: MASArchitecture | Dict[str, Any]) -> str:
    """Render a structured ASCII diagram showing flow and agent roles."""
    arch_id, all_agents, active_agents, edges, role_map = _extract_arch_data(architecture)
    active_edges = [(s, t) for s, t in edges if s in active_agents and t in active_agents]

    lines = []
    lines.append("   [START]")
    lines.append("      |")

    # Group downstream edges by source
    outgoing: Dict[str, List[str]] = {a: [] for a in active_agents}
    for src, dst in active_edges:
        if src in outgoing:
            outgoing[src].append(dst)

    # Order nodes topologically or by typical pipeline flow
    order = ["planner", "researcher", "tool_executor", "coder", "critic", "finalizer"]
    sorted_active = [n for n in order if n in active_agents] + [
        n for n in active_agents if n not in order
    ]

    for node in sorted_active:
        role = f" ({role_map.get(node, '')})" if role_map.get(node) else ""
        lines.append(f"   [{node}]{role}")
        targets = outgoing.get(node, [])
        if targets:
            for t in targets:
                t_role = f" ({role_map.get(t, '')})" if role_map.get(t) else ""
                lines.append(f"      |--> [{t}]{t_role}")
        elif node != "finalizer":
            lines.append("      |")

    if "finalizer" in active_agents:
        lines.append("      |")
        lines.append("   [END]")

    return "\n".join(lines)


def render_mermaid_graph(architecture: MASArchitecture | Dict[str, Any]) -> str:
    """Generate Mermaid graph syntax for Markdown documents and Jupyter Notebooks."""
    arch_id, all_agents, active_agents, edges, role_map = _extract_arch_data(architecture)
    active_edges = [(s, t) for s, t in edges if s in active_agents and t in active_agents]

    lines = ["graph TD"]
    lines.append("  START([Start]) --> planner")

    for agent in sorted(list(active_agents)):
        role = role_map.get(agent, "")
        label = f"{agent} ({role})" if role else agent
        lines.append(f'  {agent}["{label}"]')

    for src, dst in active_edges:
        lines.append(f"  {src} --> {dst}")

    if "finalizer" in active_agents:
        lines.append("  finalizer --> END([End])")

    return "\n".join(lines)


def render_invoked_flow_ascii(
    invoked_agents: List[str],
    tools_executed: Optional[List[str]] = None,
) -> str:
    """Render the exact LangGraph ASCII graph for the agents that actually executed."""
    if not invoked_agents:
        return ""
    try:
        sg = StateGraph(ExtendedMASState)
        for a in invoked_agents:
            sg.add_node(a, lambda s: s)

        clean_tools = [t for t in (tools_executed or []) if t]
        for t in clean_tools:
            sg.add_node(t, lambda s: s)

        if "planner" in invoked_agents:
            sg.add_edge(START, "planner")

            intermediaries = [a for a in invoked_agents if a not in ("planner", "finalizer")]
            if not intermediaries and "finalizer" in invoked_agents:
                sg.add_edge("planner", "finalizer")
            else:
                has_critic = "critic" in intermediaries
                workers = [a for a in intermediaries if a != "critic"]
                if not workers and has_critic:
                    sg.add_edge("planner", "critic")
                for w in workers:
                    sg.add_edge("planner", w)
                    next_node = "critic" if has_critic else ("finalizer" if "finalizer" in invoked_agents else None)
                    if w == "tool_executor" and clean_tools:
                        prev = "tool_executor"
                        for t in clean_tools:
                            sg.add_edge(prev, t)
                            prev = t
                        if next_node:
                            sg.add_edge(prev, next_node)
                    else:
                        if next_node:
                            sg.add_edge(w, next_node)
                if has_critic and workers:
                    def _mock_ascii_router(s):
                        return "finalizer" if "finalizer" in invoked_agents else END
                    destinations = {w: w for w in workers}
                    if "finalizer" in invoked_agents:
                        destinations["finalizer"] = "finalizer"
                    destinations[END] = END
                    sg.add_conditional_edges("critic", _mock_ascii_router, destinations)
                elif has_critic and "finalizer" in invoked_agents:
                    sg.add_edge("critic", "finalizer")
        elif invoked_agents:
            first_node = invoked_agents[0]
            sg.add_edge(START, first_node)
            for i in range(len(invoked_agents) - 1):
                curr = invoked_agents[i]
                nxt = invoked_agents[i + 1]
                if curr == "tool_executor" and clean_tools:
                    prev = "tool_executor"
                    for t in clean_tools:
                        sg.add_edge(prev, t)
                        prev = t
                    sg.add_edge(prev, nxt)
                else:
                    sg.add_edge(curr, nxt)

        if "finalizer" in invoked_agents:
            sg.add_edge("finalizer", END)
        elif invoked_agents and "planner" not in invoked_agents:
            last_node = clean_tools[-1] if (clean_tools and invoked_agents[-1] == "tool_executor") else invoked_agents[-1]
            sg.add_edge(last_node, END)

        compiled = sg.compile()
        raw_ascii = compiled.get_graph().draw_ascii()
        if "critic" in invoked_agents:
            worker_candidates = [a for a in invoked_agents if a not in ("planner", "critic", "finalizer")]
            if worker_candidates:
                return _format_ascii_with_feedback_loop(raw_ascii, worker_candidates[0], "critic")
        return raw_ascii
    except Exception:
        # Fallback to simple arrow flow
        lines = ["[START]"]
        for a in invoked_agents:
            lines.append(f"  |--> [{a}]")
            if a == "tool_executor" and tools_executed:
                for t in tools_executed:
                    lines.append(f"       |--> [{t}]")
        lines.append("  |--> [END]")
        return "\n".join(lines)
 
 
def _format_ascii_with_feedback_loop(raw_ascii: str, worker_name: str, critic_name: str = "critic") -> str:
    """Decorate LangGraph ASCII output with visual feedback loop arrows from critic back to worker."""
    if not raw_ascii:
        return raw_ascii
    lines = raw_ascii.splitlines()
    worker_idx = None
    critic_idx = None
    for i, line in enumerate(lines):
        if f"| {worker_name} |" in line or f"|{worker_name}|" in line:
            worker_idx = i
        elif f"| {critic_name} |" in line or f"|{critic_name}|" in line:
            critic_idx = i

    if worker_idx is None or critic_idx is None or worker_idx >= critic_idx:
        return raw_ascii

    loop_start_line = max(0, worker_idx - 1)
    loop_end_line = critic_idx

    new_lines = []
    for i, line in enumerate(lines):
        if i == loop_start_line:
            new_lines.append(f"+-->  {line.lstrip()}")
        elif loop_start_line < i < loop_end_line:
            new_lines.append(f"|     {line.lstrip()}")
        elif i == loop_end_line:
            new_lines.append(f"+---  {line.lstrip()}  <--(Feedback Cycle: retry on review)")
        else:
            new_lines.append(f"      {line.lstrip()}")

    return "\n".join(new_lines)


def render_mermaid_png(
    architecture: MASArchitecture | Dict[str, Any],
    output_file_path: Optional[str] = None,
) -> bytes:
    """Render the active architecture as a visual LangGraph Mermaid PNG image."""
    _, _, active_agents, edges, _ = _extract_arch_data(architecture)
    active_edges = [(s, t) for s, t in edges if s in active_agents and t in active_agents]

    sg = StateGraph(ExtendedMASState)
    for agent_id in active_agents:
        sg.add_node(agent_id, lambda s: s)

    if "planner" in active_agents:
        sg.add_edge(START, "planner")
    elif active_agents:
        first_node = sorted(list(active_agents))[0]
        sg.add_edge(START, first_node)

    for src, dst in active_edges:
        sg.add_edge(src, dst)

    if "finalizer" in active_agents:
        sg.add_edge("finalizer", END)
    elif active_agents:
        sources = {s for s, _ in active_edges}
        terminals = active_agents - sources
        for term in terminals:
            sg.add_edge(term, END)

    compiled = sg.compile()
    return compiled.get_graph().draw_mermaid_png(output_file_path=output_file_path)


def render_invoked_flow_png(
    invoked_agents: List[str],
    output_file_path: Optional[str] = None,
    tools_executed: Optional[List[str]] = None,
) -> bytes:
    """Render the exact interacting agents and executed tools as a visual LangGraph Mermaid PNG image."""
    if not invoked_agents:
        return b""

    sg = StateGraph(ExtendedMASState)
    for a in invoked_agents:
        sg.add_node(a, lambda s: s)

    clean_tools = [t for t in (tools_executed or []) if t]
    for t in clean_tools:
        sg.add_node(t, lambda s: s)

    if "planner" in invoked_agents:
        sg.add_edge(START, "planner")
        intermediaries = [a for a in invoked_agents if a not in ("planner", "finalizer")]
        if not intermediaries and "finalizer" in invoked_agents:
            sg.add_edge("planner", "finalizer")
        else:
            has_critic = "critic" in intermediaries
            workers = [a for a in intermediaries if a != "critic"]
            if not workers and has_critic:
                sg.add_edge("planner", "critic")
            for w in workers:
                sg.add_edge("planner", w)
                next_node = "critic" if has_critic else ("finalizer" if "finalizer" in invoked_agents else None)
                if w == "tool_executor" and clean_tools:
                    prev = "tool_executor"
                    for t in clean_tools:
                        sg.add_edge(prev, t)
                        prev = t
                    if next_node:
                        sg.add_edge(prev, next_node)
                else:
                    if next_node:
                        sg.add_edge(w, next_node)
            if has_critic and workers:
                def _mock_png_router(s):
                    return "finalizer" if "finalizer" in invoked_agents else END
                destinations = {w: w for w in workers}
                if "finalizer" in invoked_agents:
                    destinations["finalizer"] = "finalizer"
                destinations[END] = END
                sg.add_conditional_edges("critic", _mock_png_router, destinations)
            elif has_critic and "finalizer" in invoked_agents:
                sg.add_edge("critic", "finalizer")
    elif invoked_agents:
        first_node = invoked_agents[0]
        sg.add_edge(START, first_node)
        for i in range(len(invoked_agents) - 1):
            curr = invoked_agents[i]
            nxt = invoked_agents[i + 1]
            if curr == "tool_executor" and clean_tools:
                prev = "tool_executor"
                for t in clean_tools:
                    sg.add_edge(prev, t)
                    prev = t
                sg.add_edge(prev, nxt)
            else:
                sg.add_edge(curr, nxt)

    if "finalizer" in invoked_agents:
        sg.add_edge("finalizer", END)
    elif invoked_agents and "planner" not in invoked_agents:
        last_node = clean_tools[-1] if (clean_tools and invoked_agents[-1] == "tool_executor") else invoked_agents[-1]
        sg.add_edge(last_node, END)

    compiled = sg.compile()
    return compiled.get_graph().draw_mermaid_png(output_file_path=output_file_path)


def save_graph_image(
    graph_or_arch: Any,
    output_path: str = "runs/graph.png",
    tools_executed: Optional[List[str]] = None,
) -> Optional[str]:
    """Save a visual PNG diagram of the compiled graph, architecture, or agent list.

    Returns the absolute path to the saved image, or None if failed.
    """
    from pathlib import Path

    try:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out_str = str(p.resolve())

        if hasattr(graph_or_arch, "get_graph"):
            graph_or_arch.get_graph().draw_mermaid_png(output_file_path=out_str)
            return out_str
        elif hasattr(graph_or_arch, "draw_mermaid_png"):
            graph_or_arch.draw_mermaid_png(output_file_path=out_str)
            return out_str
        elif isinstance(graph_or_arch, (list, tuple)):
            render_invoked_flow_png(list(graph_or_arch), output_file_path=out_str, tools_executed=tools_executed)
            return out_str
        else:
            render_mermaid_png(graph_or_arch, output_file_path=out_str)
            return out_str
    except Exception:
        return None


def display_graph(
    graph_or_arch: Any,
    tools_executed: Optional[List[str]] = None,
) -> Any:
    """Display the graph image inline in Jupyter/IPython, or save to runs/graph.png."""
    try:
        from IPython.display import Image, display  # type: ignore

        if hasattr(graph_or_arch, "get_graph"):
            png_data = graph_or_arch.get_graph().draw_mermaid_png()
        elif hasattr(graph_or_arch, "draw_mermaid_png"):
            png_data = graph_or_arch.draw_mermaid_png()
        elif isinstance(graph_or_arch, (list, tuple)):
            png_data = render_invoked_flow_png(list(graph_or_arch), tools_executed=tools_executed)
        else:
            png_data = render_mermaid_png(graph_or_arch)
        return display(Image(png_data))
    except Exception:
        path = save_graph_image(graph_or_arch, "runs/graph.png", tools_executed=tools_executed)
        if path:
            print(f"Graph image saved to: {path}")
        return path


def format_architecture_display(
    architecture: MASArchitecture | Dict[str, Any],
    *,
    version: int = 0,
    actions_taken: Optional[List[Dict[str, Any]]] = None,
    cost_profile: Optional[Dict[str, Any]] = None,
    invoked_agents: Optional[List[str]] = None,
    tools_executed: Optional[List[str]] = None,
) -> str:
    """Format an end-to-end visual block showing the actual interacting agents and executed tools."""
    arch_id, all_agents, active_agents, edges, role_map = _extract_arch_data(architecture)
    
    # Actual interacting agents
    actual_invoked = [a for a in (invoked_agents or []) if a in all_agents]
    if not actual_invoked:
        actual_invoked = sorted(list(active_agents))
    bypassed_agents = [a for a in all_agents if a not in actual_invoked]

    # Dynamic tool annotation from registry
    try:
        from app.agents.registry import default_registry
    except Exception:
        default_registry = None

    invoked_formatted = []
    for a in actual_invoked:
        if a == "tool_executor" and tools_executed:
            invoked_formatted.append(f"{a} [executed: {', '.join(tools_executed)}]")
        else:
            spec = default_registry.get(a) if default_registry else None
            if spec and spec.tools:
                tool_names = ", ".join(t.tool_name for t in spec.tools)
                invoked_formatted.append(f"{a} [tool: {tool_names}]")
            else:
                invoked_formatted.append(a)

    bypassed_formatted = []
    for a in sorted(bypassed_agents):
        spec = default_registry.get(a) if default_registry else None
        if spec and spec.tools:
            tool_names = ", ".join(t.tool_name for t in spec.tools)
            bypassed_formatted.append(f"{a} [tools: {tool_names}]")
        else:
            bypassed_formatted.append(a)

    out = []
    out.append("+" + "-" * 78 + "+")
    out.append(f"|  TASK AGENT INTERACTION FLOW: {arch_id.ljust(47)} |")
    out.append("+" + "-" * 78 + "+")
    out.append(f"  * Interacting Agents ({len(actual_invoked)}): {', '.join(invoked_formatted)}")
    if bypassed_agents:
        out.append(f"  * Bypassed Agents ({len(bypassed_agents)}):    {', '.join(bypassed_formatted)} (not needed for this task)")
    out.append(f"  * Architecture Version:  v{version}")
    
    if actions_taken:
        out.append(f"  * RL Adaptation Actions ({len(actions_taken)}):")
        for act in actions_taken:
            atype = act.get("action_type") or act.get("type", "")
            target = act.get("agent_id") or f"{act.get('source')} -> {act.get('target')}"
            out.append(f"      [x] {atype}: {target}")

    if cost_profile:
        tokens = cost_profile.get("estimated_tokens", 0)
        cost = cost_profile.get("estimated_cost_usd", 0.0)
        cls_name = cost_profile.get("classification", "optimal").upper()
        out.append(f"  * Profile: {tokens:,} est. tokens | ${cost:.5f} est. cost | Status: {cls_name}")

    workers_invoked = [a for a in actual_invoked if a not in ("planner", "critic", "finalizer")]
    if "critic" in actual_invoked and workers_invoked:
        out.append(f"  * Feedback Cycles Active: critic <--(retry on low score)--> {', '.join(workers_invoked)}")

    # Automatically generate and save visual graph image with executed tools
    saved_img_path = None
    try:
        target_graph_input = actual_invoked if invoked_agents else architecture
        saved_img_path = save_graph_image(
            target_graph_input,
            output_path="runs/latest_graph.png",
            tools_executed=tools_executed,
        )
        if saved_img_path:
            save_graph_image(
                target_graph_input,
                output_path="runs/graph.png",
                tools_executed=tools_executed,
            )
    except Exception:
        pass

    if saved_img_path:
        out.append(f"  * Visual Graph Image:    {saved_img_path}")

    out.append("\n  Actual Interaction Flow Graph:")
    if invoked_agents:
        ascii_graph = render_invoked_flow_ascii(actual_invoked, tools_executed=tools_executed)
    else:
        ascii_graph = render_langgraph_ascii(architecture)
    for line in ascii_graph.splitlines():
        out.append(f"    {line}")
    out.append("+" + "-" * 78 + "+")

    return "\n".join(out)
