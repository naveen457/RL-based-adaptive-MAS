"""Architecture and Workflow Graph Visualizer.

Provides terminal ASCII and Mermaid diagram visualizers for adapted MAS architectures.
Can be rendered in terminals, logs, and Jupyter / IPynb notebooks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from langgraph.graph import END, START, StateGraph
from app.architecture.models import MASArchitecture
from app.graph.state import MASState


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


def render_invoked_flow_ascii(invoked_agents: List[str]) -> str:
    """Render the exact LangGraph ASCII graph for the agents that actually executed."""
    if not invoked_agents:
        return ""
    try:
        sg = StateGraph(MASState)
        for a in invoked_agents:
            sg.add_node(a, lambda s: s)

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
                    if has_critic:
                        sg.add_edge(w, "critic")
                    elif "finalizer" in invoked_agents:
                        sg.add_edge(w, "finalizer")
                if has_critic and "finalizer" in invoked_agents:
                    sg.add_edge("critic", "finalizer")
        elif invoked_agents:
            first_node = invoked_agents[0]
            sg.add_edge(START, first_node)
            for i in range(len(invoked_agents) - 1):
                sg.add_edge(invoked_agents[i], invoked_agents[i + 1])

        if "finalizer" in invoked_agents:
            sg.add_edge("finalizer", END)
        elif invoked_agents and "planner" not in invoked_agents:
            sg.add_edge(invoked_agents[-1], END)

        compiled = sg.compile()
        return compiled.get_graph().draw_ascii()
    except Exception:
        # Fallback to simple arrow flow
        lines = ["[START]"]
        for a in invoked_agents:
            lines.append(f"  |--> [{a}]")
        lines.append("  |--> [END]")
        return "\n".join(lines)


def format_architecture_display(
    architecture: MASArchitecture | Dict[str, Any],
    *,
    version: int = 0,
    actions_taken: Optional[List[Dict[str, Any]]] = None,
    cost_profile: Optional[Dict[str, Any]] = None,
    invoked_agents: Optional[List[str]] = None,
) -> str:
    """Format an end-to-end visual block showing the actual interacting agents."""
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

    out.append("\n  Actual Interaction Flow Graph:")
    if invoked_agents:
        ascii_graph = render_invoked_flow_ascii(actual_invoked)
    else:
        ascii_graph = render_langgraph_ascii(architecture)
    for line in ascii_graph.splitlines():
        out.append(f"    {line}")
    out.append("+" + "-" * 78 + "+")

    return "\n".join(out)
