"""Central Dynamic Agent & Node Registry.

Maintains the available agent and node definitions in one place, allowing
LLMs, RL controllers, and runtime graph builders to query available nodes,
their roles, capabilities, and tools dynamically without hardcoded agent lists.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from app.agents.tool_executor import ToolExecutor


# ---------------------------------------------------------------------------
# Specifications
# ---------------------------------------------------------------------------

class ToolSpec(BaseModel):
    """Metadata describing a tool available to a tool-calling node."""

    tool_name: str = Field(description="Unique identifier for the tool.")
    description: str = Field(description="What the tool does and when to use it.")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Expected parameters / arguments for calling the tool."
    )


class AgentSpec(BaseModel):
    """Formal specification of an agent or node in the MAS."""

    agent_id: str = Field(description="Unique identifier for the agent (e.g. 'planner', 'tool_executor').")
    role: str = Field(description="Functional role in the system (e.g. 'planning', 'tool_execution').")
    description: str = Field(description="Human- and LLM-readable description of what the node does.")
    capabilities: List[str] = Field(default_factory=list, description="List of capability tags provided.")
    node_type: str = Field(
        default="cognitive",
        description="Category of node: 'cognitive' (LLM agent) or 'tool_executor' (tool calling)."
    )
    tools: List[ToolSpec] = Field(default_factory=list, description="Tools attached to this node if any.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata.")

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


# ---------------------------------------------------------------------------
# Agent Registry
# ---------------------------------------------------------------------------

class AgentRegistry:
    """Central registry of available MAS nodes and tools.
    
    Allows dynamic registration, inspection, and prompt generation for available agents.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, AgentSpec] = {}

    def register(self, spec: AgentSpec) -> None:
        """Register or update an agent specification."""
        self._agents[spec.agent_id] = spec

    def unregister(self, agent_id: str) -> Optional[AgentSpec]:
        """Remove an agent specification from the registry."""
        return self._agents.pop(agent_id, None)

    def get(self, agent_id: str) -> Optional[AgentSpec]:
        """Retrieve an agent specification by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> List[AgentSpec]:
        """Return all registered agent specifications in deterministic order."""
        return [self._agents[k] for k in sorted(self._agents.keys())]

    def list_agent_ids(self) -> List[str]:
        """Return all registered agent IDs in sorted order."""
        return sorted(list(self._agents.keys()))

    def get_tool_nodes(self) -> List[AgentSpec]:
        """Return all registered tool-calling nodes."""
        return [a for a in self.list_agents() if a.node_type == "tool_executor"]

    def get_cognitive_nodes(self) -> List[AgentSpec]:
        """Return all registered cognitive (LLM) agents."""
        return [a for a in self.list_agents() if a.node_type == "cognitive"]

    def to_prompt_summary(self) -> str:
        """Generate a rich, structured markdown summary of all registered agents and tools
        suitable for inclusion in LLM or Planner system prompts.
        """
        lines = ["### Available Agent Nodes in Registry:"]
        for agent in self.list_agents():
            node_tag = f"[{agent.node_type.upper()}]"
            lines.append(f"- **{agent.agent_id}** ({agent.role}) {node_tag}: {agent.description}")
            if agent.capabilities:
                lines.append(f"  * Capabilities: {', '.join(agent.capabilities)}")
            if agent.tools:
                lines.append(f"  * Tools available on this node:")
                for tool in agent.tools:
                    param_str = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
                    lines.append(f"    - `{tool.tool_name}({param_str})`: {tool.description}")
        return "\n".join(lines)

    @classmethod
    def create_default_registry(cls) -> AgentRegistry:
        """Instantiate the default registry containing core cognitive agents
        and the dedicated tool executor node with live search & calculator tools.
        """
        reg = cls()

        # 1. Planner
        reg.register(
            AgentSpec(
                agent_id="planner",
                role="planning",
                description="Decomposes tasks into structured steps, determines required capabilities, and routes workflow.",
                capabilities=["planning", "task_analysis", "routing"],
                node_type="cognitive",
            )
        )

        # 2. Coder
        reg.register(
            AgentSpec(
                agent_id="coder",
                role="implementation",
                description="Writes, explains, and refactors programming code across languages.",
                capabilities=["coding", "code_explanation", "testing"],
                node_type="cognitive",
            )
        )

        # 3. Critic
        reg.register(
            AgentSpec(
                agent_id="critic",
                role="verification",
                description="Reviews intermediate outputs for correctness, logic errors, and edge case coverage.",
                capabilities=["verification", "critique", "quality_assessment"],
                node_type="cognitive",
            )
        )

        # 4. Finalizer
        reg.register(
            AgentSpec(
                agent_id="finalizer",
                role="synthesis",
                description="Consolidates intermediate outputs into a clean final response.",
                capabilities=["synthesis", "summarization"],
                node_type="cognitive",
            )
        )

        # 5. Dedicated Tool Executor Node (contains live web search, arXiv paper search, and calculator)
        reg.register(
            AgentSpec(
                agent_id="tool_executor",
                role="tool_execution",
                description=(
                    "Dedicated execution node for external tools, live web search, and scientific paper retrieval. "
                    "MUST be selected whenever the task requires live web search, current/trending "
                    "events, arXiv research papers, preprints, or mathematical calculations."
                ),
                capabilities=["tool_use", "web_search", "research", "external_api"],
                node_type="tool_executor",
                tools=[
                    ToolSpec(
                        tool_name="web_search",
                        description=(
                            "Performs live web search (via Tavily Search API with DuckDuckGo fallback). "
                            "Essential for current events, trending topics, latest news, and real-time facts."
                        ),
                        parameters={"query": "string (search terms)", "max_results": "integer (optional, default 5)"},
                    ),
                    ToolSpec(
                        tool_name="arxiv_search",
                        description=(
                            "Searches arXiv for scientific research papers, academic preprints, authors, and abstracts. "
                            "Essential whenever the task asks for research papers, scientific literature, machine learning papers, or citations."
                        ),
                        parameters={"query": "string (keywords, topic, title, or author)", "max_results": "integer (optional, default 5)"},
                    ),
                    ToolSpec(
                        tool_name="calculator",
                        description="Evaluates mathematical and arithmetic expressions accurately.",
                        parameters={"expression": "string (e.g. '24 * 365 + 18')"},
                    ),
                    ToolSpec(
                        tool_name="get_current_date",
                        description=(
                            "Returns the current live system date, time, weekday, and ISO timestamp. "
                            "Essential whenever the task asks for today's date, current year/month/day, or live time."
                        ),
                        parameters={"tz_name": "string (optional, e.g. 'UTC')", "date_format": "string (optional, e.g. '%Y-%m-%d')"},
                    ),
                ],
            )
        )


        return reg



# Global default singleton instance
default_registry = AgentRegistry.create_default_registry()
