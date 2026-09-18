"""Thread-based message store and LangGraph checkpointer management.

Provides persistence, inspection, and serialization for conversation threads
(defaulting to 'thread-1') flowing through the adaptive multi-agent system.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver


def serialize_message(message: BaseMessage | Dict[str, Any]) -> Dict[str, Any]:
    """Serialize a LangChain BaseMessage into a JSON-serializable dictionary."""
    if isinstance(message, dict):
        return message

    msg_type = getattr(message, "type", message.__class__.__name__.lower())
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }
    role = role_map.get(msg_type, msg_type)

    data: Dict[str, Any] = {
        "type": msg_type,
        "role": role,
        "content": message.content,
    }
    if hasattr(message, "name") and message.name:
        data["name"] = message.name
    if isinstance(message, ToolMessage):
        data["tool_call_id"] = getattr(message, "tool_call_id", "")
    return data


def format_history_for_prompt(
    messages: Sequence[BaseMessage | Dict[str, Any]],
    max_messages: int = 12,
) -> str:
    """Format recent messages into a clean, concise conversational history string."""
    if not messages:
        return ""
    recent = messages[-max_messages:]
    lines = []
    for msg in recent:
        if isinstance(msg, dict):
            role = msg.get("name") or msg.get("role") or msg.get("type", "message")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "name", None) or getattr(msg, "type", "message")
            content = getattr(msg, "content", "")
        
        if not content:
            continue

        role_lower = str(role).lower()
        if role_lower in {"human", "user"}:
            speaker = "User"
        elif role_lower in {"finalizer", "assistant"} or (role_lower == "ai" and not getattr(msg, "name", None)):
            speaker = "Assistant"
        else:
            speaker = f"Agent ({role})"

        lines.append(f"{speaker}: {str(content).strip()}")
    return "\n".join(lines)


class ThreadMessageStore:
    """Thread-aware memory store backed by a LangGraph checkpointer."""

    DEFAULT_THREAD_ID: str = "thread-1"

    def __init__(self, checkpointer: Optional[BaseCheckpointSaver] = None) -> None:
        if checkpointer is None:
            try:
                from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
                self.checkpointer: BaseCheckpointSaver = MemorySaver(
                    serde=JsonPlusSerializer(allowed_msgpack_modules=True)
                )
            except Exception:
                self.checkpointer = MemorySaver()
        else:
            self.checkpointer = checkpointer

    def get_messages(self, thread_id: str = DEFAULT_THREAD_ID) -> List[BaseMessage]:
        """Retrieve all queued messages for a given thread_id."""
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = self.checkpointer.get(config)
        if not checkpoint:
            return []
        channel_values = checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        return list(messages)

    def get_serialized_messages(self, thread_id: str = DEFAULT_THREAD_ID) -> List[Dict[str, Any]]:
        """Retrieve all queued messages as serialized dicts."""
        return [serialize_message(m) for m in self.get_messages(thread_id)]

    def clear_thread(self, thread_id: str = DEFAULT_THREAD_ID) -> None:
        """Reset state for a given thread."""
        config = {"configurable": {"thread_id": thread_id}}
        if hasattr(self.checkpointer, "storage"):
            self.checkpointer.storage.pop(thread_id, None)


# Default singleton instance
default_thread_store = ThreadMessageStore()

