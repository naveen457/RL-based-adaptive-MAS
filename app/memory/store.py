"""Thread-based message store and LangGraph checkpointer management.

Provides persistence, inspection, and serialization for conversation threads
(defaulting to 'thread-1') flowing through the adaptive multi-agent system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

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



def deserialize_message(data: BaseMessage | Dict[str, Any]) -> BaseMessage:
    """Reconstruct a LangChain BaseMessage from a serialized dict or BaseMessage."""
    if isinstance(data, BaseMessage):
        return data

    msg_type = str(data.get("type", "")).lower()
    role = str(data.get("role", "")).lower()
    content = data.get("content", "")
    name = data.get("name")

    if msg_type in {"human", "user"} or role in {"human", "user"}:
        return HumanMessage(content=content, name=name) if name else HumanMessage(content=content)
    elif msg_type in {"ai", "assistant"} or role in {"ai", "assistant"}:
        return AIMessage(content=content, name=name) if name else AIMessage(content=content)
    elif msg_type in {"system"} or role in {"system"}:
        return SystemMessage(content=content, name=name) if name else SystemMessage(content=content)
    elif msg_type in {"tool"} or role in {"tool"}:
        tool_call_id = data.get("tool_call_id", "call_default")
        return ToolMessage(content=content, tool_call_id=tool_call_id, name=name)
    else:
        return HumanMessage(content=content, name=name) if name else HumanMessage(content=content)


def get_allowed_msgpack_modules(
    extra_modules: Optional[Sequence[tuple[str, str]]] = None,
) -> List[tuple[str, str]]:
    """Dynamically discover state and agent output model classes for msgpack serialization.

    Loops through all agent modules and model files dynamically so newly added agents,
    tools, and state extensions are automatically registered without hardcoded lists.
    """
    import importlib
    import inspect
    import pkgutil
    from pydantic import BaseModel

    allowed: set[tuple[str, str]] = set()

    # Dynamic loop 1: Automatically walk and inspect all modules in app.agents
    try:
        import app.agents
        for _, modname, _ in pkgutil.walk_packages(app.agents.__path__, app.agents.__name__ + "."):
            try:
                mod = importlib.import_module(modname)
                for name, obj in inspect.getmembers(mod, inspect.isclass):
                    if issubclass(obj, BaseModel) and obj.__module__ == modname:
                        allowed.add((modname, name))
            except Exception:
                continue
    except Exception:
        pass

    # Dynamic loop 2: Automatically scan related state and architecture model packages
    for modname in ("app.graph.state", "app.architecture.models"):
        try:
            mod = importlib.import_module(modname)
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if issubclass(obj, BaseModel) and obj.__module__ == modname:
                    allowed.add((modname, name))
        except Exception:
            continue

    if extra_modules:
        for item in extra_modules:
            allowed.add(item)

    return sorted(list(allowed))


class ThreadMessageStore:
    """Thread-aware memory store providing multi-turn conversation persistence per thread."""

    DEFAULT_THREAD_ID: str = "thread-1"

    def __init__(
        self,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        storage_dir: Optional[str | Path] = None,
        allowed_msgpack_modules: Optional[Sequence[tuple[str, str]]] = None,
    ) -> None:
        if checkpointer is None:
            try:
                from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
                dynamic_modules = get_allowed_msgpack_modules(extra_modules=allowed_msgpack_modules)
                self.checkpointer: BaseCheckpointSaver = MemorySaver(
                    serde=JsonPlusSerializer(allowed_msgpack_modules=dynamic_modules)
                )
            except Exception:
                self.checkpointer = MemorySaver()
        else:
            self.checkpointer = checkpointer

        self.storage_dir: Optional[Path] = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory thread cache: thread_id -> List[BaseMessage]
        self._thread_cache: Dict[str, List[BaseMessage]] = {}

        # Preload any existing persisted threads if storage_dir is given
        if self.storage_dir and self.storage_dir.exists():
            for json_file in self.storage_dir.glob("*.json"):
                thread_name = json_file.stem
                self._load_from_disk(thread_name)

    def _disk_path(self, thread_id: str) -> Optional[Path]:
        """Return file path for a given thread."""
        if not self.storage_dir:
            return None
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in thread_id)
        return self.storage_dir / f"{safe_id}.json"

    def _save_to_disk(self, thread_id: str) -> None:
        """Persist serialized messages for thread_id to disk."""
        target_path = self._disk_path(thread_id)
        if not target_path:
            return
        msgs = self._thread_cache.get(thread_id, [])
        serialized = [serialize_message(m) for m in msgs]
        payload = {
            "thread_id": thread_id,
            "message_count": len(serialized),
            "messages": serialized,
        }
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _load_from_disk(self, thread_id: str) -> List[BaseMessage]:
        """Load messages for thread_id from disk into cache."""
        target_path = self._disk_path(thread_id)
        if not target_path or not target_path.exists():
            return []
        try:
            import json
            with open(target_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            raw_msgs = payload.get("messages", [])
            messages = [deserialize_message(m) for m in raw_msgs]
            self._thread_cache[thread_id] = messages
            return messages
        except Exception:
            return []

    def get_messages(self, thread_id: str = DEFAULT_THREAD_ID) -> List[BaseMessage]:
        """Retrieve all queued messages for a given thread_id."""
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = self.checkpointer.get(config)
        if checkpoint:
            channel_values = checkpoint.get("channel_values", {})
            cp_msgs = list(channel_values.get("messages", []))
            if cp_msgs:
                self._thread_cache[thread_id] = cp_msgs
                if self.storage_dir:
                    self._save_to_disk(thread_id)
                return cp_msgs

        if thread_id in self._thread_cache:
            return list(self._thread_cache[thread_id])

        # Try disk fallback
        loaded = self._load_from_disk(thread_id)
        if loaded:
            return loaded

        return []

    def get_serialized_messages(self, thread_id: str = DEFAULT_THREAD_ID) -> List[Dict[str, Any]]:
        """Retrieve all queued messages as serialized dicts."""
        return [serialize_message(m) for m in self.get_messages(thread_id)]

    def add_message(self, thread_id: str, message: BaseMessage | Dict[str, Any]) -> None:
        """Append a message to the specified thread history."""
        self.add_messages(thread_id, [message])

    def add_messages(
        self,
        thread_id: str,
        messages: Sequence[BaseMessage | Dict[str, Any]],
    ) -> None:
        """Append multiple messages to the specified thread history."""
        if not messages:
            return
        current = self.get_messages(thread_id)
        deserialized = [deserialize_message(m) for m in messages]
        updated = list(current) + deserialized
        self._thread_cache[thread_id] = updated
        if self.storage_dir:
            self._save_to_disk(thread_id)

    def sync_thread(
        self,
        thread_id: str,
        messages: Sequence[BaseMessage | Dict[str, Any]],
    ) -> None:
        """Synchronize current thread state with the latest message list."""
        deserialized = [deserialize_message(m) for m in messages]
        self._thread_cache[thread_id] = deserialized
        if self.storage_dir:
            self._save_to_disk(thread_id)

    def list_threads(self) -> List[str]:
        """List all thread IDs known in checkpointer, cache, or disk storage."""
        threads = set(self._thread_cache.keys())
        if hasattr(self.checkpointer, "storage") and isinstance(self.checkpointer.storage, dict):
            threads.update(self.checkpointer.storage.keys())
        if self.storage_dir and self.storage_dir.exists():
            threads.update(p.stem for p in self.storage_dir.glob("*.json"))
        if not threads:
            return [self.DEFAULT_THREAD_ID]
        return sorted(threads)

    def get_thread_stats(self, thread_id: str = DEFAULT_THREAD_ID) -> Dict[str, Any]:
        """Return summary statistics for a thread."""
        msgs = self.get_messages(thread_id)
        human_count = sum(1 for m in msgs if getattr(m, "type", "") in {"human", "user"})
        ai_count = sum(1 for m in msgs if getattr(m, "type", "") in {"ai", "assistant"})
        return {
            "thread_id": thread_id,
            "message_count": len(msgs),
            "human_messages": human_count,
            "ai_messages": ai_count,
            "has_history": len(msgs) > 0,
        }

    def format_history(
        self,
        thread_id: str = DEFAULT_THREAD_ID,
        max_messages: int = 12,
    ) -> str:
        """Retrieve formatted conversational history string for prompts."""
        return format_history_for_prompt(self.get_messages(thread_id), max_messages=max_messages)

    def clear_thread(self, thread_id: str = DEFAULT_THREAD_ID) -> None:
        """Reset state and delete messages for a given thread."""
        config = {"configurable": {"thread_id": thread_id}}
        if hasattr(self.checkpointer, "storage") and isinstance(self.checkpointer.storage, dict):
            self.checkpointer.storage.pop(thread_id, None)
        self._thread_cache.pop(thread_id, None)
        disk_path = self._disk_path(thread_id)
        if disk_path and disk_path.exists():
            try:
                disk_path.unlink()
            except Exception:
                pass

    def clear_all(self) -> None:
        """Clear all threads."""
        for tid in list(self.list_threads()):
            self.clear_thread(tid)

    def export_thread(self, thread_id: str, filepath: str | Path) -> Path:
        """Export serialized thread messages to a specified JSON file path."""
        import json
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "thread_id": thread_id,
            "messages": self.get_serialized_messages(thread_id),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return out_path

    def import_thread(self, thread_id: str, filepath: str | Path) -> int:
        """Import messages from a JSON file into a given thread."""
        import json
        in_path = Path(filepath)
        if not in_path.exists():
            raise FileNotFoundError(f"Cannot import from nonexistent file: {in_path}")
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_msgs = data.get("messages", [])
        msgs = [deserialize_message(m) for m in raw_msgs]
        self.add_messages(thread_id, msgs)
        return len(msgs)


# Default singleton instance (with local persistence directory data/threads)
default_thread_store = ThreadMessageStore(storage_dir="data/threads")


