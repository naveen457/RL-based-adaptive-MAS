"""Thread-based message store and LangGraph checkpointer management.

Provides persistence, inspection, and serialization for conversation threads
(defaulting to 'thread-1') flowing through the adaptive multi-agent system.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from app.memory.mongo_client import check_mongo_network_error, get_mongo_db

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


def is_dialogue_message(msg: BaseMessage | Dict[str, Any]) -> bool:
    """Return True if message is pure conversational dialogue, False for internal scratchpads."""
    if isinstance(msg, dict):
        name = str(msg.get("name", "")).lower()
        content = str(msg.get("content", "")).strip()
    else:
        name = str(getattr(msg, "name", None) or "").lower()
        content = str(getattr(msg, "content", "")).strip()

    if name in {"planner", "researcher", "coder", "critic", "tool_executor"}:
        return False
    if content.startswith("Plan:") or content.startswith("Tool Execution Results:") or content.startswith("Review Critique:"):
        return False
    return True


def serialize_message(message: BaseMessage | Dict[str, Any]) -> Dict[str, Any]:
    """Serialize a message into clean {'role': 'user'|'assistant', 'content': '...'} dictionary."""
    if isinstance(message, dict):
        raw_role = message.get("role") or message.get("type", "user")
        role_lower = str(raw_role).lower()
        role = "user" if role_lower in ("human", "user") else "assistant"
        return {
            "role": role,
            "content": str(message.get("content", "")),
        }

    msg_type = getattr(message, "type", message.__class__.__name__.lower())
    role = "user" if msg_type in ("human", "user") else "assistant"
    return {
        "role": role,
        "content": str(getattr(message, "content", "")),
    }


def format_history_for_prompt(
    messages: Sequence[BaseMessage | Dict[str, Any]],
    max_messages: int = 12,
) -> str:
    """Format recent dialogue into a clean, concise conversational history string."""
    if not messages:
        return ""
    clean_msgs = [m for m in messages if is_dialogue_message(m)]
    recent = clean_msgs[-max_messages:]
    lines = []
    for msg in recent:
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type", "user")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "type", "user")
            content = getattr(msg, "content", "")

        if not content:
            continue

        role_lower = str(role).lower()
        speaker = "User" if role_lower in {"human", "user"} else "Assistant"
        lines.append(f"{speaker}: {str(content).strip()}")
    return "\n".join(lines)


def deserialize_message(data: BaseMessage | Dict[str, Any]) -> BaseMessage:
    """Reconstruct a LangChain BaseMessage from a serialized dict or BaseMessage."""
    if isinstance(data, BaseMessage):
        return data

    role = str(data.get("role", "")).lower() or str(data.get("type", "")).lower()
    content = str(data.get("content", ""))

    if role in {"human", "user"}:
        return HumanMessage(content=content)
    return AIMessage(content=content)


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
        allowed_msgpack_modules: Optional[Sequence[tuple[str, str]]] = None,
        use_mongo: bool = True,
        storage_dir: Optional[Any] = None,
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

        self.storage_dir = None
        self.db = get_mongo_db(required=False) if use_mongo else None
        self.threads_collection = self.db["threads"] if self.db is not None else None
        self.users_collection = self.db["users"] if self.db is not None else None
        self.graphs_collection = self.db["graphs"] if self.db is not None else None

        # In-memory thread cache: thread_id -> List[BaseMessage]
        self._thread_cache: Dict[str, List[BaseMessage]] = {}
        # In-memory thread user map: thread_id -> user_id
        self._thread_user_map: Dict[str, str] = {}

    def _save_thread(self, thread_id: str, user_id: Optional[str] = None) -> None:
        """Persist serialized dialogue messages for thread_id directly to MongoDB Atlas under user_id."""
        msgs = [m for m in self._thread_cache.get(thread_id, []) if is_dialogue_message(m)]
        self._thread_cache[thread_id] = msgs
        serialized = [serialize_message(m) for m in msgs]

        resolved_user = user_id or self._thread_user_map.get(thread_id)
        if resolved_user:
            self._thread_user_map[thread_id] = str(resolved_user)

        payload = {
            "thread_id": thread_id,
            "message_count": len(serialized),
            "messages": serialized,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        if resolved_user:
            payload["user_id"] = str(resolved_user)

        if self.threads_collection is not None:
            try:
                # Prevent cross-user thread hijacking
                existing = self.threads_collection.find_one({"thread_id": thread_id})
                if existing and "user_id" in existing and resolved_user and str(existing["user_id"]) != str(resolved_user):
                    raise PermissionError(f"Access denied: Thread '{thread_id}' belongs to another user.")

                self.threads_collection.update_one(
                    {"thread_id": thread_id},
                    {"$set": payload},
                    upsert=True,
                )
            except Exception as e:
                check_mongo_network_error(e)

        # Mirror under users collection: adaptive_mas/users/<user_id>/threads/<thread_id>/messages
        if self.users_collection is not None and resolved_user:
            try:
                self.users_collection.update_one(
                    {"_id": str(resolved_user)},
                    {
                        "$set": {
                            "user_id": str(resolved_user),
                            f"threads.{thread_id}": payload,
                            "updated_at": payload["updated_at"],
                        }
                    },
                    upsert=True,
                )
            except Exception as e:
                check_mongo_network_error(e)

    _save_to_disk = _save_thread

    def _load_thread(self, thread_id: str, user_id: Optional[str] = None) -> List[BaseMessage]:
        """Load messages for thread_id directly from MongoDB Atlas, strictly scoped to user_id."""
        raw_msgs = []
        if self.threads_collection is not None:
            try:
                query = {"thread_id": thread_id}
                if user_id:
                    query["user_id"] = str(user_id)
                doc = self.threads_collection.find_one(query)
                if doc and "messages" in doc:
                    raw_msgs = doc["messages"]
                    if "user_id" in doc:
                        self._thread_user_map[thread_id] = str(doc["user_id"])
            except Exception as e:
                check_mongo_network_error(e)

        messages = [deserialize_message(m) for m in raw_msgs if is_dialogue_message(m)]
        if messages:
            self._thread_cache[thread_id] = messages
        return messages

    _load_from_disk = _load_thread

    def get_messages(self, thread_id: str = DEFAULT_THREAD_ID, user_id: Optional[str] = None) -> List[BaseMessage]:
        """Retrieve all queued messages for a given thread_id, ensuring user ownership."""
        if user_id:
            cached_user = self._thread_user_map.get(thread_id)
            if cached_user and cached_user != str(user_id):
                return []

        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = self.checkpointer.get(config)
        if checkpoint:
            channel_values = checkpoint.get("channel_values", {})
            cp_msgs = [m for m in channel_values.get("messages", []) if is_dialogue_message(m)]
            if cp_msgs:
                self._thread_cache[thread_id] = cp_msgs
                self._save_thread(thread_id, user_id=user_id)
                return cp_msgs

        if thread_id in self._thread_cache:
            if user_id and self._thread_user_map.get(thread_id) and self._thread_user_map[thread_id] != str(user_id):
                return []
            return list(self._thread_cache[thread_id])

        loaded = self._load_thread(thread_id, user_id=user_id)
        if loaded:
            return loaded

        return []

    def get_serialized_messages(self, thread_id: str = DEFAULT_THREAD_ID, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all queued messages as serialized dicts."""
        return [serialize_message(m) for m in self.get_messages(thread_id, user_id=user_id)]

    def add_message(self, thread_id: str, message: BaseMessage | Dict[str, Any], user_id: Optional[str] = None) -> None:
        """Append a message to the specified thread history."""
        self.add_messages(thread_id, [message], user_id=user_id)

    def add_messages(
        self,
        thread_id: str,
        messages: Sequence[BaseMessage | Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> None:
        """Append multiple messages to the specified thread history."""
        if not messages:
            return
        current = self.get_messages(thread_id, user_id=user_id)
        deserialized = [deserialize_message(m) for m in messages]
        updated = list(current) + deserialized
        self._thread_cache[thread_id] = updated
        self._save_thread(thread_id, user_id=user_id)

    def sync_thread(
        self,
        thread_id: str,
        messages: Sequence[BaseMessage | Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> None:
        """Synchronize current thread state with the latest message list while preserving prior history."""
        if not messages:
            return
        deserialized = [deserialize_message(m) for m in messages]
        current = self.get_messages(thread_id, user_id=user_id)

        if not current:
            self._thread_cache[thread_id] = deserialized
        else:
            first_curr_content = getattr(current[0], "content", None)
            first_new_content = getattr(deserialized[0], "content", None)
            if first_curr_content == first_new_content and len(deserialized) >= len(current):
                self._thread_cache[thread_id] = deserialized
            else:
                self._thread_cache[thread_id] = list(current) + deserialized

        self._save_thread(thread_id, user_id=user_id)

    def list_threads(self, user_id: Optional[str] = None) -> List[str]:
        """List all thread IDs strictly scoped to user_id."""
        if not user_id:
            return []

        if self.threads_collection is not None:
            try:
                matched = self.threads_collection.distinct("thread_id", {"user_id": str(user_id)})
                return sorted(matched)
            except Exception as e:
                check_mongo_network_error(e)

        return sorted([tid for tid, uid in self._thread_user_map.items() if uid == str(user_id)])

    def get_thread_stats(self, thread_id: str = DEFAULT_THREAD_ID, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Return summary statistics for a thread scoped to user_id."""
        msgs = self.get_messages(thread_id, user_id=user_id)
        human_count = sum(1 for m in msgs if getattr(m, "type", "") in {"human", "user"})
        ai_count = sum(1 for m in msgs if getattr(m, "type", "") in {"ai", "assistant"})
        return {
            "thread_id": thread_id,
            "message_count": len(msgs),
            "human_messages": human_count,
            "ai_messages": ai_count,
            "has_history": len(msgs) > 0,
        }

    def save_latest_graph(
        self,
        thread_id: str = DEFAULT_THREAD_ID,
        graph_png_base64: str = "",
        user_id: Optional[str] = None,
    ) -> None:
        """Store ONLY the latest graph PNG for this thread and user in MongoDB Atlas.

        Saves to dedicated 'graphs' collection where document _id is user_id,
        and each chat (SHA code) stores its own latest graph.
        """
        if not graph_png_base64:
            return

        resolved_user = user_id or self._thread_user_map.get(thread_id) or "default_user"
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 1. Primary storage: Dedicated 'graphs' collection (User ID -> Chat SHA -> latest PNG)
        if self.graphs_collection is not None:
            try:
                graph_entry = {
                    "thread_id": thread_id,
                    "graph": graph_png_base64,
                    "updated_at": now_iso,
                }
                self.graphs_collection.update_one(
                    {"_id": str(resolved_user)},
                    {
                        "$set": {
                            "user_id": str(resolved_user),
                            f"chats.{thread_id}": graph_entry,
                            f"threads.{thread_id}": graph_entry,
                            "updated_at": now_iso,
                        }
                    },
                    upsert=True,
                )
            except Exception as e:
                check_mongo_network_error(e)

        # 2. Secondary storage: update thread document in threads collection
        if self.threads_collection is not None:
            try:
                update_fields: Dict[str, Any] = {
                    "latest_graph": graph_png_base64,
                    "graph_updated_at": now_iso,
                }
                if resolved_user:
                    update_fields["user_id"] = str(resolved_user)

                self.threads_collection.update_one(
                    {"thread_id": thread_id},
                    {"$set": update_fields},
                    upsert=True,
                )
            except Exception as e:
                check_mongo_network_error(e)

    def get_latest_graph(
        self,
        thread_id: str = DEFAULT_THREAD_ID,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Retrieve the latest graph PNG base64 for a thread scoped to user_id.

        Checks the dedicated 'graphs' collection first, falling back to 'threads'.
        """
        resolved_user = user_id or self._thread_user_map.get(thread_id)

        # 1. Check dedicated 'graphs' collection by user_id
        if self.graphs_collection is not None and resolved_user:
            try:
                doc = self.graphs_collection.find_one({"_id": str(resolved_user)})
                if doc:
                    chat_data = doc.get("chats", {}).get(thread_id) or doc.get("threads", {}).get(thread_id)
                    if chat_data and "graph" in chat_data:
                        return chat_data["graph"]
            except Exception as e:
                check_mongo_network_error(e)

        # 2. Fallback to threads collection
        if self.threads_collection is not None:
            try:
                query: Dict[str, Any] = {"thread_id": thread_id}
                if user_id:
                    query["user_id"] = str(user_id)

                doc = self.threads_collection.find_one(query, {"latest_graph": 1, "user_id": 1})
                if doc and "latest_graph" in doc:
                    if user_id and doc.get("user_id") and str(doc["user_id"]) != str(user_id):
                        return None
                    return doc.get("latest_graph")
            except Exception as e:
                check_mongo_network_error(e)
        return None

    def format_history(
        self,
        thread_id: str = DEFAULT_THREAD_ID,
        max_messages: int = 12,
    ) -> str:
        """Retrieve formatted conversational history string for prompts."""
        return format_history_for_prompt(self.get_messages(thread_id), max_messages=max_messages)

    def clear_thread(self, thread_id: str = DEFAULT_THREAD_ID) -> None:
        """Reset state and delete messages for a given thread directly in MongoDB Atlas."""
        config = {"configurable": {"thread_id": thread_id}}
        if hasattr(self.checkpointer, "storage") and isinstance(self.checkpointer.storage, dict):
            self.checkpointer.storage.pop(thread_id, None)
        self._thread_cache.pop(thread_id, None)
        if self.threads_collection is not None:
            try:
                self.threads_collection.delete_one({"thread_id": thread_id})
            except Exception as e:
                check_mongo_network_error(e)

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


# Default singleton instance (backed directly by MongoDB Atlas)
default_thread_store = ThreadMessageStore()


