"""Tests for conversation threading and message queuing (thread-1)."""

from unittest.mock import MagicMock
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.finalizer import FinalizerOutput
from app.agents.planner import PlannerOutput
from app.architecture.actions import ActionType, ArchitectureAction
from app.architecture.llm_adapter import LLMArchitectureAdapter
from app.architecture.manager import ArchitectureManager
from app.graph.adaptive_integration import AdaptiveWorkflowAdapter
from app.graph.dynamic_builder import DynamicGraphBuilder
from app.graph.state import ExtendedMASState
from app.memory.store import ThreadMessageStore, serialize_message
from app.runtime.llm_execution import ExistingLLMAgentExecutor
from app.runtime.orchestrator import AdaptiveRuntimeOrchestrator


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeDecisionLLM:
    def __init__(self, response):
        self.response = response

    def invoke(self, messages):
        return FakeResponse(self.response)


class FakePlanner:
    def __init__(self, output, client):
        self.output = output
        self.model = client

    def plan(self, task):
        return self.output


def _mock_executor():
    return ExistingLLMAgentExecutor(
        finalizer_factory=lambda: MagicMock(
            finalize=lambda original_task, supporting_info: FinalizerOutput(
                final_answer="Mocked final answer",
                key_points=["point 1"],
                limitations=[],
            )
        ),
        tool_executor_factory=lambda: MagicMock(
            execute=lambda name, args: MagicMock(serialize=lambda: {"result": "tool_val"}),
            registered_tools={"web_search", "calculator", "get_current_date"},
        ),
    )


def test_serialize_message():
    human = HumanMessage(content="Hello world")
    data = serialize_message(human)
    assert data["role"] == "user"
    assert data["content"] == "Hello world"
    assert "type" not in data
    assert "name" not in data

    ai = AIMessage(content="I am an agent", name="finalizer")
    data_ai = serialize_message(ai)
    assert data_ai["role"] == "assistant"
    assert data_ai["content"] == "I am an agent"
    assert "type" not in data_ai
    assert "name" not in data_ai


def test_thread_message_store_initialization():
    store = ThreadMessageStore(use_mongo=False)
    assert store.DEFAULT_THREAD_ID == "thread-1"
    msgs = store.get_messages("thread-1")
    assert msgs == []


def test_dynamic_graph_compiles_with_checkpointer():
    builder = DynamicGraphBuilder()
    manager = ArchitectureManager.create_default_architecture()
    plan = PlannerOutput(
        task_understanding="Simple task",
        required_capabilities=[],
        steps=["Step 1"],
        selected_agents=["planner", "finalizer"],
    )
    store = ThreadMessageStore()
    handlers = _mock_executor().node_handlers("Simple task", plan)

    result = builder.build(
        manager.get_architecture(),
        plan,
        handlers,
        checkpointer=store.checkpointer,
    )
    assert result.metadata.compiled is True
    assert result.compiled_graph is not None


def test_messages_queue_sequentially_on_thread_1():
    manager = ArchitectureManager.create_default_architecture()
    workflow = AdaptiveWorkflowAdapter(manager=manager)
    client = FakeDecisionLLM('{"decision": "no_change", "reasoning": "Baseline", "actions": []}')
    adapter = LLMArchitectureAdapter(client, workflow_adapter=workflow)

    plan = PlannerOutput(
        task_understanding="Say hello",
        required_capabilities=[],
        steps=["Respond politely"],
        selected_agents=["planner", "finalizer"],
    )

    thread_store = ThreadMessageStore(use_mongo=False)
    orchestrator = AdaptiveRuntimeOrchestrator(
        planner=FakePlanner(plan, client),
        architecture_adapter=adapter,
        agent_executor=_mock_executor(),
        thread_store=thread_store,
    )

    # Run Task 1 on thread-1
    res1 = orchestrator.run_dynamic("Hello from user", thread_id="thread-1")
    assert res1.thread_id == "thread-1"
    assert len(res1.messages) == 2  # HumanMessage + AIMessage(finalizer)
    assert res1.messages[0]["role"] == "user"
    assert res1.messages[0]["content"] == "Hello from user"
    assert res1.messages[1]["role"] == "assistant"
    assert res1.messages[1]["content"] == "Mocked final answer"

    # Run Task 2 on the same thread-1
    res2 = orchestrator.run_dynamic("How are you today?", thread_id="thread-1")
    assert res2.thread_id == "thread-1"
    # Thread memory accumulates clean dialogue messages across turns
    assert len(res2.messages) == 4
    contents = [m["content"] for m in res2.messages]
    assert "Hello from user" in contents
    assert "How are you today?" in contents


def test_different_threads_remain_isolated():
    manager = ArchitectureManager.create_default_architecture()
    workflow = AdaptiveWorkflowAdapter(manager=manager)
    client = FakeDecisionLLM('{"decision": "no_change", "reasoning": "Baseline", "actions": []}')
    adapter = LLMArchitectureAdapter(client, workflow_adapter=workflow)

    plan = PlannerOutput(
        task_understanding="Say hello",
        required_capabilities=[],
        steps=["Respond politely"],
        selected_agents=["planner", "finalizer"],
    )

    thread_store = ThreadMessageStore()
    orchestrator = AdaptiveRuntimeOrchestrator(
        planner=FakePlanner(plan, client),
        architecture_adapter=adapter,
        agent_executor=_mock_executor(),
        thread_store=thread_store,
    )

    # Run on thread-1
    res1 = orchestrator.run_dynamic("Task for thread 1", thread_id="thread-1")
    # Run on thread-2
    res2 = orchestrator.run_dynamic("Task for thread 2", thread_id="thread-2")

    thread1_contents = [m["content"] for m in res1.messages]
    thread2_contents = [m["content"] for m in res2.messages]

    assert "Task for thread 1" in thread1_contents
    assert "Task for thread 2" not in thread1_contents

    assert "Task for thread 2" in thread2_contents
    assert "Task for thread 1" not in thread2_contents


def test_format_history_for_prompt():
    from app.memory.store import format_history_for_prompt

    msgs = [
        HumanMessage(content="hi my name is naveen"),
        AIMessage(content="Hello Naveen!", name="finalizer"),
        AIMessage(content="Plan: do something", name="planner"),
    ]
    formatted = format_history_for_prompt(msgs)
    assert "User: hi my name is naveen" in formatted
    assert "Assistant: Hello Naveen!" in formatted
    # Internal planner scratchpad should be filtered out from prompt history
    assert "Plan: do something" not in formatted


def test_finalizer_and_planner_receive_conversation_history():
    from app.memory.store import format_history_for_prompt

    captured_finalizer_history = []
    captured_planner_history = []

    class HistoryTrackingPlanner:
        def __init__(self, output, client):
            self.output = output
            self.model = client

        def plan(self, task, conversation_history=None):
            captured_planner_history.append(conversation_history)
            return self.output

    mock_finalizer = MagicMock()
    def mock_finalize(original_task, supporting_info, conversation_history=None):
        captured_finalizer_history.append(conversation_history)
        return FinalizerOutput(
            final_answer="Answer with memory",
            key_points=[],
            limitations=[],
        )
    mock_finalizer.finalize = mock_finalize

    executor = ExistingLLMAgentExecutor(
        finalizer_factory=lambda: mock_finalizer,
    )

    manager = ArchitectureManager.create_default_architecture()
    workflow = AdaptiveWorkflowAdapter(manager=manager)
    client = FakeDecisionLLM('{"decision": "no_change", "reasoning": "Baseline", "actions": []}')
    adapter = LLMArchitectureAdapter(client, workflow_adapter=workflow)

    plan = PlannerOutput(
        task_understanding="Say hello",
        required_capabilities=[],
        steps=["Respond politely"],
        selected_agents=["planner", "finalizer"],
    )

    test_thread_id = "test-history-thread-clean"
    thread_store = ThreadMessageStore(use_mongo=False)

    orchestrator = AdaptiveRuntimeOrchestrator(
        planner=HistoryTrackingPlanner(plan, client),
        architecture_adapter=adapter,
        agent_executor=executor,
        thread_store=thread_store,
    )

    # Turn 1: user introduces name
    orchestrator.run_dynamic("hi my name is naveen", thread_id=test_thread_id)
    # Planner had no prior history on turn 1
    assert captured_planner_history[0] is None
    # Finalizer had the turn 1 message
    assert "User: hi my name is naveen" in captured_finalizer_history[0]

    # Turn 2: user asks for greeting by name
    orchestrator.run_dynamic("greet me with my name", thread_id=test_thread_id)
    # Planner now has the turn 1 history!
    assert captured_planner_history[1] is not None
    assert "hi my name is naveen" in captured_planner_history[1]
    # Finalizer now has turn 1 and turn 2 history!
    assert "hi my name is naveen" in captured_finalizer_history[1]
    assert "greet me with my name" in captured_finalizer_history[1]

    # Cleanup
    thread_store.clear_thread(test_thread_id)


def test_thread_store_add_and_list_threads():
    from app.memory.store import ThreadMessageStore, deserialize_message

    store = ThreadMessageStore(use_mongo=False)
    store.add_message("thread-alpha", HumanMessage(content="Alpha task"))
    store.add_message("thread-alpha", AIMessage(content="Alpha response", name="finalizer"))
    store.add_message("thread-beta", HumanMessage(content="Beta task"))

    threads = store.list_threads()
    assert "thread-alpha" in threads
    assert "thread-beta" in threads

    alpha_msgs = store.get_messages("thread-alpha")
    assert len(alpha_msgs) == 2
    assert alpha_msgs[0].content == "Alpha task"
    assert alpha_msgs[1].content == "Alpha response"

    stats_alpha = store.get_thread_stats("thread-alpha")
    assert stats_alpha["message_count"] == 2
    assert stats_alpha["human_messages"] == 1
    assert stats_alpha["ai_messages"] == 1

    # Verify clear thread
    store.clear_thread("thread-alpha")
    assert "thread-alpha" not in store.list_threads()


def test_thread_store_export_and_import(tmp_path):
    from app.memory.store import ThreadMessageStore

    store = ThreadMessageStore(use_mongo=False)
    store.add_message("thread-x", HumanMessage(content="Export me"))
    store.add_message("thread-x", AIMessage(content="Exported answer", name="finalizer"))

    export_file = tmp_path / "exported_thread.json"
    store.export_thread("thread-x", export_file)
    assert export_file.exists()

    new_store = ThreadMessageStore(use_mongo=False)
    imported_count = new_store.import_thread("thread-imported", export_file)
    assert imported_count == 2
    msgs = new_store.get_messages("thread-imported")
    assert len(msgs) == 2
    assert msgs[0].content == "Export me"


def test_thread_store_clear(tmp_path):
    from app.memory.store import ThreadMessageStore

    store = ThreadMessageStore(use_mongo=False)
    store.add_message("thread-temp", HumanMessage(content="Temporary"))
    assert len(store.get_messages("thread-temp")) == 1

    store.clear_thread("thread-temp")
    assert len(store.get_messages("thread-temp")) == 0


