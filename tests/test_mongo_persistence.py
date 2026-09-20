"""Tests for MongoDB Atlas persistence and internet disconnection error handling."""

import pytest
from unittest.mock import MagicMock, patch
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from app.memory.mongo_client import check_mongo_network_error, get_mongo_client
from app.memory.store import ThreadMessageStore
from app.rl.q_learning import QLearningPolicy
from app.evaluation.metrics_logger import MetricsLogger
from langchain_core.messages import HumanMessage, AIMessage


def test_thread_store_mongo_persistence(tmp_path):
    """Verify ThreadMessageStore interacts directly with MongoDB collection."""
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = {
        "thread_id": "test-thread",
        "messages": [
            {"role": "user", "content": "Hello Mongo"},
        ],
    }
    mock_collection.distinct.return_value = ["test-thread", "remote-thread"]

    store = ThreadMessageStore()
    store.threads_collection = mock_collection

    # Test loading from mongo
    msgs = store._load_thread("test-thread")
    assert len(msgs) == 1
    assert msgs[0].content == "Hello Mongo"

    # Test saving to mongo
    store.add_message("test-thread", AIMessage(content="Hello back"))
    assert mock_collection.update_one.called

    # Test listing threads with mongo
    threads = store.list_threads()
    assert "remote-thread" in threads

    # Test clear thread with mongo
    store.clear_thread("test-thread")
    assert mock_collection.delete_one.called


def test_q_learning_mongo_save_and_load():
    """Verify Q-learning table saves to and loads from MongoDB Atlas."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    policy = QLearningPolicy(task_aware=False)
    state_key = ((1, 0, 0), (1, 0, 0), ("planner",), ())
    policy.q_table.set(state_key, action_id=1, value=4.5)

    with patch("app.memory.mongo_client.get_mongo_db", return_value=mock_db):
        # Test saving
        saved = policy.save_to_mongodb("test_policy")
        assert saved is True
        assert mock_collection.update_one.called

        # Test loading
        mock_collection.find_one.return_value = {
            "policy_id": "test_policy",
            "default_value": 0.0,
            "entries": [
                {
                    "state_key": [[1, 0, 0], [1, 0, 0], ["planner"], []],
                    "action_id": 1,
                    "q_value": 8.5,
                }
            ],
        }

        loaded = policy.load_from_mongodb("test_policy")
        assert loaded is True
        assert policy.q_table.get(state_key, action_id=1) == 8.5


def test_metrics_logger_mongo_logging(tmp_path):
    """Verify MetricsLogger logs task records to MongoDB Atlas."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    with patch("app.memory.mongo_client.get_mongo_db", return_value=mock_db):
        logger = MetricsLogger(log_base_dir=str(tmp_path / "runs"))
        assert logger.runs_collection is not None

        # Log a simulated task run
        sample_arch = {
            "architecture_id": "arch_test",
            "version": 0,
            "agents": [
                {"agent_id": "planner", "role": "planner", "description": "Planner node", "active": True, "capabilities": ["planning"]},
                {"agent_id": "finalizer", "role": "finalizer", "description": "Finalizer node", "active": True, "capabilities": ["synthesis"]},
            ],
            "communication_edges": [{"source": "planner", "target": "finalizer"}],
        }
        res = logger.log_task_run(
            step=1,
            task="Test task",
            result={"final_response": "done"},
            architecture=sample_arch,
            required_capabilities=["planning"],
        )

        assert mock_collection.insert_one.called
        assert res["task"] == "Test task"
        logger.close()


def test_check_mongo_network_error_raises_internet_status():
    """Verify check_mongo_network_error raises ConnectionError with exact status."""
    err = ConnectionFailure("timed out connecting to server")
    with pytest.raises(ConnectionError) as excinfo:
        check_mongo_network_error(err)
    assert "Status: Please check your internet connection." in str(excinfo.value)

    err2 = ServerSelectionTimeoutError("No replica set members found")
    with pytest.raises(ConnectionError) as excinfo2:
        check_mongo_network_error(err2)
    assert "Status: Please check your internet connection." in str(excinfo2.value)


def test_mongo_client_required_failure_exits(capsys):
    """Verify get_mongo_client(required=True) prints 'Unable to configure MongoDB' and exits."""
    with patch("app.memory.mongo_client._mongo_client", None):
        with patch("app.config.settings.settings.mongodb_uri", ""):
            with pytest.raises(SystemExit) as excinfo:
                get_mongo_client(required=True)
            assert excinfo.value.code == 1
            captured = capsys.readouterr()
            assert "Unable to configure MongoDB" in captured.out
