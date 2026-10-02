import uuid
from app.evaluation.metrics_logger import MetricsLogger
from app.architecture.manager import ArchitectureManager

def test_user_metrics_isolation(tmp_path):
    logger = MetricsLogger(log_base_dir=str(tmp_path), session_id="test_session")
    arch = ArchitectureManager.create_default_architecture().get_architecture()

    user_a = f"user_alpha_{uuid.uuid4().hex[:8]}"
    user_b = f"user_beta_{uuid.uuid4().hex[:8]}"

    # User A runs 2 tasks
    logger.log_task_run(
        step=1,
        task="Write a python binary search",
        result={"agents_actually_invoked": ["planner", "coder", "finalizer"], "architecture_version": 1},
        architecture=arch,
        required_capabilities=["coding"],
        user_id=user_a,
        thread_id="thread_a1",
    )
    logger.log_task_run(
        step=2,
        task="Optimize quicksort algorithm",
        result={"agents_actually_invoked": ["planner", "coder", "critic", "finalizer"], "architecture_version": 1},
        architecture=arch,
        required_capabilities=["coding"],
        user_id=user_a,
        thread_id="thread_a2",
    )

    # User B runs 1 task
    logger.log_task_run(
        step=3,
        task="Explain quantum entanglement in physics",
        result={"agents_actually_invoked": ["planner", "researcher", "finalizer"], "architecture_version": 2},
        architecture=arch,
        required_capabilities=["research"],
        user_id=user_b,
        thread_id="thread_b1",
    )

    # Query metrics for User A
    metrics_a = logger.get_user_metrics(user_id=user_a)
    assert metrics_a["user_id"] == user_a
    assert metrics_a["summary"]["total_runs"] == 2
    assert metrics_a["agent_invocations"]["coder"] == 2
    assert metrics_a["agent_invocations"]["researcher"] == 0
    assert len(metrics_a["runs"]) == 2

    # Query metrics for User B
    metrics_b = logger.get_user_metrics(user_id=user_b)
    assert metrics_b["user_id"] == user_b
    assert metrics_b["summary"]["total_runs"] == 1
    assert metrics_b["agent_invocations"]["coder"] == 0
    assert metrics_b["agent_invocations"]["researcher"] == 1
    assert len(metrics_b["runs"]) == 1

    # Thread filtering for User A
    metrics_a_thread1 = logger.get_user_metrics(user_id=user_a, thread_id="thread_a1")
    assert metrics_a_thread1["summary"]["total_runs"] == 1
    assert metrics_a_thread1["runs"][0]["task"].startswith("Write a python")

    # Non-existent user
    metrics_empty = logger.get_user_metrics(user_id="user_nonexistent")
    assert metrics_empty["summary"]["total_runs"] == 0
    assert len(metrics_empty["runs"]) == 0

    # Verify user_comparison based strictly on user tasks
    assert "user_comparison" in metrics_a
    assert metrics_a["user_comparison"]["baseline"]["tokens"] == 2 * 3500
    assert len(metrics_a["user_comparison"]["metrics"]) == 5
    assert len(metrics_a["runs"]) == 2
    assert "static_tokens" in metrics_a["runs"][0]

    print("SUCCESS: User metrics per-user isolation and comparative graphs verified perfectly!")
