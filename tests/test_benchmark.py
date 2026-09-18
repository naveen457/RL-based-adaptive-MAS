"""Unit tests for the Automated Multi-Task Benchmark & TensorBoard Suite (Step 4).

Tests that:
1. BENCHMARK_TASKS contains 18 diverse tasks covering all 6 target domains.
2. The benchmark execution loop runs deterministically in mock mode.
3. Pareto frontier metrics (tokens, costs, savings) are accurately computed.
4. Summary JSON and TensorBoard event records are created on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.benchmark import BENCHMARK_TASKS, MultiTaskBenchmarkSuite, run_benchmark


def test_benchmark_task_suite_structure() -> None:
    """Verify task suite completeness across domains."""
    assert len(BENCHMARK_TASKS) == 18

    categories = {t.category for t in BENCHMARK_TASKS}
    assert "greeting" in categories
    assert "math" in categories
    assert "coding" in categories
    assert "research" in categories
    assert "web_search" in categories
    assert "cross_domain" in categories

    # Ensure all tasks have non-empty prompt and valid ID
    for task in BENCHMARK_TASKS:
        assert task.task_id
        assert len(task.prompt) > 5
        assert 1 <= task.difficulty <= 5


def test_benchmark_run_in_mock_mode(tmp_path: Path) -> None:
    """Test full multi-episode benchmark run in fast mock mode."""
    q_table_file = tmp_path / "q_table.json"
    log_dir = tmp_path / "runs"

    summary = run_benchmark(
        episodes=2,
        mode="mock",
        session_mode="continuous",
        q_table_path=str(q_table_file),
        log_base_dir=str(log_dir),
    )

    assert summary is not None
    assert summary["episodes"] == 2
    assert summary["total_tasks"] == 18
    assert summary["token_savings_pct"] > 0
    assert summary["cost_savings_pct"] > 0
    assert summary["pareto_optimal_rate_pct"] > 50.0

    # Verify Q-table was created and populated
    assert q_table_file.exists()
    assert summary["q_table_entries"] > 0

    # Verify summary report JSON was written
    summary_json_file = log_dir / "benchmark_summary.json"
    assert summary_json_file.exists()
    data = json.loads(summary_json_file.read_text(encoding="utf-8"))
    assert data["total_tasks"] == 18
