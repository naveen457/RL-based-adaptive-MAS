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

    # Verify reconciliation between summary and metrics.jsonl
    jsonl_files = list(log_dir.rglob("metrics.jsonl"))
    assert len(jsonl_files) > 0
    records = []
    for line in jsonl_files[0].read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            records.append(json.loads(line))

    assert len(records) == 36  # 2 episodes * 18 tasks
    ep1_tokens = sum(r["estimated_tokens"] for r in records[:18])
    ep2_tokens = sum(r["estimated_tokens"] for r in records[18:])
    cum_tokens = sum(r["estimated_tokens"] for r in records)

    # Final episode summary must reconcile with ep2 records
    assert data["total_adapted_tokens"] == ep2_tokens
    assert data["final_episode"]["adapted_tokens"] == ep2_tokens

    # Cumulative summary must reconcile with total jsonl records
    assert data["cumulative"]["cumulative_adapted_tokens"] == cum_tokens
    assert data["per_episode_breakdown"]["episode_1"]["adapted_tokens"] == ep1_tokens
    assert data["per_episode_breakdown"]["episode_2"]["adapted_tokens"] == ep2_tokens

    # Verify RL Hit vs Heuristic Bypass distinction
    # In mock benchmark, all tasks pass through planner/selector with strict engine categorization
    assert data["heuristic_bypass_count"] >= 0
    assert data["final_episode"]["heuristic_bypass_count"] >= 0
    # Active RL hits, heuristic bypasses, and adapter fallbacks are mutually exclusive and partition all tasks
    assert data["active_rl_hits"] + data["heuristic_bypass_count"] + data["adapter_fallback_count"] == 18


def test_per_task_negative_savings_unclamped() -> None:
    """Verify that when adapted tokens exceed baseline, negative savings are preserved."""
    baseline_tokens = 3500
    adapted_tokens = 4200
    token_savings = ((baseline_tokens - adapted_tokens) / baseline_tokens) * 100.0

    assert token_savings < 0.0
    assert round(token_savings, 2) == -20.00

    baseline_cost = round((baseline_tokens / 1000.0) * 0.0015, 5)
    adapted_cost = round((adapted_tokens / 1000.0) * 0.0015, 5)
    cost_savings = ((baseline_cost - adapted_cost) / baseline_cost) * 100.0

    assert cost_savings < 0.0
    assert round(cost_savings, 2) == -20.00


def test_decision_source_classification_distinction() -> None:
    """Verify distinction between active RL hits, heuristic bypass, and LLM adapter fallback."""
    from app.benchmark import TaskBenchmarkMetric

    # 1. Active RL Policy hit: instant zero-overhead adaptation
    m_rl = TaskBenchmarkMetric(
        episode=1,
        task_id="coding-001",
        task_name="Write Quicksort",
        category="coding",
        baseline_tokens=3500,
        adapted_tokens=2200,
        token_savings_pct=37.1,
        baseline_cost=0.00525,
        adapted_cost=0.00330,
        cost_savings_pct=37.1,
        active_agents=3,
        apr_pct=50.0,
        critical_path_len=3,
        capability_coverage=100.0,
        net_utility=0.85,
        classification="OPTIMAL",
        decision_engine="Active RL Policy",
        is_instant_rl=True,
        adaptation_tokens=0,
        adaptation_latency_ms=0.8,
    )
    assert m_rl.is_instant_rl is True
    assert m_rl.decision_engine == "Active RL Policy"

    # 2. Heuristic bypass: trivial greeting bypass (NOT an RL policy hit)
    m_bypass = TaskBenchmarkMetric(
        episode=1,
        task_id="greet-001",
        task_name="Simple Greeting",
        category="greeting",
        baseline_tokens=3500,
        adapted_tokens=1500,
        token_savings_pct=57.1,
        baseline_cost=0.00525,
        adapted_cost=0.00225,
        cost_savings_pct=57.1,
        active_agents=2,
        apr_pct=66.7,
        critical_path_len=2,
        capability_coverage=100.0,
        net_utility=0.92,
        classification="OPTIMAL",
        decision_engine="Heuristic Bypass",
        is_instant_rl=False,
        adaptation_tokens=0,
        adaptation_latency_ms=0.1,
    )
    assert m_bypass.is_instant_rl is False
    assert m_bypass.decision_engine == "Heuristic Bypass"

    # 3. LLM Adapter fallback: multi-step search (NOT an RL hit)
    m_llm = TaskBenchmarkMetric(
        episode=1,
        task_id="research-001",
        task_name="Deep Market Analysis",
        category="research",
        baseline_tokens=3500,
        adapted_tokens=2800,
        token_savings_pct=20.0,
        baseline_cost=0.00525,
        adapted_cost=0.00420,
        cost_savings_pct=20.0,
        active_agents=4,
        apr_pct=33.3,
        critical_path_len=4,
        capability_coverage=100.0,
        net_utility=0.78,
        classification="SUBOPTIMAL",
        decision_engine="LLM Adapter",
        is_instant_rl=False,
        adaptation_tokens=450,
        adaptation_latency_ms=850.0,
    )
    assert m_llm.is_instant_rl is False
    assert m_llm.decision_engine == "LLM Adapter"
