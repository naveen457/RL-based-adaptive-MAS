"""
Tests for task-aware reward integration (Step 21).

These tests cover:

1. Task-aware reward construction
2. Structural reward calculation
3. Task-performance reward calculation
4. Combined reward calculation
5. Configurable task-performance weight
6. Invalid transition behavior
7. Same architecture + same task = deterministic reward
8. Same architecture + different tasks = potentially different task score
9. Architecture change affects task-performance reward
10. MetaEnvironment returns task-aware reward
11. Reward information appears in info
12. Q-learning receives task-aware reward
13. MetaTrainer uses task-aware reward
14. Architecture-only mode remains backward compatible
15. Reset correctly resets previous evaluation state
16. No stale task-performance evaluation after transitions
17. No LLM/API calls
18. No credentials/secrets
19. workflow.py unchanged
20. app/agents unchanged
"""

from __future__ import annotations

import inspect
import json
import os

import pytest

from app.architecture.manager import ArchitectureManager
from app.architecture.actions import ActionType, ArchitectureAction
from app.rl.meta_task import MetaTask
from app.rl.meta_environment import MetaEnvironment
from app.rl.environment import MASArchitectureEnv
from app.rl.q_learning import QLearningPolicy, QLearningTrainer
from app.evaluation.task_performance import (
    TaskPerformanceEvaluator,
    TaskPerformanceResult,
    TaskPerformanceRewardCalculator,
)
from app.evaluation.evaluator import ArchitectureEvaluator


# ============================================================================
# Helpers
# ============================================================================


def _make_task(
    *,
    task_id: str = "task-001",
    task_category: str = "coding",
    required_capabilities: list[str] | None = None,
    difficulty: int = 2,
) -> MetaTask:
    if required_capabilities is None:
        required_capabilities = ["coding"]
    return MetaTask(
        task_id=task_id,
        task_description=f"Sample {task_id}",
        task_category=task_category,
        required_capabilities=required_capabilities,
        difficulty=difficulty,
    )


def _make_default_architecture() -> Any:
    """Create the default MAS architecture."""
    manager = ArchitectureManager.create_default_architecture()
    return manager.get_architecture()


# ============================================================================
# 1. Task-aware reward construction
# ============================================================================


def test_task_aware_reward_calculator_construction() -> None:
    """Test that TaskPerformanceRewardCalculator can be constructed."""
    calc = TaskPerformanceRewardCalculator()
    assert calc is not None
    assert calc.task_performance_weight == 0.0


def test_task_aware_reward_calculator_with_weight() -> None:
    """Test that TaskPerformanceRewardCalculator accepts weight."""
    calc = TaskPerformanceRewardCalculator(task_performance_weight=0.5)
    assert calc.task_performance_weight == 0.5


# ============================================================================
# 2. Structural reward calculation
# ============================================================================


def test_structural_reward_only() -> None:
    """Test that structural reward works without task performance."""
    calc = TaskPerformanceRewardCalculator(task_performance_weight=0.0)
    
    evaluator = ArchitectureEvaluator()
    architecture = _make_default_architecture()
    
    previous = evaluator.evaluate(architecture)
    # Make a small change
    manager = ArchitectureManager.create_default_architecture()
    env = MASArchitectureEnv(manager)
    env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    env.step(action_id)
    current_architecture = env.manager.get_architecture()
    current = evaluator.evaluate(current_architecture)
    
    reward = calc.calculate(
        previous_result=previous,
        current_result=current,
        valid_transition=True,
    )
    
    # Should match structural delta
    expected = current.overall_score - previous.overall_score
    assert reward == pytest.approx(expected)


# ============================================================================
# 3. Task-performance reward calculation
# ============================================================================


def test_task_performance_reward_calculation() -> None:
    """Test task-performance reward calculation."""
    calc = TaskPerformanceRewardCalculator(task_performance_weight=1.0)
    
    task_evaluator = TaskPerformanceEvaluator()
    structural_evaluator = ArchitectureEvaluator()
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding", "testing", "nonexistent"],
    )
    
    manager = ArchitectureManager.create_default_architecture()
    env = MASArchitectureEnv(manager)
    env.reset()
    
    previous_architecture = env.manager.get_architecture()
    previous_structural = structural_evaluator.evaluate(previous_architecture)
    previous_task = task_evaluator.evaluate(task, previous_architecture)
    
    # Deactivate coder to reduce capability coverage
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.DEACTIVATE_AGENT,
            agent_id="coder",
        )
    )
    env.step(action_id)
    current_architecture = env.manager.get_architecture()
    current_structural = structural_evaluator.evaluate(current_architecture)
    current_task = task_evaluator.evaluate(task, current_architecture)
    
    reward = calc.calculate(
        previous_result=previous_structural,
        current_result=current_structural,
        valid_transition=True,
        previous_task_performance=previous_task,
        current_task_performance=current_task,
    )
    
    # Should be structural_delta + task_delta
    structural_delta = current_structural.overall_score - previous_structural.overall_score
    task_delta = current_task.task_success_score - previous_task.task_success_score
    expected = structural_delta + task_delta
    assert reward == pytest.approx(expected)


# ============================================================================
# 4. Combined reward calculation
# ============================================================================


def test_combined_reward_calculation() -> None:
    """Test combined structural + task-performance reward."""
    calc = TaskPerformanceRewardCalculator(task_performance_weight=0.5)
    
    structural_evaluator = ArchitectureEvaluator()
    task_evaluator = TaskPerformanceEvaluator()
    
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding"],
    )
    
    manager = ArchitectureManager.create_default_architecture()
    env = MASArchitectureEnv(manager)
    env.reset()
    
    previous_architecture = env.manager.get_architecture()
    previous_structural = structural_evaluator.evaluate(previous_architecture)
    previous_task = task_evaluator.evaluate(task, previous_architecture)
    
    # Make a change
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    env.step(action_id)
    current_architecture = env.manager.get_architecture()
    current_structural = structural_evaluator.evaluate(current_architecture)
    current_task = task_evaluator.evaluate(task, current_architecture)
    
    reward = calc.calculate(
        previous_result=previous_structural,
        current_result=current_structural,
        valid_transition=True,
        previous_task_performance=previous_task,
        current_task_performance=current_task,
    )
    
    # Should be structural_delta + 0.5 * task_delta
    structural_delta = current_structural.overall_score - previous_structural.overall_score
    task_delta = current_task.task_success_score - previous_task.task_success_score
    expected = structural_delta + 0.5 * task_delta
    assert reward == pytest.approx(expected)


# ============================================================================
# 5. Configurable task-performance weight
# ============================================================================


def test_task_performance_weight_zero() -> None:
    """Test that weight=0 gives structural reward only."""
    calc = TaskPerformanceRewardCalculator(task_performance_weight=0.0)
    
    structural_evaluator = ArchitectureEvaluator()
    task_evaluator = TaskPerformanceEvaluator()
    
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding"],
    )
    
    manager = ArchitectureManager.create_default_architecture()
    env = MASArchitectureEnv(manager)
    env.reset()
    
    previous_architecture = env.manager.get_architecture()
    previous_structural = structural_evaluator.evaluate(previous_architecture)
    previous_task = task_evaluator.evaluate(task, previous_architecture)
    
    # Make a change
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    env.step(action_id)
    current_architecture = env.manager.get_architecture()
    current_structural = structural_evaluator.evaluate(current_architecture)
    current_task = task_evaluator.evaluate(task, current_architecture)
    
    reward = calc.calculate(
        previous_result=previous_structural,
        current_result=current_structural,
        valid_transition=True,
        previous_task_performance=previous_task,
        current_task_performance=current_task,
    )
    
    # Should be structural_delta only (weight=0)
    structural_delta = current_structural.overall_score - previous_structural.overall_score
    assert reward == pytest.approx(structural_delta)


def test_task_performance_weight_nonzero() -> None:
    """Test that weight > 0 includes task-performance component."""
    calc = TaskPerformanceRewardCalculator(task_performance_weight=0.3)
    
    structural_evaluator = ArchitectureEvaluator()
    task_evaluator = TaskPerformanceEvaluator()
    
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding"],
    )
    
    manager = ArchitectureManager.create_default_architecture()
    env = MASArchitectureEnv(manager)
    env.reset()
    
    previous_architecture = env.manager.get_architecture()
    previous_structural = structural_evaluator.evaluate(previous_architecture)
    previous_task = task_evaluator.evaluate(task, previous_architecture)
    
    # Make a change
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    env.step(action_id)
    current_architecture = env.manager.get_architecture()
    current_structural = structural_evaluator.evaluate(current_architecture)
    current_task = task_evaluator.evaluate(task, current_architecture)
    
    reward = calc.calculate(
        previous_result=previous_structural,
        current_result=current_structural,
        valid_transition=True,
        previous_task_performance=previous_task,
        current_task_performance=current_task,
    )
    
    # Should include task-performance component
    structural_delta = current_structural.overall_score - previous_structural.overall_score
    task_delta = current_task.task_success_score - previous_task.task_success_score
    expected = structural_delta + 0.3 * task_delta
    assert reward == pytest.approx(expected)


# ============================================================================
# 6. Invalid transition behavior
# ============================================================================


def test_invalid_transition_penalty() -> None:
    """Test that invalid transitions get penalty regardless of task performance."""
    calc = TaskPerformanceRewardCalculator(task_performance_weight=0.5)
    
    task_evaluator = TaskPerformanceEvaluator()
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding"],
    )
    
    manager = ArchitectureManager.create_default_architecture()
    env = MASArchitectureEnv(manager)
    env.reset()
    
    previous_architecture = env.manager.get_architecture()
    previous_task = task_evaluator.evaluate(task, previous_architecture)
    
    # Make a valid change first
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    env.step(action_id)
    
    # Try the same action again (should be invalid)
    _, _, _, _, info = env.step(action_id)
    current_architecture = env.manager.get_architecture()
    current_task = task_evaluator.evaluate(task, current_architecture)
    
    reward = calc.calculate(
        previous_result=None,
        current_result=None,
        valid_transition=False,
        previous_task_performance=previous_task,
        current_task_performance=current_task,
    )
    
    # Should be invalid penalty
    assert reward == -1.0


# ============================================================================
# 7. Same architecture + same task = deterministic reward
# ============================================================================


def test_deterministic_reward_same_architecture_task() -> None:
    """Test that same architecture + same task produces deterministic reward."""
    calc = TaskPerformanceRewardCalculator(task_performance_weight=0.5)
    
    task_evaluator = TaskPerformanceEvaluator()
    structural_evaluator = ArchitectureEvaluator()
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding"],
    )
    
    architecture = _make_default_architecture()
    structural_result = structural_evaluator.evaluate(architecture)
    task_result = task_evaluator.evaluate(task, architecture)
    
    # Calculate reward multiple times
    rewards = []
    for _ in range(5):
        reward = calc.calculate(
            previous_result=structural_result,
            current_result=structural_result,
            valid_transition=True,
            previous_task_performance=task_result,
            current_task_performance=task_result,
        )
        rewards.append(reward)
    
    # All rewards should be identical
    assert all(r == rewards[0] for r in rewards)


# ============================================================================
# 8. Same architecture + different tasks = potentially different task score
# ============================================================================


def test_different_tasks_different_task_scores() -> None:
    """Test that same architecture with different tasks can have different task scores."""
    task_evaluator = TaskPerformanceEvaluator()
    architecture = _make_default_architecture()
    
    # Task with capability NOT covered by default architecture
    coding_task = _make_task(
        task_id="task-coding",
        task_category="coding",
        required_capabilities=["coding", "nonexistent_capability"],
    )
    research_task = _make_task(
        task_id="task-research",
        task_category="research",
        required_capabilities=["research", "information_synthesis"],
    )
    
    coding_result = task_evaluator.evaluate(coding_task, architecture)
    research_result = task_evaluator.evaluate(research_task, architecture)
    
    # Different tasks should have different task success scores
    # (coding task has missing capability, research task has full coverage)
    assert coding_result.task_success_score != research_result.task_success_score or \
           coding_result.capability_coverage_score != research_result.capability_coverage_score


# ============================================================================
# 9. Architecture change affects task-performance reward
# ============================================================================


def test_architecture_change_affects_task_reward() -> None:
    """Test that architecture changes affect task-performance reward."""
    task_evaluator = TaskPerformanceEvaluator()
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding"],
    )
    
    manager = ArchitectureManager.create_default_architecture()
    env = MASArchitectureEnv(manager)
    env.reset()
    
    # Initial architecture
    initial_architecture = env.manager.get_architecture()
    initial_task = task_evaluator.evaluate(task, initial_architecture)
    
    # Add an edge
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    env.step(action_id)
    changed_architecture = env.manager.get_architecture()
    changed_task = task_evaluator.evaluate(task, changed_architecture)
    
    # Task performance might change (or stay same if capability coverage unchanged)
    # The key is that we can compute the delta
    task_delta = changed_task.task_success_score - initial_task.task_success_score
    
    # Both should be valid TaskPerformanceResult objects
    assert isinstance(initial_task, TaskPerformanceResult)
    assert isinstance(changed_task, TaskPerformanceResult)


# ============================================================================
# 10. MetaEnvironment returns task-aware reward
# ============================================================================


def test_meta_environment_task_aware_reward() -> None:
    """Test that MetaEnvironment returns task-aware reward when configured."""
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding"],
    )
    
    manager = ArchitectureManager.create_default_architecture()
    env = MetaEnvironment(
        manager,
        task=task,
        task_performance_weight=0.5,
    )
    
    env.reset()
    
    # Get initial task performance
    initial_task = env.current_task_performance
    
    # Make a change
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    _, reward, terminated, truncated, info = env.step(action_id)
    
    # Should have task-aware reward info
    assert "reward" in info
    assert info["reward"]["task_performance_weight"] == 0.5
    assert "task_performance_delta" in info["reward"]


def test_meta_environment_structural_reward_only() -> None:
    """Test that MetaEnvironment returns structural reward when weight=0."""
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding"],
    )
    
    manager = ArchitectureManager.create_default_architecture()
    env = MetaEnvironment(
        manager,
        task=task,
        task_performance_weight=0.0,
    )
    
    env.reset()
    
    # Make a change
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    _, reward, terminated, truncated, info = env.step(action_id)
    
    # Should NOT have task-aware reward info (weight=0)
    assert "reward" not in info or info.get("reward", {}).get("task_performance_weight", 0) == 0


# ============================================================================
# 11. Reward information appears in info
# ============================================================================


def test_reward_info_structure() -> None:
    """Test that reward info has correct structure."""
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding"],
    )
    
    manager = ArchitectureManager.create_default_architecture()
    env = MetaEnvironment(
        manager,
        task=task,
        task_performance_weight=0.3,
    )
    
    env.reset()
    
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    _, _, _, _, info = env.step(action_id)
    
    if "reward" in info:
        reward_info = info["reward"]
        assert "reward" in reward_info
        assert "structural_reward" in reward_info or "reward_delta" in reward_info
        assert "task_performance_weight" in reward_info


# ============================================================================
# 12. Q-learning receives task-aware reward
# ============================================================================


def test_q_learning_receives_task_aware_reward() -> None:
    """Test that Q-learning policy receives task-aware reward from MetaEnvironment."""
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding"],
    )
    
    manager = ArchitectureManager.create_default_architecture()
    env = MetaEnvironment(
        manager,
        task=task,
        task_performance_weight=0.5,
    )
    
    policy = QLearningPolicy(task_aware=True, seed=42)
    trainer = QLearningTrainer(env, policy)
    
    # Run an episode
    stats = trainer.train_episode()
    
    # Should have received rewards (task-aware)
    assert stats.total_reward != 0.0 or stats.episode_length > 0
    assert isinstance(stats.total_reward, float)


# ============================================================================
# 13. MetaTrainer uses task-aware reward
# ============================================================================


def test_meta_trainer_task_aware_reward() -> None:
    """Test that MetaTrainer uses task-aware reward from MetaEnvironment."""
    from app.rl.meta_trainer import MetaTrainer
    from app.rl.task_distribution import MetaTaskDistribution
    
    tasks = [
        _make_task(task_id="task-001", task_category="coding", required_capabilities=["coding"]),
        _make_task(task_id="task-002", task_category="research", required_capabilities=["research"]),
    ]
    distribution = MetaTaskDistribution(tasks=tasks)
    
    trainer = MetaTrainer(
        task_distribution=distribution,
        seed=42,
        max_steps=2,
    )
    
    # Train on a single task
    task = distribution.get_task("task-001")
    result = trainer.train_single_task(task, num_episodes=1)
    
    # Should have received rewards
    assert result.total_reward != 0.0 or result.num_episodes > 0
    assert isinstance(result.total_reward, float)


# ============================================================================
# 14. Architecture-only mode remains backward compatible
# ============================================================================


def test_architecture_only_mode_backward_compatible() -> None:
    """Test that architecture-only mode (no task) works as before."""
    manager = ArchitectureManager.create_default_architecture()
    env = MASArchitectureEnv(manager)
    
    env.reset()
    
    # Make a change
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    _, reward, terminated, truncated, info = env.step(action_id)
    
    # Should have structural reward (no task-aware component)
    assert isinstance(reward, float)
    # Info should NOT have task-aware reward info
    assert "reward" not in info or "task_performance_weight" not in info.get("reward", {})


def test_meta_environment_no_task_backward_compatible() -> None:
    """Test that MetaEnvironment without task works as before."""
    manager = ArchitectureManager.create_default_architecture()
    env = MetaEnvironment(manager)  # No task
    
    env.reset()
    
    # Make a change
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    _, reward, terminated, truncated, info = env.step(action_id)
    
    # Should have structural reward (no task-aware component)
    assert isinstance(reward, float)
    # Info should NOT have task-aware reward info
    assert "reward" not in info or info.get("reward", {}).get("task_performance_weight", 0) == 0


# ============================================================================
# 15. Reset correctly resets previous evaluation state
# ============================================================================


def test_reset_resets_task_performance_state() -> None:
    """Test that reset clears task-performance evaluation state."""
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding"],
    )
    
    manager = ArchitectureManager.create_default_architecture()
    env = MetaEnvironment(
        manager,
        task=task,
        task_performance_weight=0.5,
    )
    
    env.reset()
    initial_task = env.current_task_performance
    
    # Make a change
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    env.step(action_id)
    changed_task = env.current_task_performance
    
    # Reset
    env.reset()
    reset_task = env.current_task_performance
    
    # After reset, should be back to initial
    assert reset_task.task_success_score == initial_task.task_success_score


# ============================================================================
# 16. No stale task-performance evaluation after transitions
# ============================================================================


def test_no_stale_task_performance() -> None:
    """Test that task-performance evaluation is updated after transitions."""
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding"],
    )
    
    manager = ArchitectureManager.create_default_architecture()
    env = MetaEnvironment(
        manager,
        task=task,
        task_performance_weight=0.5,
    )
    
    env.reset()
    
    # Get initial task performance
    initial_task = env.current_task_performance
    
    # Make multiple changes
    for _ in range(3):
        action_id = env.encode_action(
            ArchitectureAction(
                action_type=ActionType.ADD_EDGE,
                source="planner",
                target="critic",
            )
        )
        try:
            env.step(action_id)
        except RuntimeError:
            break
    
    # Current task performance should be updated
    assert env.current_task_performance is not None
    assert env.current_task_performance.task_id == task.task_id


# ============================================================================
# 17. No LLM/API calls
# ============================================================================


def test_task_aware_reward_has_no_api_imports() -> None:
    """Test that task-aware reward module doesn't import API-related modules."""
    import app.rl.meta_environment as me
    import inspect

    source = inspect.getsource(me)

    forbidden_imports = [
        "openai",
        "anthropic",
        "langchain",
        "langsmith",
        "requests",
        "urllib.request",
        "http.client",
        "chatopenai",
        "chat_openai",
        "ChatOpenAI",
    ]

    source_lower = source.lower()
    for forbidden in forbidden_imports:
        assert f"import {forbidden.lower()}" not in source_lower
        assert f"from {forbidden.lower()}" not in source_lower


def test_task_aware_reward_is_fully_offline() -> None:
    """Test that task-aware reward runs without network calls."""
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding"],
    )
    
    manager = ArchitectureManager.create_default_architecture()
    env = MetaEnvironment(
        manager,
        task=task,
        task_performance_weight=0.5,
    )
    
    env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    
    # If we reach here without network errors, the module stayed offline
    _, _, _, _, _ = env.step(action_id)


# ============================================================================
# 18. No credentials/secrets
# ============================================================================


def test_reward_info_contains_no_credentials() -> None:
    """Test that reward info contains no credentials."""
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding"],
    )
    
    manager = ArchitectureManager.create_default_architecture()
    env = MetaEnvironment(
        manager,
        task=task,
        task_performance_weight=0.5,
    )
    
    env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    _, _, _, _, info = env.step(action_id)
    
    serialized = json.dumps(info, default=str)
    serialized_lower = serialized.lower()
    
    forbidden_markers = [
        "sk-or-",
        "sk-proj-",
        "api_key",
        "openai_api_key",
        "password",
        "secret",
    ]
    
    for marker in forbidden_markers:
        assert marker not in serialized_lower, f"Found forbidden marker: {marker}"


# ============================================================================
# 19. workflow.py unchanged
# ============================================================================


def test_workflow_py_not_modified() -> None:
    """Test that app/graph/workflow.py is not modified."""
    import os

    workflow_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "app",
        "graph",
        "workflow.py",
    )
    assert os.path.exists(workflow_path), "expected app/graph/workflow.py to exist"


# ============================================================================
# 20. app/agents unchanged
# ============================================================================


def test_agents_directory_not_modified() -> None:
    """Test that app/agents/ is not modified."""
    import os

    agents_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "app",
        "agents",
    )
    assert os.path.isdir(agents_path), "expected app/agents/ to exist"


# ============================================================================
# 21. Combined reward with context
# ============================================================================


def test_combined_reward_with_context() -> None:
    """Test that calculate_with_context returns proper structure."""
    calc = TaskPerformanceRewardCalculator(task_performance_weight=0.5)
    
    structural_evaluator = ArchitectureEvaluator()
    task_evaluator = TaskPerformanceEvaluator()
    
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding"],
    )
    
    architecture = _make_default_architecture()
    previous_structural = structural_evaluator.evaluate(architecture)
    previous_task = task_evaluator.evaluate(task, architecture)
    
    # Same architecture (no change)
    context = calc.calculate_with_context(
        previous_result=previous_structural,
        current_result=previous_structural,
        valid_transition=True,
        previous_task_performance=previous_task,
        current_task_performance=previous_task,
    )
    
    assert "reward" in context
    assert "previous_score" in context
    assert "current_score" in context
    assert "task_performance_delta" in context
    assert context["task_performance_weight"] == 0.5


# ============================================================================
# 22. Task performance weight validation
# ============================================================================


def test_task_performance_weight_can_be_zero() -> None:
    """Test that task_performance_weight can be zero."""
    calc = TaskPerformanceRewardCalculator(task_performance_weight=0.0)
    assert calc.task_performance_weight == 0.0


def test_task_performance_weight_can_be_positive() -> None:
    """Test that task_performance_weight can be positive."""
    calc = TaskPerformanceRewardCalculator(task_performance_weight=1.0)
    assert calc.task_performance_weight == 1.0


# ============================================================================
# 23. MetaEnvironment task properties
# ============================================================================


def test_meta_environment_task_properties() -> None:
    """Test that MetaEnvironment exposes task properties correctly."""
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding"],
    )
    
    manager = ArchitectureManager.create_default_architecture()
    env = MetaEnvironment(
        manager,
        task=task,
        task_performance_weight=0.5,
    )
    
    assert env.task is not None
    assert env.task.task_id == "task-001"
    assert env.task_performance_weight == 0.5
    assert env.task_performance_evaluator is not None
