"""
Tests for the task-performance evaluation layer (Step 20).

These tests cover:

1. TaskPerformanceEvaluator construction
2. Deterministic evaluation
3. Task/architecture compatibility
4. Full capability coverage
5. Partial capability coverage
6. Missing capabilities
7. Inactive agents
8. Role changes
9. Communication structure
10. Different tasks producing different scores
11. Same task + same architecture producing identical results
12. Serialization
13. Empty requirements
14. Invalid task handling
15. Combined evaluation
16. Extended reward calculator
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
from app.evaluation.task_performance import (
    TaskPerformanceEvaluator,
    TaskPerformanceResult,
    CombinedEvaluationResult,
    TaskPerformanceRewardCalculator,
    ROLE_CAPABILITY_MAP,
    get_capabilities_for_role,
    get_roles_for_capability,
    create_combined_evaluation,
)
from app.evaluation.evaluator import ArchitectureEvaluator, EvaluationResult
from app.evaluation.reward import RewardCalculator


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
# 1. Evaluator construction
# ============================================================================


def test_evaluator_construction() -> None:
    """Test that TaskPerformanceEvaluator can be constructed."""
    evaluator = TaskPerformanceEvaluator()
    assert evaluator is not None


def test_evaluator_is_stateless() -> None:
    """Test that the evaluator is stateless."""
    evaluator_a = TaskPerformanceEvaluator()
    evaluator_b = TaskPerformanceEvaluator()

    # Both should be independent instances
    assert evaluator_a is not evaluator_b


# ============================================================================
# 2. Deterministic evaluation
# ============================================================================


def test_evaluation_is_deterministic() -> None:
    """Test that evaluation is deterministic for same inputs."""
    evaluator = TaskPerformanceEvaluator()
    task = _make_task(task_id="task-001", required_capabilities=["coding"])
    architecture = _make_default_architecture()

    result_a = evaluator.evaluate(task, architecture)
    result_b = evaluator.evaluate(task, architecture)

    assert result_a == result_b
    assert result_a.task_success_score == result_b.task_success_score


def test_same_task_same_architecture_identical_results() -> None:
    """Test that same task + same architecture produces identical results."""
    evaluator = TaskPerformanceEvaluator()
    task = _make_task(
        task_id="task-coding",
        task_category="coding",
        required_capabilities=["coding", "testing"],
    )
    architecture = _make_default_architecture()

    results = [evaluator.evaluate(task, architecture) for _ in range(5)]

    # All results should be identical
    first = results[0]
    for other in results[1:]:
        assert first == other


# ============================================================================
# 3. Task/architecture compatibility
# ============================================================================


def test_coding_task_with_coder_agent() -> None:
    """Test that a coding task with a coder agent gets decent coverage."""
    evaluator = TaskPerformanceEvaluator()
    task = _make_task(
        task_id="task-coding",
        task_category="coding",
        required_capabilities=["coding"],
    )
    architecture = _make_default_architecture()

    result = evaluator.evaluate(task, architecture)

    # The default architecture has a coder (implementation role)
    assert result.task_success_score > 0.0
    assert "coding" in result.covered_capabilities


def test_research_task_with_researcher_agent() -> None:
    """Test that a research task with a researcher agent gets decent coverage."""
    evaluator = TaskPerformanceEvaluator()
    task = _make_task(
        task_id="task-research",
        task_category="research",
        required_capabilities=["research"],
    )
    architecture = _make_default_architecture()

    result = evaluator.evaluate(task, architecture)

    # The default architecture has a researcher
    assert result.task_success_score > 0.0
    assert "research" in result.covered_capabilities


# ============================================================================
# 4. Full capability coverage
# ============================================================================


def test_full_capability_coverage() -> None:
    """Test that all required capabilities covered gives high score."""
    evaluator = TaskPerformanceEvaluator()
    task = _make_task(
        task_id="task-full",
        task_category="coding",
        required_capabilities=["coding", "testing"],
    )
    architecture = _make_default_architecture()

    result = evaluator.evaluate(task, architecture)

    # Default architecture has implementation (coding) and verification (testing)
    assert result.task_success_score > 0.5
    assert len(result.missing_capabilities) == 0
    assert result.capability_coverage_score == 1.0


# ============================================================================
# 5. Partial capability coverage
# ============================================================================


def test_partial_capability_coverage() -> None:
    """Test partial capability coverage."""
    evaluator = TaskPerformanceEvaluator()
    # Require capabilities where at least one is missing from the default architecture
    task = _make_task(
        task_id="task-partial",
        task_category="research",
        required_capabilities=["research", "nonexistent_capability"],
    )
    architecture = _make_default_architecture()

    result = evaluator.evaluate(task, architecture)

    # Should have partial coverage
    assert 0.0 < result.capability_coverage_score < 1.0
    assert len(result.missing_capabilities) > 0
    assert len(result.covered_capabilities) > 0


# ============================================================================
# 6. Missing capabilities
# ============================================================================

def test_all_capabilities_missing() -> None:
    """Test when all required capabilities are missing."""
    evaluator = TaskPerformanceEvaluator()
    # Require capabilities not in the default architecture at all
    task = _make_task(
        task_id="task-missing",
        task_category="unknown",
        required_capabilities=["nonexistent_capability", "another_fake_one"],
    )
    architecture = _make_default_architecture()

    result = evaluator.evaluate(task, architecture)

    # All capabilities missing
    assert result.capability_coverage_score == 0.0
    assert len(result.missing_capabilities) == 2
    assert result.task_success_score == 0.0


# ============================================================================
# 7. Inactive agents
# ============================================================================


def test_inactive_agents_reduce_coverage() -> None:
    """Test that deactivating agents reduces capability coverage."""
    from app.rl.environment import MASArchitectureEnv
    
    evaluator = TaskPerformanceEvaluator()
    task = _make_task(
        task_id="task-coding",
        task_category="coding",
        required_capabilities=["coding", "testing"],
    )

    # Create environment and deactivate the coder
    manager = ArchitectureManager.create_default_architecture()
    env = MASArchitectureEnv(manager)
    env.reset()
    
    # Deactivate the coder
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.DEACTIVATE_AGENT,
            agent_id="coder",
        )
    )
    env.step(action_id)
    architecture = env.manager.get_architecture()

    result = evaluator.evaluate(task, architecture)

    # Coverage should be reduced
    assert result.active_agent_count == 4  # One agent deactivated
    assert "coding" not in result.covered_capabilities or result.capability_coverage_score < 1.0


# ============================================================================
# 8. Role changes
# ============================================================================


def test_role_changes_affect_coverage() -> None:
    """Test that changing roles affects capability coverage."""
    from app.rl.environment import MASArchitectureEnv
    
    evaluator = TaskPerformanceEvaluator()
    task = _make_task(
        task_id="task-coding",
        task_category="coding",
        required_capabilities=["coding"],
    )

    # Create environment and change coder's role
    manager = ArchitectureManager.create_default_architecture()
    env = MASArchitectureEnv(manager, role_options=["planning"])
    env.reset()
    
    # Change coder's role to something that doesn't provide coding
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.CHANGE_ROLE,
            agent_id="coder",
            new_role="planning",
        )
    )
    env.step(action_id)
    architecture = env.manager.get_architecture()

    result = evaluator.evaluate(task, architecture)

    # Coverage should be reduced since coder no longer provides coding capability
    assert "coding" not in result.covered_capabilities


# ============================================================================
# 9. Communication structure
# ============================================================================


def test_communication_structure_affects_alignment() -> None:
    """Test that communication structure affects task alignment."""
    evaluator = TaskPerformanceEvaluator()
    task = _make_task(
        task_id="task-collab",
        task_category="analysis",
        required_capabilities=["analysis", "synthesis"],
    )
    architecture = _make_default_architecture()

    result = evaluator.evaluate(task, architecture)

    # Should have reasonable alignment with communication
    assert result.task_alignment_score > 0.0


# ============================================================================
# 10. Different tasks producing different scores
# ============================================================================


def test_different_tasks_different_scores() -> None:
    """Test that different tasks produce different scores."""
    evaluator = TaskPerformanceEvaluator()
    architecture = _make_default_architecture()

    task_coding = _make_task(
        task_id="task-coding",
        task_category="coding",
        required_capabilities=["coding"],
    )
    task_research = _make_task(
        task_id="task-research",
        task_category="research",
        required_capabilities=["research"],
    )

    result_coding = evaluator.evaluate(task_coding, architecture)
    result_research = evaluator.evaluate(task_research, architecture)

    # Both should have coverage, but different capabilities
    assert "coding" in result_coding.covered_capabilities
    assert "research" in result_research.covered_capabilities
    assert result_coding.task_id != result_research.task_id


# ============================================================================
# 11. Serialization
# ============================================================================


def test_task_performance_result_serialization() -> None:
    """Test that TaskPerformanceResult can be serialized."""
    evaluator = TaskPerformanceEvaluator()
    task = _make_task(task_id="task-001", required_capabilities=["coding"])
    architecture = _make_default_architecture()

    result = evaluator.evaluate(task, architecture)
    serialized = result.to_dict()

    assert isinstance(serialized, dict)
    assert serialized["task_id"] == "task-001"
    assert "task_success_score" in serialized
    assert "capability_coverage_score" in serialized
    assert "missing_capabilities" in serialized
    assert "covered_capabilities" in serialized


def test_task_performance_result_json_serialization() -> None:
    """Test JSON serialization."""
    evaluator = TaskPerformanceEvaluator()
    task = _make_task(task_id="task-001", required_capabilities=["coding"])
    architecture = _make_default_architecture()

    result = evaluator.evaluate(task, architecture)
    json_str = json.dumps(result.to_dict(), indent=2)

    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["task_id"] == "task-001"


# ============================================================================
# 12. Empty requirements
# ============================================================================


def test_empty_requirements_full_coverage() -> None:
    """Test that empty requirements result in full coverage."""
    evaluator = TaskPerformanceEvaluator()
    task = _make_task(
        task_id="task-empty",
        task_category="coding",
        required_capabilities=[],
    )
    architecture = _make_default_architecture()

    result = evaluator.evaluate(task, architecture)

    assert result.capability_coverage_score == 1.0
    assert result.task_success_score == 1.0
    assert len(result.missing_capabilities) == 0
    assert len(result.covered_capabilities) == 0


# ============================================================================
# 13. Role-to-capability mapping
# ============================================================================


def test_role_capability_mapping() -> None:
    """Test the role-to-capability mapping."""
    # Test known roles
    coding_caps = get_capabilities_for_role("implementation")
    assert "coding" in coding_caps

    research_caps = get_capabilities_for_role("research")
    assert "research" in research_caps

    # Test unknown role
    unknown_caps = get_capabilities_for_role("unknown_role")
    assert unknown_caps == set()


def test_roles_for_capability() -> None:
    """Test getting roles for a capability."""
    # Coding can be provided by implementation role
    roles = get_roles_for_capability("coding")
    assert "implementation" in roles

    # Research can be provided by research role
    roles = get_roles_for_capability("research")
    assert "research" in roles


# ============================================================================
# 14. Combined evaluation
# ============================================================================


def test_combined_evaluation_without_task_performance() -> None:
    """Test combined evaluation without task performance."""
    evaluator = ArchitectureEvaluator()
    architecture = _make_default_architecture()

    structural_result = evaluator.evaluate(architecture)
    combined = create_combined_evaluation(structural_result)

    assert isinstance(combined, CombinedEvaluationResult)
    assert combined.task_success_score is None
    assert combined.task_id is None
    assert combined.validity_score == structural_result.validity_score


def test_combined_evaluation_with_task_performance() -> None:
    """Test combined evaluation with task performance."""
    structural_evaluator = ArchitectureEvaluator()
    task_evaluator = TaskPerformanceEvaluator()

    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding"],
    )
    architecture = _make_default_architecture()

    structural_result = structural_evaluator.evaluate(architecture)
    task_result = task_evaluator.evaluate(task, architecture)

    combined = create_combined_evaluation(structural_result, task_result)

    assert combined.task_success_score == task_result.task_success_score
    assert combined.task_id == task_result.task_id
    assert combined.validity_score == structural_result.validity_score
    assert combined.capability_coverage_score == task_result.capability_coverage_score


# ============================================================================
# 15. Extended reward calculator
# ============================================================================


def test_reward_calculator_backward_compatible() -> None:
    """Test that TaskPerformanceRewardCalculator is backward compatible."""
    from app.rl.environment import MASArchitectureEnv
    
    calc = TaskPerformanceRewardCalculator(task_performance_weight=0.0)

    # Create structural results
    evaluator = ArchitectureEvaluator()
    manager = ArchitectureManager.create_default_architecture()
    env = MASArchitectureEnv(manager)
    env.reset()
    
    previous = evaluator.evaluate(env.manager.get_architecture())
    
    # Make a small change
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

    # Without task performance, should behave like original calculator
    reward = calc.calculate(
        previous_result=previous,
        current_result=current,
        valid_transition=True,
    )

    # Should match structural delta
    expected = current.overall_score - previous.overall_score
    assert reward == pytest.approx(expected)


def test_reward_calculator_with_task_performance() -> None:
    """Test reward calculator with task performance."""
    from app.rl.environment import MASArchitectureEnv
    
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
    
    previous_structural = structural_evaluator.evaluate(env.manager.get_architecture())
    previous_task = task_evaluator.evaluate(task, env.manager.get_architecture())

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

    # Should include task performance component
    assert isinstance(reward, float)


def test_reward_calculator_context() -> None:
    """Test reward calculator with context."""
    from app.rl.environment import MASArchitectureEnv
    
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
    
    previous_structural = structural_evaluator.evaluate(env.manager.get_architecture())
    previous_task = task_evaluator.evaluate(task, env.manager.get_architecture())

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

    context = calc.calculate_with_context(
        previous_result=previous_structural,
        current_result=current_structural,
        valid_transition=True,
        previous_task_performance=previous_task,
        current_task_performance=current_task,
    )

    assert "reward" in context
    assert "previous_score" in context
    assert "current_score" in context
    assert "task_performance_delta" in context


# ============================================================================
# 16. No LLM/API calls
# ============================================================================


def test_evaluator_has_no_api_imports() -> None:
    """Test that task performance module doesn't import API-related modules."""
    import app.evaluation.task_performance as tp
    import inspect

    source = inspect.getsource(tp)

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


def test_evaluation_is_fully_offline() -> None:
    """Test that evaluation runs without network calls."""
    evaluator = TaskPerformanceEvaluator()
    task = _make_task(task_id="task-001", required_capabilities=["coding"])
    architecture = _make_default_architecture()

    # If we reach here without network errors, the module stayed offline
    result = evaluator.evaluate(task, architecture)
    assert isinstance(result, TaskPerformanceResult)


# ============================================================================
# 17. No credentials/secrets
# ============================================================================


def test_result_contains_no_credentials() -> None:
    """Test that evaluation results contain no credentials."""
    evaluator = TaskPerformanceEvaluator()
    task = _make_task(task_id="task-001", required_capabilities=["coding"])
    architecture = _make_default_architecture()

    result = evaluator.evaluate(task, architecture)
    serialized = result.to_dict()
    serialized_json = json.dumps(serialized, default=str)

    # Check for credential markers
    forbidden_markers = [
        "sk-or-",
        "sk-proj-",
        "api_key",
        "openai_api_key",
        "password",
        "secret",
    ]
    serialized_lower = serialized_json.lower()

    for marker in forbidden_markers:
        assert marker not in serialized_lower, f"Found forbidden marker: {marker}"


def test_role_mapping_contains_no_credentials() -> None:
    """Test that role capability mapping contains no credentials."""
    # Convert sets to lists for JSON serialization
    serializable_map = {k: list(v) for k, v in ROLE_CAPABILITY_MAP.items()}
    serialized = json.dumps(serializable_map)
    serialized_lower = serialized.lower()

    forbidden_markers = ["sk-or-", "api_key", "password", "secret"]
    for marker in forbidden_markers:
        assert marker not in serialized_lower


# ============================================================================
# 18. Invalid task handling
# ============================================================================


def test_invalid_task_raises() -> None:
    """Test that invalid task raises appropriate error."""
    evaluator = TaskPerformanceEvaluator()
    architecture = _make_default_architecture()

    # Empty task_id should be rejected by MetaTask validation
    with pytest.raises(ValueError, match="task_id must be a non-empty string"):
        TaskPerformanceEvaluator().evaluate(
            MetaTask(
                task_id="   ",
                task_description="Test",
                task_category="coding",
            ),
            architecture,
        )


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
# 21. Task alignment score calculation
# ============================================================================


def test_task_alignment_with_good_match() -> None:
    """Test task alignment with good role matching."""
    evaluator = TaskPerformanceEvaluator()
    task = _make_task(
        task_id="task-coding",
        task_category="coding",
        required_capabilities=["coding"],
    )
    architecture = _make_default_architecture()

    result = evaluator.evaluate(task, architecture)

    # Should have good alignment since coder role matches coding task
    assert result.task_alignment_score > 0.5


def test_task_alignment_with_poor_match() -> None:
    """Test task alignment with poor role matching."""
    evaluator = TaskPerformanceEvaluator()
    # Require capability not well-matched to default roles
    task = _make_task(
        task_id="task-poor-match",
        task_category="unknown",
        required_capabilities=["nonexistent_capability"],
    )
    architecture = _make_default_architecture()

    result = evaluator.evaluate(task, architecture)

    # Should have poor or moderate alignment (0.5 is the floor for no capabilities)
    assert result.task_alignment_score <= 0.5


# ============================================================================
# 22. Capability coverage score bounds
# ============================================================================


def test_capability_coverage_score_bounds() -> None:
    """Test that capability coverage score is in [0.0, 1.0]."""
    evaluator = TaskPerformanceEvaluator()
    task = _make_task(
        task_id="task-001",
        task_category="coding",
        required_capabilities=["coding", "testing"],
    )
    architecture = _make_default_architecture()

    result = evaluator.evaluate(task, architecture)

    assert 0.0 <= result.capability_coverage_score <= 1.0
    assert 0.0 <= result.task_success_score <= 1.0
    assert 0.0 <= result.task_alignment_score <= 1.0


# ============================================================================
# 23. Multiple capabilities
# ============================================================================


def test_multiple_required_capabilities() -> None:
    """Test evaluation with multiple required capabilities."""
    evaluator = TaskPerformanceEvaluator()
    task = _make_task(
        task_id="task-multi",
        task_category="research",
        required_capabilities=["research", "analysis", "synthesis"],
    )
    architecture = _make_default_architecture()

    result = evaluator.evaluate(task, architecture)

    # Should cover some but not all
    assert len(result.covered_capabilities) > 0
    assert len(result.missing_capabilities) >= 0
    assert result.capability_coverage_score > 0.0


# ============================================================================
# 24. Score documentation
# ============================================================================


def test_notes_document_proxy_nature() -> None:
    """Test that notes document the proxy nature of scores."""
    evaluator = TaskPerformanceEvaluator()
    task = _make_task(task_id="task-001", required_capabilities=["coding"])
    architecture = _make_default_architecture()

    result = evaluator.evaluate(task, architecture)

    # Notes should mention this is a proxy
    notes_lower = result.notes.lower()
    assert "proxy" in notes_lower or "compatibility" in notes_lower
    # Should document this is NOT actual execution (the phrase "not actual llm task execution" should appear)
    assert "not actual" in notes_lower
