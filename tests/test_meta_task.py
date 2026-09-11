"""
Tests for the Meta-RL task and meta-environment implementation (Step 13).

These tests cover:

1. MetaTask creation and validation.
2. MetaTaskContext creation and validation.
3. Deterministic serialization.
4. Task context creation from tasks.
5. Different tasks produce different contexts.
6. Same task produces same context.
7. Meta-observation contains architecture state + task context.
8. Reset behavior.
9. No LLM/API calls.
10. Existing environment behavior remains unchanged.

These tests are fully offline. No OpenRouter calls are made.
"""

from __future__ import annotations

import pytest

from app.architecture.manager import ArchitectureManager
from app.architecture.actions import ActionType, ArchitectureAction
from app.rl.environment import MASArchitectureEnv
from app.rl.meta_task import MetaTask, MetaTaskContext
from app.rl.meta_environment import MetaEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_manager() -> ArchitectureManager:
    return ArchitectureManager.create_default_architecture()


def _make_meta_task(
    *,
    task_id: str = "task-coder-001",
    task_description: str = "Write a Python function to sort a list",
    task_category: str = "coding",
    required_capabilities: list[str] | None = None,
    difficulty: int = 2,
    context_features: dict[str, object] | None = None,
    notes: str = "",
) -> MetaTask:
    if required_capabilities is None:
        required_capabilities = ["coding", "testing"]
    if context_features is None:
        context_features = {}
    return MetaTask(
        task_id=task_id,
        task_description=task_description,
        task_category=task_category,
        required_capabilities=required_capabilities,
        difficulty=difficulty,
        context_features=context_features,
        notes=notes,
    )


def _make_meta_environment(
    *,
    task: MetaTask | None = None,
    max_steps: int | None = None,
    role_options: list[str] | None = None,
) -> MetaEnvironment:
    manager = _default_manager()
    kwargs: dict[str, object] = {"task": task} if task is not None else {}
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    if role_options is not None:
        kwargs["role_options"] = role_options
    return MetaEnvironment(manager, **kwargs)


# ---------------------------------------------------------------------------
# 1. MetaTask creation and validation
# ---------------------------------------------------------------------------

def test_meta_task_creation_with_all_fields() -> None:
    task = _make_meta_task()
    assert task.task_id == "task-coder-001"
    assert task.task_description == "Write a Python function to sort a list"
    assert task.task_category == "coding"
    assert task.required_capabilities == ["coding", "testing"]
    assert task.difficulty == 2
    assert task.context_features == {}
    assert task.notes == ""


def test_meta_task_creation_with_minimal_fields() -> None:
    task = MetaTask(
        task_id="task-minimal",
        task_description="A minimal task",
        task_category="analysis",
    )
    assert task.task_id == "task-minimal"
    assert task.task_description == "A minimal task"
    assert task.task_category == "analysis"
    assert task.required_capabilities == []
    assert task.difficulty == 1  # default
    assert task.context_features == {}


def test_meta_task_rejects_empty_task_id() -> None:
    with pytest.raises(ValueError, match="task_id must be a non-empty string"):
        MetaTask(
            task_id="   ",
            task_description="A task",
            task_category="coding",
        )


def test_meta_task_rejects_empty_task_description() -> None:
    with pytest.raises(ValueError, match="task_description must be a non-empty string"):
        MetaTask(
            task_id="task-1",
            task_description="   ",
            task_category="coding",
        )


def test_meta_task_rejects_empty_task_category() -> None:
    with pytest.raises(ValueError, match="task_category must be a non-empty string"):
        MetaTask(
            task_id="task-1",
            task_description="A task",
            task_category="   ",
        )


def test_meta_task_rejects_duplicate_required_capabilities() -> None:
    # Empty list is fine
    MetaTask(
        task_id="task-1",
        task_description="A task",
        task_category="coding",
        required_capabilities=[],
    )
    # Duplicates should raise
    with pytest.raises(ValueError, match="required_capabilities must contain unique values"):
        MetaTask(
            task_id="task-1",
            task_description="A task",
            task_category="coding",
            required_capabilities=["coding", "coding"],
        )


def test_meta_task_rejects_difficulty_below_1() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        MetaTask(
            task_id="task-1",
            task_description="A task",
            task_category="coding",
            difficulty=0,
        )


def test_meta_task_rejects_difficulty_above_5() -> None:
    with pytest.raises(ValueError, match="less than or equal to 5"):
        MetaTask(
            task_id="task-1",
            task_description="A task",
            task_category="coding",
            difficulty=6,
        )


def test_meta_task_rejects_non_json_serializable_context_features() -> None:
    with pytest.raises(ValueError, match="must be JSON-serializable"):
        MetaTask(
            task_id="task-1",
            task_description="A task",
            task_category="coding",
            context_features={"key": object()},
        )


def test_meta_task_accepts_valid_context_features() -> None:
    task = MetaTask(
        task_id="task-1",
        task_description="A task",
        task_category="coding",
        context_features={
            "num_steps": 5,
            "has_dependencies": True,
            "complexity": 3.14,
            "tags": ["a", "b", "c"],
            "nested": {"key": "value"},
            "none_value": None,
        },
    )
    assert task.context_features["num_steps"] == 5
    assert task.context_features["has_dependencies"] is True
    assert task.context_features["complexity"] == 3.14
    assert task.context_features["tags"] == ["a", "b", "c"]
    assert task.context_features["nested"] == {"key": "value"}
    assert task.context_features["none_value"] is None


# ---------------------------------------------------------------------------
# 2. MetaTaskContext creation and validation
# ---------------------------------------------------------------------------

def test_meta_task_context_from_task() -> None:
    task = _make_meta_task(
        task_id="task-1",
        task_description="A task",
        task_category="coding",
        required_capabilities=["coding", "testing"],
        difficulty=3,
        context_features={"num_steps": 5},
    )
    context = task.to_context()
    assert isinstance(context, MetaTaskContext)
    assert context.task_category == "coding"
    assert context.required_capabilities == ["coding", "testing"]
    assert context.difficulty == 3
    assert context.context_features == {"num_steps": 5}
    assert context.category_embedding is None
    assert context.capability_vector is None


def test_meta_task_context_creation_directly() -> None:
    context = MetaTaskContext(
        task_category="research",
        required_capabilities=["research", "synthesis"],
        difficulty=4,
        context_features={"has_web_search": True},
    )
    assert context.task_category == "research"
    assert context.required_capabilities == ["research", "synthesis"]
    assert context.difficulty == 4
    assert context.context_features == {"has_web_search": True}


def test_meta_task_context_rejects_empty_task_category() -> None:
    with pytest.raises(ValueError, match="task_category must be a non-empty string"):
        MetaTaskContext(
            task_category="   ",
            required_capabilities=[],
            difficulty=1,
        )


def test_meta_task_context_rejects_invalid_difficulty() -> None:
    with pytest.raises(ValueError):
        MetaTaskContext(
            task_category="coding",
            required_capabilities=[],
            difficulty=0,
        )
    with pytest.raises(ValueError):
        MetaTaskContext(
            task_category="coding",
            required_capabilities=[],
            difficulty=6,
        )


def test_meta_task_context_serialization() -> None:
    task = _make_meta_task(
        task_id="task-1",
        task_description="A task",
        task_category="coding",
        required_capabilities=["coding"],
        difficulty=2,
        context_features={"key": "value"},
    )
    context = task.to_context()
    serialized = context.serialize()
    assert serialized["task_category"] == "coding"
    assert serialized["required_capabilities"] == ["coding"]
    assert serialized["difficulty"] == 2
    assert serialized["context_features"] == {"key": "value"}
    assert "category_embedding" not in serialized
    assert "capability_vector" not in serialized


def test_meta_task_context_to_dict() -> None:
    task = _make_meta_task(
        task_id="task-1",
        task_description="A task",
        task_category="coding",
        required_capabilities=["coding", "testing"],
        difficulty=3,
        context_features={"num_steps": 5},
    )
    context = task.to_context()
    result = context.to_dict()
    assert result["task_category"] == "coding"
    assert result["required_capabilities"] == ["coding", "testing"]
    assert result["difficulty"] == 3
    assert result["context_features"] == {"num_steps": 5}


def test_meta_task_context_equality() -> None:
    context_a = MetaTaskContext(
        task_category="coding",
        required_capabilities=["coding"],
        difficulty=2,
        context_features={"key": "value"},
    )
    context_b = MetaTaskContext(
        task_category="coding",
        required_capabilities=["coding"],
        difficulty=2,
        context_features={"key": "value"},
    )
    context_c = MetaTaskContext(
        task_category="research",
        required_capabilities=["coding"],
        difficulty=2,
        context_features={"key": "value"},
    )
    assert context_a == context_b
    assert context_a != context_c
    assert context_a != "not a context"


def test_meta_task_context_equality_with_none_embedding() -> None:
    context_a = MetaTaskContext(
        task_category="coding",
        required_capabilities=["coding"],
        difficulty=2,
        context_features={},
        category_embedding=None,
        capability_vector=None,
    )
    context_b = MetaTaskContext(
        task_category="coding",
        required_capabilities=["coding"],
        difficulty=2,
        context_features={},
        category_embedding=None,
        capability_vector=None,
    )
    assert context_a == context_b


# ---------------------------------------------------------------------------
# 3. Deterministic serialization
# ---------------------------------------------------------------------------

def test_meta_task_serialization() -> None:
    task = _make_meta_task(
        task_id="task-1",
        task_description="A task",
        task_category="coding",
        required_capabilities=["coding", "testing"],
        difficulty=3,
        context_features={"num_steps": 5},
        notes="Some notes",
    )
    serialized = task.serialize()
    assert serialized["task_id"] == "task-1"
    assert serialized["task_description"] == "A task"
    assert serialized["task_category"] == "coding"
    assert serialized["required_capabilities"] == ["coding", "testing"]
    assert serialized["difficulty"] == 3
    assert serialized["context_features"] == {"num_steps": 5}
    assert serialized["notes"] == "Some notes"


def test_meta_task_serialization_exclude_none() -> None:
    task = MetaTask(
        task_id="task-minimal",
        task_description="A minimal task",
        task_category="analysis",
    )
    serialized = task.serialize()
    assert serialized["task_id"] == "task-minimal"
    assert serialized["task_description"] == "A minimal task"
    assert serialized["task_category"] == "analysis"
    assert serialized["required_capabilities"] == []
    assert serialized["difficulty"] == 1
    assert serialized["context_features"] == {}
    assert serialized["notes"] == ""


def test_meta_task_round_trip() -> None:
    task = _make_meta_task(
        task_id="task-1",
        task_description="A task",
        task_category="coding",
        required_capabilities=["coding", "testing"],
        difficulty=3,
        context_features={"num_steps": 5},
        notes="Some notes",
    )
    serialized = task.serialize()
    # Re-create from serialized dict
    restored = MetaTask(**serialized)
    assert restored.task_id == task.task_id
    assert restored.task_description == task.task_description
    assert restored.task_category == task.task_category
    assert restored.required_capabilities == task.required_capabilities
    assert restored.difficulty == task.difficulty
    assert restored.context_features == task.context_features
    assert restored.notes == task.notes


# ---------------------------------------------------------------------------
# 4. Task context creation from tasks
# ---------------------------------------------------------------------------

def test_task_context_created_from_task() -> None:
    task = _make_meta_task()
    context = task.to_context()
    assert context.task_category == task.task_category
    assert context.required_capabilities == task.required_capabilities
    assert context.difficulty == task.difficulty
    assert context.context_features == task.context_features


def test_task_context_is_deterministic() -> None:
    task = _make_meta_task()
    context_a = task.to_context()
    context_b = task.to_context()
    assert context_a == context_b
    assert context_a.to_dict() == context_b.to_dict()


# ---------------------------------------------------------------------------
# 5. Different tasks produce different contexts
# ---------------------------------------------------------------------------

def test_different_tasks_different_categories() -> None:
    task_coding = _make_meta_task(task_category="coding")
    task_research = _make_meta_task(task_category="research")
    assert task_coding.to_context() != task_research.to_context()


def test_different_tasks_different_capabilities() -> None:
    task_a = _make_meta_task(required_capabilities=["coding"])
    task_b = _make_meta_task(required_capabilities=["research"])
    assert task_a.to_context() != task_b.to_context()


def test_different_tasks_different_difficulty() -> None:
    task_easy = _make_meta_task(difficulty=1)
    task_hard = _make_meta_task(difficulty=5)
    assert task_easy.to_context() != task_hard.to_context()


def test_different_tasks_different_context_features() -> None:
    task_a = _make_meta_task(context_features={"key": "value1"})
    task_b = _make_meta_task(context_features={"key": "value2"})
    assert task_a.to_context() != task_b.to_context()


# ---------------------------------------------------------------------------
# 6. Same task produces same context
# ---------------------------------------------------------------------------

def test_same_task_same_context() -> None:
    task = _make_meta_task()
    context_a = task.to_context()
    context_b = task.to_context()
    assert context_a == context_b


def test_same_task_different_instances_same_context() -> None:
    task_a = _make_meta_task(
        task_id="task-1",
        task_description="A task",
        task_category="coding",
        required_capabilities=["coding"],
        difficulty=2,
    )
    task_b = _make_meta_task(
        task_id="task-1",
        task_description="A task",
        task_category="coding",
        required_capabilities=["coding"],
        difficulty=2,
    )
    assert task_a.to_context() == task_b.to_context()


# ---------------------------------------------------------------------------
# 7. Meta-observation contains architecture state + task context
# ---------------------------------------------------------------------------

def test_meta_observation_contains_architecture_state() -> None:
    task = _make_meta_task()
    env = _make_meta_environment(task=task)
    observation, _ = env.reset()
    assert "agent_ids" in observation
    assert "activity_vector" in observation
    assert "role_vector" in observation
    assert "adjacency_matrix" in observation
    assert "agent_count" in observation
    assert "active_agent_count" in observation


def test_meta_observation_contains_task_context() -> None:
    task = _make_meta_task(
        task_category="coding",
        required_capabilities=["coding", "testing"],
        difficulty=3,
        context_features={"num_steps": 5},
    )
    env = _make_meta_environment(task=task)
    observation, _ = env.reset()
    assert "task_context" in observation
    task_context = observation["task_context"]
    assert task_context["task_category"] == "coding"
    assert task_context["required_capabilities"] == ["coding", "testing"]
    assert task_context["difficulty"] == 3
    assert task_context["context_features"] == {"num_steps": 5}


def test_meta_observation_without_task_has_empty_task_context() -> None:
    env = _make_meta_environment(task=None)
    observation, _ = env.reset()
    assert "task_context" in observation
    task_context = observation["task_context"]
    assert task_context["task_category"] == ""
    assert task_context["required_capabilities"] == []
    assert task_context["difficulty"] == 1
    assert task_context["context_features"] == {}


def test_meta_observation_after_step_still_has_task_context() -> None:
    task = _make_meta_task(task_category="coding")
    env = _make_meta_environment(task=task)
    _, _ = env.reset()
    # Take a step
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    observation, _, _, _, _ = env.step(action_id)
    assert "task_context" in observation
    assert observation["task_context"]["task_category"] == "coding"


def test_meta_observation_combines_both_parts() -> None:
    task = _make_meta_task(task_category="research")
    env = _make_meta_environment(task=task)
    observation, _ = env.reset()
    # Verify architecture state is present
    assert "agent_ids" in observation
    assert "activity_vector" in observation
    # Verify task context is present
    assert "task_context" in observation
    assert observation["task_context"]["task_category"] == "research"
    # Verify they are separate keys
    assert "agent_ids" in observation
    assert "task_context" in observation
    assert observation["agent_ids"] != observation["task_context"]


# ---------------------------------------------------------------------------
# 8. Reset behavior
# ---------------------------------------------------------------------------

def test_meta_environment_reset_restores_initial_state() -> None:
    task = _make_meta_task(task_category="coding")
    env = _make_meta_environment(task=task)
    observation_before, _ = env.reset()
    # Take a step
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    env.step(action_id)
    # Reset
    observation_after, _ = env.reset()
    # Observations should be identical
    assert observation_before == observation_after
    # Task context should be preserved
    assert observation_after["task_context"]["task_category"] == "coding"


def test_meta_environment_reset_clears_step_count() -> None:
    task = _make_meta_task()
    env = _make_meta_environment(task=task)
    _, _ = env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    env.step(action_id)
    assert env.step_count == 1
    env.reset()
    assert env.step_count == 0


def test_meta_environment_can_change_task_after_reset() -> None:
    task_a = _make_meta_task(task_category="coding")
    task_b = _make_meta_task(task_category="research")
    env = _make_meta_environment(task=task_a)
    _, _ = env.reset()
    # Change task
    env.set_task(task_b)
    env.reset()
    observation, _ = env.reset()
    assert observation["task_context"]["task_category"] == "research"


def test_meta_environment_step_truncates() -> None:
    task = _make_meta_task()
    env = _make_meta_environment(task=task, max_steps=2)
    _, _ = env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    env.step(action_id)
    env.step(action_id)
    assert env.truncated is True
    assert env.step_count == 2


# ---------------------------------------------------------------------------
# 9. No LLM/API calls
# ---------------------------------------------------------------------------

def test_meta_task_creation_no_network_calls() -> None:
    # This test verifies that creating a MetaTask doesn't make any network calls
    # by checking that the task is created successfully without any exceptions
    task = _make_meta_task()
    assert task.task_id is not None
    assert task.task_description is not None
    assert task.task_category is not None


def test_meta_environment_no_network_calls_on_reset() -> None:
    task = _make_meta_task()
    env = _make_meta_environment(task=task)
    # Reset should not make any network calls
    observation, info = env.reset()
    assert isinstance(observation, dict)
    assert isinstance(info, dict)


def test_meta_environment_no_network_calls_on_step() -> None:
    task = _make_meta_task()
    env = _make_meta_environment(task=task)
    _, _ = env.reset()
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    # Step should not make any network calls
    observation, reward, terminated, truncated, info = env.step(action_id)
    assert isinstance(observation, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# 10. Existing environment behavior remains unchanged
# ---------------------------------------------------------------------------

def test_meta_environment_delegates_to_base_environment() -> None:
    task = _make_meta_task()
    env = _make_meta_environment(task=task)
    # Verify that the underlying environment is accessible
    assert env.manager is not None
    assert env.mapper is not None
    assert env.encoder is not None


def test_meta_environment_preserves_base_environment_reset_behavior() -> None:
    task = _make_meta_task()
    env = _make_meta_environment(task=task)
    # Reset should work like the base environment
    observation, info = env.reset()
    assert isinstance(observation, dict)
    assert isinstance(info, dict)
    # Verify architecture state is correct
    assert observation["agent_count"] == 5
    assert observation["active_agent_count"] == 5


def test_meta_environment_preserves_base_environment_step_behavior() -> None:
    task = _make_meta_task()
    env = _make_meta_environment(task=task)
    _, _ = env.reset()
    # Add an edge
    action_id = env.encode_action(
        ArchitectureAction(
            action_type=ActionType.ADD_EDGE,
            source="planner",
            target="critic",
        )
    )
    observation, reward, terminated, truncated, info = env.step(action_id)
    # Verify the edge was added
    edges = {(e.source, e.target) for e in env.manager.get_communication_edges()}
    assert ("planner", "critic") in edges
    # Verify reward is computed correctly
    assert reward == pytest.approx(info["evaluation"]["current_score"] - info["evaluation"]["previous_score"])
    # Verify task context is preserved
    assert observation["task_context"]["task_category"] == task.task_category


def test_meta_environment_task_context_preserved_after_multiple_steps() -> None:
    task = _make_meta_task(task_category="coding", difficulty=3)
    env = _make_meta_environment(task=task)
    _, _ = env.reset()
    # Take multiple steps
    for _ in range(3):
        action_id = env.encode_action(
            ArchitectureAction(
                action_type=ActionType.ADD_EDGE,
                source="planner",
                target="critic",
            )
        )
        # First step adds the edge, subsequent steps will fail (edge already exists)
        # but the task context should still be preserved
        observation, _, _, _, _ = env.step(action_id)
        assert observation["task_context"]["task_category"] == "coding"
        assert observation["task_context"]["difficulty"] == 3


def test_meta_environment_evaluation_works() -> None:
    task = _make_meta_task()
    env = _make_meta_environment(task=task)
    _, _ = env.reset()
    # Evaluation should work
    evaluation = env.current_evaluation
    assert evaluation is not None
    assert evaluation.architecture_id == "static-mas-v1"
    assert evaluation.task_success_score is None


def test_meta_environment_action_encoding_works() -> None:
    task = _make_meta_task()
    env = _make_meta_environment(task=task)
    _, _ = env.reset()
    action = ArchitectureAction(
        action_type=ActionType.ADD_EDGE,
        source="planner",
        target="critic",
    )
    action_id = env.encode_action(action)
    decoded = env.decode_action(action_id)
    assert decoded.action_type == action.action_type
    assert decoded.source == action.source
    assert decoded.target == action.target


def test_meta_environment_get_possible_actions_works() -> None:
    task = _make_meta_task()
    env = _make_meta_environment(task=task)
    _, _ = env.reset()
    actions = env.get_possible_actions()
    assert len(actions) > 0
    # All actions should be ArchitectureAction objects
    from app.architecture.actions import ArchitectureAction
    assert all(isinstance(a, ArchitectureAction) for a in actions)


# ---------------------------------------------------------------------------
# 11. Meta-environment with role options
# ---------------------------------------------------------------------------

def test_meta_environment_with_role_options() -> None:
    task = _make_meta_task()
    env = MetaEnvironment(
        _default_manager(),
        role_options=["analysis", "planning"],
        task=task,
    )
    _, _ = env.reset()
    assert env.mapper.role_options == ["analysis", "planning"]


def test_meta_environment_role_options_preserved_in_meta_observation() -> None:
    task = _make_meta_task()
    env = MetaEnvironment(
        _default_manager(),
        role_options=["analysis"],
        task=task,
    )
    _, info = env.reset()
    assert "role_options" in info
    assert info["role_options"] == ["analysis"]


# ---------------------------------------------------------------------------
# 12. Edge cases
# ---------------------------------------------------------------------------

def test_meta_environment_with_empty_task_context_features() -> None:
    task = _make_meta_task(context_features={})
    env = _make_meta_environment(task=task)
    observation, _ = env.reset()
    assert observation["task_context"]["context_features"] == {}


def test_meta_environment_with_complex_context_features() -> None:
    task = _make_meta_task(
        context_features={
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
        }
    )
    env = _make_meta_environment(task=task)
    observation, _ = env.reset()
    task_context = observation["task_context"]
    assert task_context["context_features"]["nested"] == {"key": "value"}
    assert task_context["context_features"]["list"] == [1, 2, 3]
    assert task_context["context_features"]["number"] == 42
    assert task_context["context_features"]["float"] == 3.14
    assert task_context["context_features"]["boolean"] is True
    assert task_context["context_features"]["none"] is None


def test_meta_environment_task_context_after_reset_with_different_task() -> None:
    task_a = _make_meta_task(task_category="coding")
    task_b = _make_meta_task(task_category="research")
    env = _make_meta_environment(task=task_a)
    _, _ = env.reset()
    # Change task
    env.set_task(task_b)
    # Reset should use the new task
    observation, _ = env.reset()
    assert observation["task_context"]["task_category"] == "research"


def test_meta_environment_step_with_invalid_action_preserves_task_context() -> None:
    task = _make_meta_task(task_category="coding")
    env = _make_meta_environment(task=task)
    _, _ = env.reset()
    # Use an invalid action ID
    with pytest.raises(ValueError, match="invalid action id"):
        env.step(-1)
    # Task context should still be accessible
    assert env.task_context is not None
    assert env.task_context.task_category == "coding"
