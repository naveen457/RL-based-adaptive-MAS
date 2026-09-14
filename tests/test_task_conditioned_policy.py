from __future__ import annotations

import math

import pytest
import torch

from app.architecture.actions import ArchitectureAction
from app.architecture.manager import ArchitectureManager
from app.rl.environment import MASArchitectureEnv
from app.rl.meta_task import MetaTaskContext
from app.rl.task_conditioned_policy import (
    ArchitectureStateTensorEncoder,
    DeterministicTaskEncoder,
    TaskConditionedArchitectureNet,
    TaskConditionedPolicy,
    create_task_conditioned_policy_from_environment,
)


def _env() -> MASArchitectureEnv:
    env = MASArchitectureEnv(
        ArchitectureManager.create_default_architecture(),
        role_options=["analysis", "planning"],
    )
    env.reset()
    return env


def _task_context(
    *,
    category: str = "coding",
    capabilities: list[str] | None = None,
    difficulty: int = 3,
    features: dict | None = None,
) -> MetaTaskContext:
    return MetaTaskContext(
        task_category=category,
        required_capabilities=capabilities or ["coding", "testing"],
        difficulty=difficulty,
        context_features=features or {"num_steps": 5, "has_dependencies": True},
    )


def _policy(env: MASArchitectureEnv, *, seed: int = 7) -> TaskConditionedPolicy:
    return create_task_conditioned_policy_from_environment(env, seed=seed)


def test_deterministic_task_encoding() -> None:
    encoder = DeterministicTaskEncoder()
    context = _task_context()

    first = encoder.encode_context(context)
    second = encoder.encode_context(context)

    assert torch.equal(first, second)
    assert first.shape == (encoder.output_dim,)


def test_unknown_category_handling() -> None:
    encoder = DeterministicTaskEncoder()
    context = _task_context(category="brand_new_category")

    encoded = encoder.encode_context(context)

    unknown_idx = encoder.category_vocabulary.index("unknown")
    assert encoded[unknown_idx].item() == pytest.approx(1.0)


def test_unknown_capability_handling() -> None:
    encoder = DeterministicTaskEncoder()
    context = _task_context(capabilities=["unseen_capability"])

    encoded = encoder.encode_context(context)

    capability_start = len(encoder.category_vocabulary) + 1
    unknown_idx = capability_start + encoder.capability_vocabulary.index("unknown")
    assert encoded[unknown_idx].item() == pytest.approx(1.0)


def test_context_feature_handling() -> None:
    encoder = DeterministicTaskEncoder(context_feature_keys=["num_steps", "flag", "label"])
    context = _task_context(features={"num_steps": 8, "flag": True, "label": "python"})

    encoded = encoder.encode_context(context)
    features = encoded[-3:]

    assert features.tolist() == pytest.approx([8.0, 1.0, 0.06])


def test_architecture_state_encoding() -> None:
    env = _env()
    observation = env.encoder.encode()
    encoder = ArchitectureStateTensorEncoder(max_agent_count=observation["agent_count"])

    encoded = encoder.encode_observation(observation)

    assert encoded.shape == (encoder.output_dim,)
    assert torch.equal(encoded, encoder.encode_observation(observation))
    assert torch.isfinite(encoded).all()


def test_network_output_dimension_matches_existing_action_space() -> None:
    env = _env()
    arch_encoder = ArchitectureStateTensorEncoder(max_agent_count=5)
    task_encoder = DeterministicTaskEncoder()
    net = TaskConditionedArchitectureNet(
        arch_encoder.output_dim,
        task_encoder.output_dim,
        env.mapper.action_count,
    )

    logits = net(
        arch_encoder.encode_observation(env.encoder.encode()),
        task_encoder.encode_context(_task_context()),
    )

    assert logits.shape == (1, env.mapper.action_count)


def test_policy_can_be_instantiated() -> None:
    env = _env()
    policy = _policy(env)

    assert policy.action_dim == env.mapper.action_count
    assert policy.action_mapper is env.mapper


def test_deterministic_inference_returns_same_action_for_same_input() -> None:
    env = _env()
    policy = _policy(env, seed=11)
    observation = env.encoder.encode()
    valid = list(range(env.mapper.action_count))

    first = policy.select_action(observation, valid, _task_context())
    second = policy.select_action(observation, valid, _task_context())

    assert first == second


def test_selected_action_maps_to_existing_architecture_action() -> None:
    env = _env()
    policy = _policy(env)
    action = policy.select_architecture_action(
        env.encoder.encode(), list(range(env.mapper.action_count)), _task_context()
    )

    assert isinstance(action, ArchitectureAction)
    assert action in env.mapper.actions


def test_selected_action_is_valid_for_current_architecture_state() -> None:
    env = _env()
    policy = _policy(env)
    action_id = policy.select_action(
        env.encoder.encode(), list(range(env.mapper.action_count)), _task_context()
    )

    _, _, _, _, info = env.step(action_id)

    assert info["transition"]["valid"] is True


def test_exploration_can_produce_different_actions_when_enabled() -> None:
    env = _env()
    policy = _policy(env, seed=3)
    observation = env.encoder.encode()
    valid = list(range(env.mapper.action_count))

    actions = {
        policy.select_action(
            observation,
            valid,
            _task_context(),
            deterministic=False,
            exploration_rate=1.0,
        )
        for _ in range(30)
    }

    assert len(actions) > 1


def test_update_returns_finite_loss() -> None:
    env = _env()
    policy = _policy(env)
    target = [0.0] * env.mapper.action_count
    target[0] = 1.0

    loss = policy.update_supervised(env.encoder.encode(), _task_context(), target)

    assert math.isfinite(loss)


def test_model_parameters_change_after_update() -> None:
    env = _env()
    policy = _policy(env)
    before = policy.clone_parameters()
    target = [0.0] * env.mapper.action_count
    target[0] = 1.0

    policy.update_supervised(env.encoder.encode(), _task_context(), target)

    assert policy.parameters_changed_since(before)


def test_save_load_restores_identical_model_outputs(tmp_path) -> None:
    env = _env()
    policy = _policy(env, seed=13)
    observation = env.encoder.encode()
    context = _task_context()
    before = policy.action_scores_tensor(observation, context)
    path = tmp_path / "task_conditioned_policy.pt"

    policy.save(path)
    loaded = TaskConditionedPolicy.load(path, action_mapper=env.mapper, seed=999)
    after = loaded.action_scores_tensor(observation, context)

    assert torch.allclose(before, after)


def test_factory_construction_works_from_existing_environment() -> None:
    env = _env()
    policy = create_task_conditioned_policy_from_environment(env, seed=5)

    action_id = policy.select_action(
        env.encoder.encode(), list(range(env.mapper.action_count)), _task_context()
    )

    assert env.mapper.is_valid_id(action_id)
