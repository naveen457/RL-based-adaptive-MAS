"""
RL environment for the MAS architecture adaptation layer.

This module provides a Gymnasium-compatible environment that lets a future
RL/Meta-RL controller observe and mutate the MAS architecture.

Responsibilities
----------------
* Maintain the current architecture, version, and transition history.
* Encode architecture state deterministically.
* Map ArchitectureAction objects to integer action IDs and back.
* Apply actions through the existing AdaptiveArchitecture / ArchitectureManager
  layer.
* Compute a simple baseline reward.
* Terminate episodes after a configurable maximum number of steps.
* Return safe debug info without credentials.

This environment DOES NOT implement:
* RL algorithm, policy, training loop, or Meta-RL controller.
* Dynamic LangGraph rebuilding.
* Persistent memory.
* Any LLM calls.

API style
---------
This environment follows the newer Gymnasium step signature:

    observation, reward, terminated, truncated, info = env.step(action)

If the installed environment uses an older Gym API, adapt the signature
accordingly. The current implementation targets the Gymnasium-style tuple.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.architecture.actions import ArchitectureAction
from app.architecture.manager import ArchitectureManager
from app.architecture.adaptive import AdaptiveArchitecture
from app.architecture.models import MASArchitecture
from app.architecture.actions import ActionType
from app.rl.action_space import ArchitectureActionMapper
from app.rl.state import ArchitectureStateEncoder
from app.evaluation.evaluator import ArchitectureEvaluator, EvaluationResult
from app.evaluation.reward import RewardCalculator


class MASArchitectureEnv:
    """Gymnasium-style environment over MAS architecture adaptation.

    Parameters
    ----------
    manager:
        Initial architecture manager. The environment keeps its own copy and
        resets to this initial architecture.
    role_options:
        Optional finite role vocabulary used for CHANGE_ROLE actions. If
        omitted, role-change actions are not included in the action space.
    max_steps:
        Maximum number of architecture transitions per episode. After this
        many steps the episode is truncated. If None, use a default of 50.
    """

    DEFAULT_MAX_STEPS = 50

    def __init__(
        self,
        manager: ArchitectureManager,
        role_options: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        if max_steps is None:
            max_steps = self.DEFAULT_MAX_STEPS
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        # Keep an independent copy of the initial architecture manager so
        # reset() can restore the episode start deterministically.
        self._initial_manager = ArchitectureManager(manager.to_architecture_model())
        self._manager = ArchitectureManager(manager.to_architecture_model())
        self._adaptive = AdaptiveArchitecture(self._manager)

        self._role_options = list(role_options) if role_options is not None else None
        self._max_steps = max_steps

        # Episode state
        self._step_count = 0
        self._terminated = False
        self._truncated = False

        # Evaluation and reward integration (Step 9).
        self._evaluator = ArchitectureEvaluator()
        self._reward_calculator = RewardCalculator()
        self._current_evaluation: EvaluationResult | None = None

        # Rebuild derived artifacts from the current architecture.
        self._rebuild()
        self._initialize_evaluation()

    # ------------------------------------------------------------------
    # Environment lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict, Dict]:
        """Reset the environment to the initial architecture.

        Returns
        -------
        observation:
            Deterministic initial observation.
        info:
            Safe debug information.
        """
        del seed, options  # unused in the deterministic baseline

        self._manager = ArchitectureManager(self._initial_manager.to_architecture_model())
        self._adaptive = AdaptiveArchitecture(self._manager)
        self._step_count = 0
        self._terminated = False
        self._truncated = False

        self._rebuild()
        self._initialize_evaluation()
        observation = self._current_observation()
        info = self._current_info(
            action_id=None,
            action=None,
            transition_result=None,
        )

        return observation, info

    def observation_space(self) -> None:
        """Placeholder for a Gymnasium observation space.

        This baseline returns a plain Python dict observation. A concrete
        Gymnasium space can be added later without changing the environment
        contract implemented here.
        """
        return None

    def action_space(self) -> None:
        """Placeholder for a Gymnasium action space.

        This baseline uses a discrete action space derived from the current
        architecture action set. A concrete Gymnasium space can be added
        later.
        """
        return None

    def step(self, action_id: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one architecture transition.

        Parameters
        ----------
        action_id:
            Integer action selected by the agent/controller.

        Returns
        -------
        observation:
            Deterministic observation of the resulting architecture. If the
            action is invalid, this is the unchanged architecture observation.
        reward:
            Baseline reward for the transition.
        terminated:
            True when the episode reached a terminal condition.
        truncated:
            True when the episode exceeded max_steps.
        info:
            Safe debug information.
        """
        if self._terminated or self._truncated:
            raise RuntimeError("episode has already terminated/truncated; call reset()")

        action_id_int = action_id
        action = self._mapper.decode(action_id_int)
        transition_result = self._apply_action(action, action_id=action_id_int)
        transition_result["action_id"] = action_id_int
        self._step_count += 1

        if self._step_count >= self._max_steps:
            self._truncated = True

        observation = self._current_observation()
        reward = self._compute_reward(transition_result)
        info = self._current_info(
            action_id=action_id_int,
            action=action,
            transition_result=transition_result,
        )

        if transition_result.get("valid"):
            self._current_evaluation = self._evaluator.evaluate(self._manager.get_architecture())

        terminated = self._terminated
        truncated = self._truncated
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize_evaluation(self) -> None:
        """Evaluate the current architecture and store the result.

        This creates the baseline evaluation used by the reward calculator.
        It is called after reset and after any state rebuild that changes the
        underlying architecture.
        """
        architecture = self._manager.get_architecture()
        self._current_evaluation = self._evaluator.evaluate(architecture)

    def _rebuild(self) -> None:
        """Rebuild encoder and action mapping from the current architecture."""
        architecture = self._manager.get_architecture()
        self._encoder = ArchitectureStateEncoder(architecture)
        self._mapper = ArchitectureActionMapper.from_manager(
            self._manager, role_options=self._role_options
        )

    def _current_observation(self) -> Dict:
        return self._encoder.encode()

    def _current_info(
        self,
        *,
        action_id: Optional[int],
        action: Optional[ArchitectureAction],
        transition_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a safe info dict for logging/debugging."""
        architecture = self._manager.get_architecture()
        architecture = self._manager.get_architecture()
        architecture = self._manager.get_architecture()
        info: Dict[str, Any] = {
            "architecture_id": architecture.architecture_id,
            "architecture_version": self._adaptive.version(),
            "step_count": self._step_count,
            "max_steps": self._max_steps,
            "action_id": action_id,
            "action": action.serialize() if action is not None else None,
            "active_agent_count": architecture.active_agent_count,
            "agent_count": architecture.agent_count,
            "action_space_size": self._mapper.action_count,
            "role_options": self._role_options,
        }
        if transition_result is not None:
            info["transition"] = transition_result

        # Attach Step 9 evaluation information safely.
        evaluation = self._current_evaluation
        if evaluation is None:
            return info

        current_score = self._evaluator.evaluate(architecture).overall_score
        info["evaluation"] = {
            "architecture_id": evaluation.architecture_id,
            "architecture_version": evaluation.architecture_version,
            "previous_score": evaluation.overall_score,
            "current_score": current_score,
            "validity_score": evaluation.validity_score,
            "efficiency_score": evaluation.efficiency_score,
            "communication_cost": evaluation.communication_cost,
            "active_agent_count": evaluation.active_agent_count,
            "edge_count": evaluation.edge_count,
            "overall_score": current_score,
            "task_success_score": evaluation.task_success_score,
        }
        return info

    def _apply_action(self, action: ArchitectureAction, *, action_id: int) -> Dict[str, Any]:
        """Attempt to apply *action* and return structured transition metadata.

        The input *action* is assumed to have already been drawn from the
        current environment action mapping. Even so, the action may be
        rejected by the underlying architecture layer (for example because the
        architecture changed between mapping construction and step execution,
        or because the action is invalid for the current architecture).

        On success, the architecture, version, and history are updated and
        the returned metadata marks the transition as valid.

        On failure, the architecture is unchanged and the returned metadata
        marks the transition as invalid.
        """
        previous_architecture = self._manager.to_architecture_model()
        previous_version = self._adaptive.version()

        try:
            new_architecture = self._adaptive.step(action)
        except (TypeError, ValueError) as exc:
            return {
                "valid": False,
                "action": action.serialize(),
                "error": str(exc),
                "architecture_id": previous_architecture.architecture_id,
                "architecture_version": previous_version,
                "action_id": action_id,
            }

        # Defensive check: the adaptive layer already validated, but confirm
        # here so the environment can report structured results.
        if not new_architecture.is_valid:
            return {
                "valid": False,
                "action": action.serialize(),
                "error": "resulting architecture is invalid",
                "architecture_id": previous_architecture.architecture_id,
                "architecture_version": previous_version,
                "action_id": action_id,
            }

        return {
            "valid": True,
            "action": action.serialize(),
            "previous_architecture_id": previous_architecture.architecture_id,
            "architecture_id": new_architecture.architecture_id,
            "previous_version": previous_version,
            "architecture_version": self._adaptive.version(),
            "step": self._adaptive.version(),
            "action_id": action_id,
        }

    def evaluate_architecture(self, architecture: MASArchitecture) -> EvaluationResult:
        """Evaluate *architecture* using the current evaluator.

        This is a small public helper for tests and for callers that want
        direct access to the Step 9 evaluator without going through a step.
        """
        return self._evaluator.evaluate(architecture)

    def _compute_reward(self, transition_result: Dict[str, Any]) -> float:
        """Compute the reward using the Step 9 RewardCalculator.

        On a valid transition this evaluates the current architecture and
        computes the evaluation delta against the previous evaluation stored
        in the environment. On an invalid/rejected transition it uses the
        RewardCalculator invalid-transition penalty.

        The reward formula itself lives in app/evaluation/reward.py; this
        environment only delegates to it.
        """
        previous = self._current_evaluation
        if previous is None:
            # Should not happen in normal use after reset, but keep this
            # defensive so the environment remains robust.
            previous = self._evaluator.evaluate(self._manager.get_architecture())
            self._current_evaluation = previous

        current = self._evaluator.evaluate(self._manager.get_architecture())
        valid_transition = bool(transition_result.get("valid"))
        reward = self._reward_calculator.calculate(
            previous_result=previous,
            current_result=current,
            valid_transition=valid_transition,
        )
        return reward

    # ------------------------------------------------------------------
    # Public accessors for inspection / tests
    # ------------------------------------------------------------------

    @property
    def manager(self) -> ArchitectureManager:
        """Current architecture manager (live, mutated by step)."""
        return self._manager

    @property
    def adaptive(self) -> AdaptiveArchitecture:
        """Current adaptive architecture wrapper."""
        return self._adaptive

    @property
    def encoder(self) -> ArchitectureStateEncoder:
        """Current state encoder."""
        return self._encoder

    @property
    def mapper(self) -> ArchitectureActionMapper:
        """Current action mapper."""
        return self._mapper

    @property
    def max_steps(self) -> int:
        """Episode step limit."""
        return self._max_steps

    @property
    def step_count(self) -> int:
        """Number of steps taken in the current episode."""
        return self._step_count

    @property
    def terminated(self) -> bool:
        """Episode termination flag."""
        return self._terminated

    @property
    def truncated(self) -> bool:
        """Episode truncation flag (max steps reached)."""
        return self._truncated

    def get_possible_actions(self) -> List[ArchitectureAction]:
        """Return the current valid action set through the manager."""
        return self._manager.get_possible_actions(role_options=self._role_options)

    def encode_action(self, action: ArchitectureAction) -> int:
        """Public alias for mapper.encode for environment-style callers.

        This exists for convenience and is equivalent to ``self.mapper.encode``
        for the current action mapping.
        """
        return self._mapper.encode(action)

    def decode_action(self, action_id: int) -> ArchitectureAction:
        """Public alias for mapper.decode for environment-style callers."""
        return self._mapper.decode(action_id)

    def encode_current_action_id(self, action: ArchitectureAction) -> int:
        """Encode *action* against the *current* environment mapper.

        This is the method used internally by ``step(...)``. In the baseline,
        the set of valid actions can change after a transition, so an action
        that exists in the mapping used to select an action id may not exist
        in the mapping used at step time. This method makes that relationship
        explicit.
        """
        return self._mapper.encode(action)

    # ------------------------------------------------------------------
    # Step 9 inspection accessors
    # ------------------------------------------------------------------

    @property
    def evaluator(self) -> ArchitectureEvaluator:
        """Current architecture evaluator."""
        return self._evaluator

    @property
    def reward_calculator(self) -> RewardCalculator:
        """Current reward calculator."""
        return self._reward_calculator

    @property
    def current_evaluation(self) -> EvaluationResult:
        """Current architecture evaluation result.

        This is set after reset and updated after successful transitions.
        It is unchanged by invalid/rejected transitions.
        """
        if self._current_evaluation is None:
            raise RuntimeError("current evaluation has not been initialized; call reset()")
        return self._current_evaluation
