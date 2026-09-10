"""
Tabular Q-learning implementation for the adaptive MAS research project.

This module provides a simple learned RL baseline using tabular Q-learning.
It is designed to be compatible with the existing BasePolicy interface and
RLController, while remaining separate from the Meta-RL system that will
be implemented later.

Design notes
------------
* This is a BASELINE implementation only. It does NOT implement:
  - Meta-RL or meta-learning
  - Neural networks or policy networks
  - PPO, DQN, or policy gradients
  - Dynamic LangGraph rebuilding
  - Persistent memory
  - LLM reward judges
  - Web search
  - Actual task-success reward

* The Q-learning agent uses:
  - A tabular Q-table (dictionary-based)
  - Epsilon-greedy action selection
  - Standard Q-learning update rule
  - Configurable hyperparameters (alpha, gamma, epsilon, decay)

* State representation:
  - Uses a deterministic hashable state key derived from the architecture
  - Captures active agents, roles, and communication topology
  - Equivalent architectures produce identical state keys

* Action representation:
  - Uses integer action IDs from ArchitectureActionMapper
  - Only valid action IDs are considered for selection
  - Unknown actions are initialized with a default Q-value

Limitations
-----------
* The state space is discretized and may not capture all architectural nuances.
* The Q-table can grow large with complex architectures.
* This is a simple baseline and is not expected to match Meta-RL performance.
* The reward remains the structural evaluation-based reward from Step 9.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.architecture.actions import ArchitectureAction
from app.rl.environment import MASArchitectureEnv
from app.rl.policy import BasePolicy
from app.rl.state import ArchitectureStateEncoder


@dataclass
class QTable:
    """
    Tabular Q-table for storing state-action values.

    The Q-table is a dictionary mapping (state_key, action_id) tuples
    to Q-values (floats). It supports:

    - Initialization of new state-action pairs
    - Retrieval of Q-values (with default for unseen pairs)
    - Update via the Q-learning rule
    - Inspection and export

    Attributes
    ----------
    table : Dict[Tuple, float]
        The underlying dictionary storing Q-values.
    default_value : float
        Default Q-value for unseen state-action pairs.
    """

    table: Dict[Tuple, float] = field(default_factory=dict)
    default_value: float = 0.0

    def get(self, state_key: Tuple, action_id: int) -> float:
        """
        Get the Q-value for a state-action pair.

        Parameters
        ----------
        state_key : Tuple
            The hashable state representation.
        action_id : int
            The action ID.

        Returns
        -------
        float
            The Q-value, or default_value if unseen.
        """
        return self.table.get((state_key, action_id), self.default_value)

    def set(self, state_key: Tuple, action_id: int, value: float) -> None:
        """
        Set the Q-value for a state-action pair.

        Parameters
        ----------
        state_key : Tuple
            The hashable state representation.
        action_id : int
            The action ID.
        value : float
            The Q-value to set.
        """
        self.table[(state_key, action_id)] = value

    def update(
        self,
        state_key: Tuple,
        action_id: int,
        reward: float,
        next_state_key: Tuple,
        next_valid_actions: List[int],
        alpha: float,
        gamma: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Perform a Q-learning update.

        Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))

        For terminal/truncated states, max_a' Q(s',a') is 0.

        Parameters
        ----------
        state_key : Tuple
            Current state key.
        action_id : int
            Action taken.
        reward : float
            Reward received.
        next_state_key : Tuple
            Next state key.
        next_valid_actions : List[int]
            Valid actions in the next state.
        alpha : float
            Learning rate (0 < alpha <= 1).
        gamma : float
            Discount factor (0 <= gamma <= 1).
        terminated : bool
            Whether the episode terminated.
        truncated : bool
            Whether the episode truncated.
        """
        current_q = self.get(state_key, action_id)

        # For terminal/truncated states, future value is 0
        if terminated or truncated:
            max_next_q = 0.0
        else:
            # Get max Q-value over all valid next actions
            if next_valid_actions:
                max_next_q = max(self.get(next_state_key, a) for a in next_valid_actions)
            else:
                max_next_q = 0.0

        # Q-learning update
        td_target = reward + gamma * max_next_q
        td_error = td_target - current_q
        new_q = current_q + alpha * td_error

        self.set(state_key, action_id, new_q)

    def get_state_actions(self, state_key: Tuple) -> Dict[int, float]:
        """
        Get all Q-values for a given state.

        Parameters
        ----------
        state_key : Tuple
            The state key.

        Returns
        -------
        Dict[int, float]
            Dictionary mapping action_id -> Q-value for this state.
        """
        return {
            action_id: q_value
            for (s, action_id), q_value in self.table.items()
            if s == state_key
        }

    def get_all_states(self) -> List[Tuple]:
        """
        Get all state keys present in the Q-table.

        Returns
        -------
        List[Tuple]
            List of unique state keys.
        """
        return list(set(s for s, _ in self.table.keys()))

    def to_dict(self) -> Dict:
        """
        Export the Q-table as a dictionary for inspection/serialization.

        Returns
        -------
        Dict
            Dictionary with 'default_value' and 'entries' (list of dicts).
        """
        # Convert state_key tuples to lists for JSON serialization
        # State keys are tuples of (activity_vector, role_vector, agent_ids)
        # Each component is already a tuple, so we need to convert them to lists
        def convert_state_key(key: Tuple) -> List:
            """Convert a state key tuple to a list for JSON serialization."""
            # key is (activity_vector, role_vector, agent_ids)
            # Each component is a tuple that needs to be converted to a list
            return [list(component) for component in key]
        
        return {
            "default_value": self.default_value,
            "entries": [
                {"state_key": convert_state_key(s), "action_id": a, "q_value": q}
                for (s, a), q in self.table.items()
            ]
        }

    def clear(self) -> None:
        """Clear all entries from the Q-table."""
        self.table.clear()


@dataclass
class StateEncoder:
    """
    Deterministic state encoder for tabular Q-learning.

    Converts an ArchitectureStateEncoder observation into a hashable,
    comparable state key suitable for use in a Q-table.

    The state key captures the essential architectural information:
    1. Active agent configuration (activity vector)
    2. Agent roles (role vector)
    3. Communication topology (adjacency matrix as tuple of tuples)

    This ensures that equivalent architectures produce identical keys,
    while different architectures produce different keys.

    Attributes
    ----------
    None (stateless encoder)
    """

    def encode(self, observation: Dict) -> Tuple:
        """
        Encode an observation into a hashable state key.

        Parameters
        ----------
        observation : Dict
            Observation from ArchitectureStateEncoder.encode().

        Returns
        -------
        Tuple
            Hashable state key representing the architecture state.

        The state key is a tuple of:
        - activity_vector: tuple of 0/1 for each agent
        - role_vector: tuple of role strings
        - agent_ids: tuple of agent ID strings (for identification)

        Note: We exclude the adjacency matrix from the state key because
        the ArchitectureStateEncoder already encodes this information
        implicitly through the activity_vector and role_vector. The
        adjacency matrix is deterministic based on the architecture and
        doesn't need to be part of the state key for Q-learning purposes.
        """
        # Extract components from observation
        activity_vector = tuple(observation.get("activity_vector", []))
        role_vector = tuple(observation.get("role_vector", []))
        agent_ids = tuple(observation.get("agent_ids", []))

        # Create a composite state key using only hashable components
        # Lists are converted to tuples for hashability
        state_key = (activity_vector, role_vector, agent_ids)

        return state_key

    def encode_from_env(self, env: MASArchitectureEnv) -> Tuple:
        """
        Convenience method to encode the current environment state.

        Parameters
        ----------
        env : MASArchitectureEnv
            The environment to encode.

        Returns
        -------
        Tuple
            State key for the current architecture.
        """
        observation = env.encoder.encode()
        return self.encode(observation)


class QLearningPolicy(BasePolicy):
    """
    Tabular Q-learning policy for architecture adaptation.

    This policy implements epsilon-greedy action selection using a Q-table.
    It is compatible with the BasePolicy interface and can be used with
    RLController.

    The policy:
    1. Encodes the current observation into a state key
    2. With probability epsilon: selects a random valid action (explore)
    3. Otherwise: selects the valid action with highest Q-value (exploit)
    4. Uses deterministic tie-breaking for exploitation

    Hyperparameters:
    - epsilon: exploration rate (0 = pure exploitation, 1 = pure exploration)
    - epsilon_min: minimum epsilon value
    - epsilon_decay: multiplicative decay factor per episode (0 < decay <= 1)
    - alpha: learning rate (for Q-learning updates)
    - gamma: discount factor (for Q-learning updates)

    Attributes
    ----------
    q_table : QTable
        The tabular Q-values.
    state_encoder : StateEncoder
        Converts observations to state keys.
    epsilon : float
        Current exploration rate.
    epsilon_min : float
        Minimum exploration rate.
    epsilon_decay : float
        Decay factor for epsilon.
    alpha : float
        Learning rate for Q-learning updates.
    gamma : float
        Discount factor for Q-learning updates.
    rng : random.Random
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        default_q_value: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the Q-learning policy.

        Parameters
        ----------
        alpha : float, optional
            Learning rate (default: 0.1).
        gamma : float, optional
            Discount factor (default: 0.9).
        epsilon : float, optional
            Initial exploration rate (default: 1.0).
        epsilon_min : float, optional
            Minimum exploration rate (default: 0.01).
        epsilon_decay : float, optional
            Multiplicative decay factor per episode (default: 0.995).
        default_q_value : float, optional
            Default Q-value for unseen state-action pairs (default: 0.0).
        seed : int, optional
            Random seed for reproducibility.
        """
        # Validate hyperparameters
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not (0 <= gamma <= 1):
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if not (0 <= epsilon <= 1):
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if not (0 <= epsilon_min <= 1):
            raise ValueError(f"epsilon_min must be in [0, 1], got {epsilon_min}")
        if not (0 < epsilon_decay <= 1):
            raise ValueError(f"epsilon_decay must be in (0, 1], got {epsilon_decay}")

        self.q_table = QTable(default_value=default_q_value)
        self.state_encoder = StateEncoder()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)

    def select_action(
        self,
        observation: Dict,
        valid_action_ids: List[int],
    ) -> int:
        """
        Select an action using epsilon-greedy policy.

        Parameters
        ----------
        observation : Dict
            Current architecture observation.
        valid_action_ids : List[int]
            List of currently valid action IDs.

        Returns
        -------
        int
            The selected action ID.

        Raises
        ------
        ValueError
            If valid_action_ids is empty.
        """
        if not valid_action_ids:
            raise ValueError("no valid actions available")

        # Encode the current state
        state_key = self.state_encoder.encode(observation)

        # Epsilon-greedy action selection
        if self.rng.random() < self.epsilon:
            # Explore: select random valid action
            return self.rng.choice(valid_action_ids)
        else:
            # Exploit: select action with highest Q-value
            return self._select_best_action(state_key, valid_action_ids)

    def _select_best_action(
        self,
        state_key: Tuple,
        valid_action_ids: List[int],
    ) -> int:
        """
        Select the valid action with the highest Q-value for the state.

        Uses deterministic tie-breaking: if multiple actions have the same
        Q-value, the one with the smallest action_id is selected.

        Parameters
        ----------
        state_key : Tuple
            Current state key.
        valid_action_ids : List[int]
            List of valid action IDs.

        Returns
        -------
        int
            The action ID with the highest Q-value.
        """
        # Get Q-values for all valid actions
        q_values = [(action_id, self.q_table.get(state_key, action_id))
                    for action_id in valid_action_ids]

        # Sort by Q-value (descending), then by action_id (ascending) for tie-breaking
        q_values.sort(key=lambda x: (-x[1], x[0]))

        # Return the action_id of the best action
        return q_values[0][0]

    def update(
        self,
        state_key: Tuple,
        action_id: int,
        reward: float,
        next_state_key: Tuple,
        next_valid_actions: List[int],
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Update the Q-table using the Q-learning rule.

        Parameters
        ----------
        state_key : Tuple
            Current state key.
        action_id : int
            Action taken.
        reward : float
            Reward received.
        next_state_key : Tuple
            Next state key.
        next_valid_actions : List[int]
            Valid actions in the next state.
        terminated : bool
            Whether the episode terminated.
        truncated : bool
            Whether the episode truncated.
        """
        self.q_table.update(
            state_key=state_key,
            action_id=action_id,
            reward=reward,
            next_state_key=next_state_key,
            next_valid_actions=next_valid_actions,
            alpha=self.alpha,
            gamma=self.gamma,
            terminated=terminated,
            truncated=truncated,
        )

    def decay_epsilon(self) -> None:
        """
        Decay the epsilon value by the decay factor.

        This is called after each episode to reduce exploration over time.
        Epsilon will not go below epsilon_min.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_for_episode(self) -> None:
        """
        Reset any episode-specific state.

        Currently, this just decays epsilon. Override in subclasses if needed.
        """
        self.decay_epsilon()

    def get_state_key(self, observation: Dict) -> Tuple:
        """
        Get the state key for an observation.

        Parameters
        ----------
        observation : Dict
            Architecture observation.

        Returns
        -------
        Tuple
            State key.
        """
        return self.state_encoder.encode(observation)

    def get_q_value(self, state_key: Tuple, action_id: int) -> float:
        """
        Get the Q-value for a state-action pair.

        Parameters
        ----------
        state_key : Tuple
            State key.
        action_id : int
            Action ID.

        Returns
        -------
        float
            Q-value.
        """
        return self.q_table.get(state_key, action_id)

    def set_q_value(self, state_key: Tuple, action_id: int, value: float) -> None:
        """
        Set the Q-value for a state-action pair.

        Parameters
        ----------
        state_key : Tuple
            State key.
        action_id : int
            Action ID.
        value : float
            Q-value to set.
        """
        self.q_table.set(state_key, action_id, value)

    def get_q_table(self) -> QTable:
        """
        Get the Q-table for inspection.

        Returns
        -------
        QTable
            The Q-table.
        """
        return self.q_table

    def clear_q_table(self) -> None:
        """Clear the Q-table."""
        self.q_table.clear()

    def save_q_table(self, filepath: str) -> None:
        """
        Save the Q-table to a JSON file.

        Parameters
        ----------
        filepath : str
            Path to save the Q-table.
        """
        import json

        data = self.q_table.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_q_table(self, filepath: str) -> None:
        """
        Load a Q-table from a JSON file.

        Parameters
        ----------
        filepath : str
            Path to load the Q-table from.
        """
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.q_table.clear()
        self.q_table.default_value = data.get("default_value", 0.0)
        for entry in data.get("entries", []):
            # Convert state_key from list back to tuple of tuples
            # state_key is [activity_vector, role_vector, agent_ids]
            # Each component is a list that needs to be converted to a tuple
            state_key_list = entry["state_key"]
            state_key = tuple(tuple(component) for component in state_key_list)
            action_id = entry["action_id"]
            q_value = entry["q_value"]
            self.q_table.set(state_key, action_id, q_value)


@dataclass
class TrainingStats:
    """
    Statistics collected during Q-learning training.

    Attributes
    ----------
    episode : int
        Episode number.
    total_reward : float
        Total reward for the episode.
    episode_length : int
        Number of steps in the episode.
    final_score : Optional[float]
        Final architecture evaluation score (if available).
    terminated : bool
        Whether the episode terminated.
    truncated : bool
        Whether the episode truncated.
    """

    episode: int
    total_reward: float
    episode_length: int
    final_score: Optional[float] = None
    terminated: bool = False
    truncated: bool = False


class QLearningTrainer:
    """
    Training loop for Q-learning policy.

    This trainer runs multiple episodes of interaction between the Q-learning
    policy and the environment, updating the Q-table after each step.

    The trainer is kept separate from RLController to maintain modularity:
    - RLController is for running episodes with any policy
    - QLearningTrainer is specifically for training Q-learning policies

    Both can use the same BasePolicy interface.

    Attributes
    ----------
    env : MASArchitectureEnv
        The environment.
    policy : QLearningPolicy
        The Q-learning policy to train.
    stats : List[TrainingStats]
        Training statistics for each episode.
    """

    def __init__(
        self,
        env: MASArchitectureEnv,
        policy: QLearningPolicy,
    ) -> None:
        """
        Initialize the trainer.

        Parameters
        ----------
        env : MASArchitectureEnv
            The environment to train in.
        policy : QLearningPolicy
            The Q-learning policy to train.
        """
        self.env = env
        self.policy = policy
        self.stats: List[TrainingStats] = []

    def train_episode(self) -> TrainingStats:
        """
        Run a single training episode.

        Returns
        -------
        TrainingStats
            Statistics for the episode.
        """
        # Reset environment
        observation, info = self.env.reset()

        # Encode initial state
        current_state_key = self.policy.get_state_key(observation)
        current_valid_actions = list(range(self.env.mapper.action_count))

        total_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Select action using current policy
            action_id = self.policy.select_action(
                observation=observation,
                valid_action_ids=current_valid_actions,
            )

            # Execute action
            next_observation, reward, terminated, truncated, step_info = (
                self.env.step(action_id)
            )

            # Encode next state
            next_state_key = self.policy.get_state_key(next_observation)
            next_valid_actions = list(range(self.env.mapper.action_count))

            # Update Q-table
            self.policy.update(
                state_key=current_state_key,
                action_id=action_id,
                reward=reward,
                next_state_key=next_state_key,
                next_valid_actions=next_valid_actions,
                terminated=terminated,
                truncated=truncated,
            )

            # Update tracking variables
            total_reward += reward
            step_count += 1

            # Move to next state
            observation = next_observation
            current_state_key = next_state_key
            current_valid_actions = next_valid_actions

        # Decay epsilon after episode
        self.policy.reset_for_episode()

        # Get final evaluation score if available
        final_score = None
        try:
            evaluation = self.env.current_evaluation
            final_score = evaluation.overall_score
        except (RuntimeError, AttributeError):
            pass

        # Record stats
        stats = TrainingStats(
            episode=len(self.stats) + 1,
            total_reward=total_reward,
            episode_length=step_count,
            final_score=final_score,
            terminated=terminated,
            truncated=truncated,
        )
        self.stats.append(stats)

        return stats

    def train(self, num_episodes: int) -> List[TrainingStats]:
        """
        Train for multiple episodes.

        Parameters
        ----------
        num_episodes : int
            Number of episodes to train.

        Returns
        -------
        List[TrainingStats]
            Statistics for all episodes.
        """
        for _ in range(num_episodes):
            self.train_episode()

        return self.stats

    def get_average_reward(self, window: int = 10) -> Optional[float]:
        """
        Get the average reward over the last N episodes.

        Parameters
        ----------
        window : int, optional
            Number of episodes to average over (default: 10).

        Returns
        -------
        Optional[float]
            Average reward, or None if not enough episodes.
        """
        if len(self.stats) < window:
            return None

        recent = self.stats[-window:]
        return sum(s.total_reward for s in recent) / window

    def get_average_score(self, window: int = 10) -> Optional[float]:
        """
        Get the average final score over the last N episodes.

        Parameters
        ----------
        window : int, optional
            Number of episodes to average over (default: 10).

        Returns
        -------
        Optional[float]
            Average score, or None if not enough episodes.
        """
        if len(self.stats) < window:
            return None

        recent = [s for s in self.stats[-window:] if s.final_score is not None]
        if not recent:
            return None

        return sum(s.final_score for s in recent) / len(recent)
