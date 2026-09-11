"""
Meta-RL task distribution and sampler for the adaptive MAS research project.

This module provides:
* MetaTaskDistribution: A validated collection of MetaTask objects.
* MetaTaskSampler: A deterministic sampler for sampling tasks from a distribution.

These components enable the Meta-RL formulation where:

    Task τ = a particular MAS adaptation problem.

    Task distribution p(τ) = a collection/distribution of different adaptation
    tasks.

    Training tasks: τ₁, τ₂, τ₃, ... (used for future Meta-RL learning)

    Unseen evaluation tasks: τₙ₊₁, τₙ₊₂, ... (used for future adaptation /
    evaluation)

    Future Meta-RL objective: learn an adaptation strategy that generalizes
    across tasks and adapts quickly to a new task.

This module does NOT implement:
* Meta-RL training loops (inner-loop / outer-loop).
* Neural networks, PPO, DQN, or policy gradients.
* LLM calls or API interactions.
* Persistent memory.
* Actual task execution or task-performance measurement.

Design notes
------------

* The distribution is a simple in-memory collection with validation.
* Sampling is deterministic when a seed is provided.
* The split operation creates non-overlapping train/test subsets.
* The baseline task set provides representative tasks from multiple categories
  for development and testing purposes only.
* No claims are made that these tasks represent real LLM task performance.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from app.architecture.manager import ArchitectureManager
from app.rl.meta_environment import MetaEnvironment
from app.rl.meta_task import MetaTask


# ---------------------------------------------------------------------------
# Baseline task set
# ---------------------------------------------------------------------------

def create_baseline_task_distribution() -> MetaTaskDistribution:
    """Create a baseline MetaTaskDistribution with representative tasks.

    This creates a small set of tasks across multiple categories for
    development and testing. These tasks are baseline definitions for the
    future Meta-RL environment and do NOT claim to represent real LLM task
    performance.

    Categories included:
    * coding
    * research
    * analysis
    * summarization
    * reasoning

    Returns
    -------
    MetaTaskDistribution
        A distribution containing the baseline tasks.
    """
    tasks = [
        # Coding tasks
        MetaTask(
            task_id="task-coding-001",
            task_description="Write a Python function to sort a list using quicksort",
            task_category="coding",
            required_capabilities=["coding", "testing"],
            difficulty=2,
            context_features={
                "language": "python",
                "function_required": True,
                "algorithm_complexity": "medium",
            },
        ),
        MetaTask(
            task_id="task-coding-002",
            task_description="Implement a binary search tree with insert and search operations",
            task_category="coding",
            required_capabilities=["coding", "data_structures"],
            difficulty=3,
            context_features={
                "language": "python",
                "data_structure": "binary_search_tree",
                "operations": ["insert", "search"],
            },
        ),
        MetaTask(
            task_id="task-coding-003",
            task_description="Create a REST API endpoint for user authentication",
            task_category="coding",
            required_capabilities=["coding", "api_design", "security"],
            difficulty=4,
            context_features={
                "language": "python",
                "framework": "fastapi",
                "endpoint_type": "authentication",
            },
        ),

        # Research tasks
        MetaTask(
            task_id="task-research-001",
            task_description="Research the latest developments in quantum computing",
            task_category="research",
            required_capabilities=["research", "information_synthesis"],
            difficulty=3,
            context_features={
                "domain": "quantum_computing",
                "timeframe": "latest",
                "output_format": "structured_report",
            },
        ),
        MetaTask(
            task_id="task-research-002",
            task_description="Find and summarize recent papers on transformer architectures",
            task_category="research",
            required_capabilities=["research", "summarization"],
            difficulty=2,
            context_features={
                "domain": "machine_learning",
                "topic": "transformers",
                "output_format": "summary",
            },
        ),
        MetaTask(
            task_id="task-research-003",
            task_description="Compare different approaches to reinforcement learning from human feedback",
            task_category="research",
            required_capabilities=["research", "comparison", "analysis"],
            difficulty=4,
            context_features={
                "domain": "machine_learning",
                "topic": "RLHF",
                "output_format": "comparison_matrix",
            },
        ),

        # Analysis tasks
        MetaTask(
            task_id="task-analysis-001",
            task_description="Analyze a dataset of customer reviews and identify sentiment patterns",
            task_category="analysis",
            required_capabilities=["analysis", "data_interpretation"],
            difficulty=3,
            context_features={
                "data_type": "text",
                "analysis_type": "sentiment",
                "output_format": "patterns_report",
            },
        ),
        MetaTask(
            task_id="task-analysis-002",
            task_description="Examine code for potential security vulnerabilities",
            task_category="analysis",
            required_capabilities=["analysis", "security_audit"],
            difficulty=4,
            context_features={
                "data_type": "code",
                "analysis_type": "security_audit",
                "output_format": "vulnerability_report",
            },
        ),
        MetaTask(
            task_id="task-analysis-003",
            task_description="Evaluate the performance characteristics of different sorting algorithms",
            task_category="analysis",
            required_capabilities=["analysis", "performance_evaluation"],
            difficulty=2,
            context_features={
                "data_type": "algorithm",
                "analysis_type": "performance",
                "output_format": "benchmark_results",
            },
        ),

        # Summarization tasks
        MetaTask(
            task_id="task-summarization-001",
            task_description="Summarize a long research paper into key findings",
            task_category="summarization",
            required_capabilities=["summarization", "information_synthesis"],
            difficulty=3,
            context_features={
                "input_type": "research_paper",
                "output_length": "short",
                "focus": "key_findings",
            },
        ),
        MetaTask(
            task_id="task-summarization-002",
            task_description="Create an executive summary from a detailed technical report",
            task_category="summarization",
            required_capabilities=["summarization", "synthesis"],
            difficulty=2,
            context_features={
                "input_type": "technical_report",
                "output_length": "medium",
                "audience": "executive",
            },
        ),
        MetaTask(
            task_id="task-summarization-003",
            task_description="Condense a lengthy legal document into plain language summary",
            task_category="summarization",
            required_capabilities=["summarization", "simplification"],
            difficulty=4,
            context_features={
                "input_type": "legal_document",
                "output_length": "medium",
                "target_audience": "non_expert",
            },
        ),

        # Reasoning tasks
        MetaTask(
            task_id="task-reasoning-001",
            task_description="Solve a multi-step logical puzzle with constraints",
            task_category="reasoning",
            required_capabilities=["reasoning", "logical_deduction"],
            difficulty=3,
            context_features={
                "problem_type": "logic_puzzle",
                "steps_required": 5,
                "constraint_type": "combinatorial",
            },
        ),
        MetaTask(
            task_id="task-reasoning-002",
            task_description="Determine the optimal strategy for a resource allocation problem",
            task_category="reasoning",
            required_capabilities=["reasoning", "optimization"],
            difficulty=4,
            context_features={
                "problem_type": "optimization",
                "constraints": ["budget", "time"],
                "objective": "maximize_utility",
            },
        ),
        MetaTask(
            task_id="task-reasoning-003",
            task_description="Identify the root cause of a system failure from symptom descriptions",
            task_category="reasoning",
            required_capabilities=["reasoning", "diagnostic"],
            difficulty=3,
            context_features={
                "problem_type": "diagnostic",
                "domain": "system_administration",
                "output_format": "root_cause_analysis",
            },
        ),
    ]
    return MetaTaskDistribution(tasks=tasks)


# ---------------------------------------------------------------------------
# MetaTaskDistribution
# ---------------------------------------------------------------------------

class MetaTaskDistribution(BaseModel):
    """A validated collection of MetaTask objects.

    This represents a task distribution p(τ) for Meta-RL. It stores a
    collection of tasks, prevents duplicate task IDs, and provides methods
    for accessing, sampling, and splitting tasks.

    Parameters
    ----------
    tasks:
        List of MetaTask objects. Must have unique task_ids.
    name:
        Optional name for this distribution (for logging/identification).
    description:
        Optional description of the distribution.
    """

    tasks: List[MetaTask] = Field(
        default_factory=list,
        description="List of MetaTask objects with unique task_ids.",
    )
    name: str = Field(
        default="baseline",
        description="Optional name for this distribution.",
    )
    description: str = Field(
        default="",
        description="Optional description of the distribution.",
    )

    @field_validator("tasks")
    @classmethod
    def tasks_must_have_unique_ids(cls, v: List[MetaTask]) -> List[MetaTask]:
        """Validate that all tasks have unique task_ids."""
        seen_ids: set = set()
        for task in v:
            if task.task_id in seen_ids:
                raise ValueError(
                    f"Duplicate task_id found: '{task.task_id}'. "
                    "All tasks in a distribution must have unique task_ids."
                )
            seen_ids.add(task.task_id)
        return v

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_tasks(self) -> List[MetaTask]:
        """Return a copy of the task list."""
        return list(self.tasks)

    def task_ids(self) -> List[str]:
        """Return a list of all task IDs in deterministic order."""
        return [task.task_id for task in self.tasks]

    def task_count(self) -> int:
        """Return the number of tasks in the distribution."""
        return len(self.tasks)

    def get_task(self, task_id: str) -> MetaTask:
        """Get a task by its ID.

        Parameters
        ----------
        task_id:
            The task ID to look up.

        Returns
        -------
        MetaTask
            The task with the given ID.

        Raises
        ------
        KeyError:
            If no task with the given ID exists.
        """
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(f"No task with id '{task_id}'")

    def get_tasks_by_category(self, category: str) -> List[MetaTask]:
        """Get all tasks in a specific category.

        Parameters
        ----------
        category:
            The category to filter by.

        Returns
        -------
        List[MetaTask]
            Tasks matching the category.
        """
        return [task for task in self.tasks if task.task_category == category]

    def get_categories(self) -> List[str]:
        """Return a sorted list of all categories in the distribution."""
        categories = {task.task_category for task in self.tasks}
        return sorted(categories)

    def get_tasks_by_difficulty(self, difficulty: int) -> List[MetaTask]:
        """Get all tasks with a specific difficulty rating.

        Parameters
        ----------
        difficulty:
            The difficulty rating to filter by (1-5).

        Returns
        -------
        List[MetaTask]
            Tasks with the specified difficulty.
        """
        return [task for task in self.tasks if task.difficulty == difficulty]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        *,
        n: int = 1,
        seed: Optional[int] = None,
        category: Optional[str] = None,
        difficulty: Optional[int] = None,
    ) -> List[MetaTask]:
        """Sample tasks from the distribution.

        Parameters
        ----------
        n:
            Number of tasks to sample.
        seed:
            Optional random seed for reproducibility. When provided, sampling
            is deterministic.
        category:
            If provided, only sample from tasks in this category.
        difficulty:
            If provided, only sample from tasks with this difficulty.

        Returns
        -------
        List[MetaTask]
            Sampled tasks. When n=1, returns a list with one task. When
            sampling without replacement and n > task count, returns all
            available tasks.

        Raises
        ------
        ValueError:
            If n < 1 or if no tasks match the filter criteria.
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        # Filter tasks by category and/or difficulty
        candidate_tasks = list(self.tasks)
        if category is not None:
            candidate_tasks = [
                task for task in candidate_tasks
                if task.task_category == category
            ]
        if difficulty is not None:
            candidate_tasks = [
                task for task in candidate_tasks
                if task.difficulty == difficulty
            ]

        if not candidate_tasks:
            filter_desc = []
            if category is not None:
                filter_desc.append(f"category='{category}'")
            if difficulty is not None:
                filter_desc.append(f"difficulty={difficulty}")
            filter_str = f" with {', '.join(filter_desc)}" if filter_desc else ""
            raise ValueError(
                f"No tasks found in distribution '{self.name}'{filter_str}"
            )

        # Sample without replacement
        if n >= len(candidate_tasks):
            return list(candidate_tasks)

        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        sampled = rng.sample(candidate_tasks, n)
        return sampled

    def sample_one(
        self,
        *,
        seed: Optional[int] = None,
        category: Optional[str] = None,
        difficulty: Optional[int] = None,
    ) -> MetaTask:
        """Sample a single task from the distribution.

        Convenience method equivalent to sample(n=1, ...)[0].

        Parameters
        ----------
        seed:
            Optional random seed for reproducibility.
        category:
            If provided, only sample from tasks in this category.
        difficulty:
            If provided, only sample from tasks with this difficulty.

        Returns
        -------
        MetaTask
            A single sampled task.
        """
        sampled = self.sample(
            n=1,
            seed=seed,
            category=category,
            difficulty=difficulty,
        )
        return sampled[0]

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def split(
        self,
        *,
        train_ratio: float = 0.8,
        seed: Optional[int] = None,
    ) -> Tuple[MetaTaskDistribution, MetaTaskDistribution]:
        """Split the distribution into train and test subsets.

        The split is deterministic when a seed is provided. Tasks are
        shuffled (optionally with a seed) and divided according to the
        train_ratio.

        Parameters
        ----------
        train_ratio:
            Fraction of tasks to use for training (0 < train_ratio < 1).
            Must result in at least 1 task in each split.
        seed:
            Optional random seed for reproducibility.

        Returns
        -------
        Tuple[MetaTaskDistribution, MetaTaskDistribution]
            (train_distribution, test_distribution)

        Raises
        ------
        ValueError:
            If train_ratio is not in (0, 1) or if the split would result
            in an empty train or test set.
        """
        if not (0 < train_ratio < 1):
            raise ValueError("train_ratio must be in (0, 1)")

        total_tasks = len(self.tasks)
        if total_tasks < 2:
            raise ValueError(
                "Cannot split a distribution with fewer than 2 tasks. "
                "At least 2 tasks are required for a train/test split."
            )

        train_count = int(total_tasks * train_ratio)
        test_count = total_tasks - train_count

        if train_count < 1:
            raise ValueError(
                f"train_ratio={train_ratio} results in 0 training tasks. "
                "Increase train_ratio or add more tasks."
            )
        if test_count < 1:
            raise ValueError(
                f"train_ratio={train_ratio} results in 0 test tasks. "
                "Decrease train_ratio or add more tasks."
            )

        # Create a shuffled copy of tasks
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        shuffled_tasks = list(self.tasks)
        rng.shuffle(shuffled_tasks)

        train_tasks = shuffled_tasks[:train_count]
        test_tasks = shuffled_tasks[train_count:]

        train_dist = MetaTaskDistribution(
            tasks=train_tasks,
            name=f"{self.name}_train",
            description=f"Training split from '{self.name}' (train_ratio={train_ratio})",
        )
        test_dist = MetaTaskDistribution(
            tasks=test_tasks,
            name=f"{self.name}_test",
            description=f"Test split from '{self.name}' (train_ratio={train_ratio})",
        )

        return train_dist, test_dist

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self) -> Dict[str, Any]:
        """Return a JSON-friendly serialization of the distribution."""
        return {
            "name": self.name,
            "description": self.description,
            "task_count": self.task_count(),
            "categories": self.get_categories(),
            "tasks": [task.serialize() for task in self.tasks],
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def create_environment(
        self,
        task_id: str,
        *,
        role_options: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
    ) -> MetaEnvironment:
        """Create a MetaEnvironment for a specific task in this distribution.

        This is a convenience helper that creates a MetaEnvironment configured
        with the specified task from this distribution.

        Parameters
        ----------
        task_id:
            The ID of the task to use.
        role_options:
            Optional role vocabulary for the environment.
        max_steps:
            Optional maximum steps per episode.

        Returns
        -------
        MetaEnvironment
            A MetaEnvironment configured with the specified task.
        """
        task = self.get_task(task_id)
        manager = ArchitectureManager.create_default_architecture()
        return MetaEnvironment(
            manager,
            role_options=role_options,
            max_steps=max_steps,
            task=task,
        )

    def create_sampler(self, seed: Optional[int] = None) -> MetaTaskSampler:
        """Create a MetaTaskSampler for this distribution.

        Parameters
        ----------
        seed:
            Optional default random seed for the sampler.

        Returns
        -------
        MetaTaskSampler
            A sampler configured for this distribution.
        """
        return MetaTaskSampler(distribution=self, default_seed=seed)


# ---------------------------------------------------------------------------
# MetaTaskSampler
# ---------------------------------------------------------------------------

class MetaTaskSampler(BaseModel):
    """Deterministic sampler for MetaTaskDistribution.

    This sampler provides reproducible sampling from a task distribution.
    It supports sampling with or without a seed, and can filter by category
    or difficulty.

    Parameters
    ----------
    distribution:
        The MetaTaskDistribution to sample from.
    default_seed:
        Optional default random seed. When provided, sampling operations
            that don't specify a seed will use this default.
    """

    distribution: MetaTaskDistribution = Field(
        description="The task distribution to sample from.",
    )
    default_seed: Optional[int] = Field(
        default=None,
        description="Optional default random seed for reproducible sampling.",
    )

    # ------------------------------------------------------------------
    # Sampling methods
    # ------------------------------------------------------------------

    def sample(
        self,
        *,
        n: int = 1,
        seed: Optional[int] = None,
        category: Optional[str] = None,
        difficulty: Optional[int] = None,
    ) -> List[MetaTask]:
        """Sample tasks from the distribution.

        Parameters
        ----------
        n:
            Number of tasks to sample.
        seed:
            Optional random seed. If None, uses default_seed.
        category:
            If provided, only sample from tasks in this category.
        difficulty:
            If provided, only sample from tasks with this difficulty.

        Returns
        -------
        List[MetaTask]
            Sampled tasks.
        """
        if seed is None:
            seed = self.default_seed
        return self.distribution.sample(
            n=n,
            seed=seed,
            category=category,
            difficulty=difficulty,
        )

    def sample_one(
        self,
        *,
        seed: Optional[int] = None,
        category: Optional[str] = None,
        difficulty: Optional[int] = None,
    ) -> MetaTask:
        """Sample a single task from the distribution.

        Parameters
        ----------
        seed:
            Optional random seed.
        category:
            If provided, only sample from tasks in this category.
        difficulty:
            If provided, only sample from tasks with this difficulty.

        Returns
        -------
        MetaTask
            A single sampled task.
        """
        if seed is None:
            seed = self.default_seed
        return self.distribution.sample_one(
            seed=seed,
            category=category,
            difficulty=difficulty,
        )

    def sample_train_test(
        self,
        *,
        train_ratio: float = 0.8,
        seed: Optional[int] = None,
    ) -> Tuple[MetaTaskDistribution, MetaTaskDistribution]:
        """Split the distribution into train and test subsets.

        Parameters
        ----------
        train_ratio:
            Fraction of tasks for training.
        seed:
            Optional random seed. If None, uses default_seed.

        Returns
        -------
        Tuple[MetaTaskDistribution, MetaTaskDistribution]
            (train_distribution, test_distribution)
        """
        if seed is None:
            seed = self.default_seed
        return self.distribution.split(train_ratio=train_ratio, seed=seed)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_distribution(self) -> MetaTaskDistribution:
        """Return the underlying distribution."""
        return self.distribution

    def task_count(self) -> int:
        """Return the number of tasks in the distribution."""
        return self.distribution.task_count()

    def get_categories(self) -> List[str]:
        """Return a sorted list of all categories."""
        return self.distribution.get_categories()
