import pytest

from langchain_openai import ChatOpenAI

from app.agents.critic import Critic, CriticOutput, create_critic_review
from app.agents.coder import Coder, CoderOutput, create_code_solution
from app.agents.finalizer import Finalizer, FinalizerOutput, create_final_answer
from app.agents.planner import Planner, PlannerOutput
from app.agents.researcher import Researcher, ResearcherOutput, create_research_report


# ---------------------------------------------------------------------------
# Planner helpers
# ---------------------------------------------------------------------------

def _make_planner_with_api_key(key: str) -> Planner:
    """Build a Planner with an explicit API key, bypassing settings."""
    return Planner.from_settings(api_key=key)


def _make_planner_without_model() -> Planner:
    """Build a Planner with an explicit model override, bypassing settings."""
    return Planner.from_settings(model="")


def _settings_has_key() -> bool:
    from app.config.settings import settings
    return bool(settings.openrouter_api_key)


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------

def test_planner_output_schema_is_valid() -> None:
    """Pydantic model accepts valid fields and serialises cleanly."""
    out = PlannerOutput(
        task_understanding="Build a web scraper",
        required_capabilities=["coding", "research"],
        steps=["Research libraries", "Write scraper", "Test scraper"],
        requires_research=True,
        requires_coding=True,
        requires_verification=True,
        estimated_complexity="medium",
    )
    data = out.model_dump()
    assert data["task_understanding"] == "Build a web scraper"
    assert "coding" in data["required_capabilities"]
    assert len(data["steps"]) == 3
    assert data["requires_research"] is True
    assert data["requires_coding"] is True
    assert data["requires_verification"] is True
    assert data["estimated_complexity"] == "medium"


def test_planner_construction_from_settings() -> None:
    """Planner can be constructed from environment/settings without error."""
    planner = Planner.from_settings()
    assert planner.model is not None
    assert planner.model.model_name == "nvidia/nemotron-3.5-lightning:free"
    assert planner.model.openai_api_base == "https://openrouter.ai/api/v1"
    assert planner.model.openai_api_key
    assert planner.structured_llm is not None


def test_planner_missing_api_key_raises() -> None:
    """Construction fails early when the API key is empty."""
    with pytest.raises(ValueError, match="OpenRouter API key"):
        _make_planner_with_api_key("")


def test_planner_missing_model_raises() -> None:
    """Construction fails early when the model is empty."""
    with pytest.raises(ValueError, match="OPENROUTER_MODEL"):
        _make_planner_without_model()


def test_planner_does_not_expose_credentials_in_repr() -> None:
    """Construction and repr must not leak the API key."""
    planner = Planner.from_settings()
    text = repr(planner) + str(planner.__dict__)
    assert _settings_has_key()
    assert "sk-or-" not in text and "sk-proj-" not in text


def test_planner_output_default_fields() -> None:
    """Output model has sensible defaults for optional boolean flags."""
    out = PlannerOutput(
        task_understanding="Hello",
        required_capabilities=[],
        steps=["do it"],
    )
    assert out.requires_research is False
    assert out.requires_coding is False
    assert out.requires_verification is False
    assert out.estimated_complexity == "medium"
    assert out.required_capabilities == []
    assert out.steps == ["do it"]


@pytest.mark.integration
def test_planner_real_call() -> None:
    """End-to-end: Planner plans a real task via OpenRouter."""
    planner = Planner.from_settings()
    result = planner.plan(
        "List three ways a multi-agent system can improve over a single LLM."
    )
    assert isinstance(result, PlannerOutput)
    assert result.task_understanding
    assert result.steps
    assert len(result.steps) >= 1
    assert result.estimated_complexity in {
        "trivial", "low", "medium", "high", "unknown"
    }


# ---------------------------------------------------------------------------
# Researcher tests
# ---------------------------------------------------------------------------

def _make_researcher_with_api_key(key: str) -> Researcher:
    return Researcher.from_settings(api_key=key)


def _make_researcher_without_model() -> Researcher:
    return Researcher.from_settings(model="")


def test_researcher_output_schema_is_valid() -> None:
    out = ResearcherOutput(
        research_question="What is LangGraph?",
        findings=["LangGraph is a graph-based execution framework.",
                  "It supports stateful, multi-actor workflows."],
        sources_or_evidence=["General knowledge about LangGraph."],
        uncertainties=["Exact version details may have changed."],
        requires_further_research=False,
    )
    data = out.model_dump()
    assert data["research_question"] == "What is LangGraph?"
    assert len(data["findings"]) == 2
    assert len(data["sources_or_evidence"]) >= 1
    assert isinstance(data["requires_further_research"], bool)


def test_researcher_construction_from_settings() -> None:
    researcher = Researcher.from_settings()
    assert researcher.model is not None
    assert researcher.model.model_name == "nvidia/nemotron-3.5-lightning:free"
    assert researcher.model.openai_api_base == "https://openrouter.ai/api/v1"
    assert researcher.model.openai_api_key
    assert researcher.structured_llm is not None


def test_researcher_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="OpenRouter API key"):
        _make_researcher_with_api_key("")


def test_researcher_missing_model_raises() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_MODEL"):
        _make_researcher_without_model()


def test_researcher_does_not_expose_credentials_in_repr() -> None:
    researcher = Researcher.from_settings()
    text = repr(researcher) + str(researcher.__dict__)
    assert "sk-or-" not in text and "sk-proj-" not in text


def test_researcher_default_fields() -> None:
    out = ResearcherOutput(research_question="Q")
    assert out.findings == []
    assert out.sources_or_evidence == []
    assert out.uncertainties == []
    assert out.requires_further_research is False


@pytest.mark.integration
def test_researcher_real_call() -> None:
    researcher = Researcher.from_settings()
    result = researcher.research(
        "Explain what LangGraph is and identify the main concepts "
        "that a beginner should understand."
    )
    assert isinstance(result, ResearcherOutput)
    assert result.research_question
    assert result.findings
    assert len(result.findings) >= 1
    assert isinstance(result.requires_further_research, bool)


# ---------------------------------------------------------------------------
# Coder tests
# ---------------------------------------------------------------------------

def _make_coder_with_api_key(key: str) -> Coder:
    return Coder.from_settings(api_key=key)


def _make_coder_without_model() -> Coder:
    return Coder.from_settings(model="")


def test_coder_output_schema_is_valid() -> None:
    out = CoderOutput(
        approach="Use two pointers from each end.",
        code="def is_palindrome(s: str) -> bool:\n    return s == s[::-1]",
        explanation="Compare characters from both ends moving inward.",
        assumptions=["Input is a plain string."],
        testing_notes=["Test empty string.", "Test single character."],
    )
    data = out.model_dump()
    assert data["approach"]
    assert data["code"]
    assert data["explanation"]
    assert isinstance(data["assumptions"], list)
    assert isinstance(data["testing_notes"], list)


def test_coder_construction_from_settings() -> None:
    coder = Coder.from_settings()
    assert coder.model is not None
    assert coder.model.model_name == "nvidia/nemotron-3.5-lightning:free"
    assert coder.model.openai_api_base == "https://openrouter.ai/api/v1"
    assert coder.model.openai_api_key
    assert coder.structured_llm is not None


def test_coder_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="OpenRouter API key"):
        _make_coder_with_api_key("")


def test_coder_missing_model_raises() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_MODEL"):
        _make_coder_without_model()


def test_coder_does_not_expose_credentials_in_repr() -> None:
    coder = Coder.from_settings()
    text = repr(coder) + str(coder.__dict__)
    assert "sk-or-" not in text and "sk-proj-" not in text


def test_coder_default_fields() -> None:
    out = CoderOutput(approach="X", code="x", explanation="y")
    assert out.assumptions == []
    assert out.testing_notes == []


@pytest.mark.integration
def test_coder_real_call() -> None:
    coder = Coder.from_settings()
    result = coder.code(
        "Write a Python function that checks whether a string is a palindrome."
    )
    assert isinstance(result, CoderOutput)
    assert result.approach
    assert result.code
    assert result.explanation
    # The code field must contain something that looks like code (not empty).
    assert result.code.strip()
    assert isinstance(result.assumptions, list)
    assert isinstance(result.testing_notes, list)


# ---------------------------------------------------------------------------
# Critic tests
# ---------------------------------------------------------------------------

def _make_critic_with_api_key(key: str) -> Critic:
    return Critic.from_settings(api_key=key)


def _make_critic_without_model() -> Critic:
    return Critic.from_settings(model="")


def test_critic_output_schema_is_valid() -> None:
    out = CriticOutput(
        overall_assessment="The statement is factually incorrect.",
        issues=["Python lists are mutable, not immutable."],
        missing_requirements=[],
        corrections=["Replace 'immutable' with 'mutable'."],
        verification_status="incorrect",
    )
    data = out.model_dump()
    assert data["overall_assessment"]
    assert data["verification_status"] == "incorrect"
    assert isinstance(data["issues"], list)
    assert isinstance(data["corrections"], list)


def test_critic_construction_from_settings() -> None:
    critic = Critic.from_settings()
    assert critic.model is not None
    assert critic.model.model_name == "nvidia/nemotron-3.5-lightning:free"
    assert critic.model.openai_api_base == "https://openrouter.ai/api/v1"
    assert critic.model.openai_api_key
    assert critic.structured_llm is not None


def test_critic_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="OpenRouter API key"):
        _make_critic_with_api_key("")


def test_critic_missing_model_raises() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_MODEL"):
        _make_critic_without_model()


def test_critic_does_not_expose_credentials_in_repr() -> None:
    critic = Critic.from_settings()
    text = repr(critic) + str(critic.__dict__)
    assert "sk-or-" not in text and "sk-proj-" not in text


def test_critic_validation_status_values() -> None:
    for status in ("correct", "partially_correct", "incorrect", "unclear"):
        out = CriticOutput(
            overall_assessment="x",
            issues=[],
            missing_requirements=[],
            corrections=[],
            verification_status=status,
        )
        assert out.verification_status == status


def test_critic_default_fields() -> None:
    out = CriticOutput(
        overall_assessment="OK",
        issues=[],
        missing_requirements=[],
        corrections=[],
        verification_status="correct",
    )
    assert out.issues == []
    assert out.missing_requirements == []
    assert out.corrections == []


@pytest.mark.integration
def test_critic_real_call() -> None:
    critic = Critic.from_settings()
    flawed = (
        "Python lists are immutable, so you cannot change an element "
        "after creating a list."
    )
    result = critic.review(
        original_task="Review the statement below for factual correctness.",
        output_to_review=flawed,
    )
    assert isinstance(result, CriticOutput)
    assert result.overall_assessment
    assert result.verification_status in {
        "correct", "partially_correct", "incorrect", "unclear"
    }
    # The critic should identify that the statement is wrong.
    combined = (
        result.overall_assessment
        + " "
        + " ".join(result.issues)
        + " "
        + " ".join(result.corrections)
    ).lower()
    assert "mutable" in combined or "incorrect" in combined


# ---------------------------------------------------------------------------
# Finalizer tests
# ---------------------------------------------------------------------------

def _make_finalizer_with_api_key(key: str) -> Finalizer:
    return Finalizer.from_settings(api_key=key)


def _make_finalizer_without_model() -> Finalizer:
    return Finalizer.from_settings(model="")


def test_finalizer_output_schema_is_valid() -> None:
    out = FinalizerOutput(
        final_answer="Python lists are useful because they are ordered and mutable.",
        key_points=["Ordered", "Mutable", "Can hold multiple values"],
        limitations=["Does not cover list performance characteristics."],
    )
    data = out.model_dump()
    assert data["final_answer"]
    assert len(data["key_points"]) == 3
    assert isinstance(data["limitations"], list)


def test_finalizer_construction_from_settings() -> None:
    finalizer = Finalizer.from_settings()
    assert finalizer.model is not None
    assert finalizer.model.model_name == "nvidia/nemotron-3.5-lightning:free"
    assert finalizer.model.openai_api_base == "https://openrouter.ai/api/v1"
    assert finalizer.model.openai_api_key
    assert finalizer.structured_llm is not None


def test_finalizer_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="OpenRouter API key"):
        _make_finalizer_with_api_key("")


def test_finalizer_missing_model_raises() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_MODEL"):
        _make_finalizer_without_model()


def test_finalizer_does_not_expose_credentials_in_repr() -> None:
    finalizer = Finalizer.from_settings()
    text = repr(finalizer) + str(finalizer.__dict__)
    assert "sk-or-" not in text and "sk-proj-" not in text


def test_finalizer_default_fields() -> None:
    out = FinalizerOutput(final_answer="Answer")
    assert out.key_points == []
    assert out.limitations == []


@pytest.mark.integration
def test_finalizer_real_call() -> None:
    finalizer = Finalizer.from_settings()
    result = finalizer.finalize(
        original_task="Explain why Python lists are useful.",
        supporting_info=(
            "Python lists are ordered, mutable collections that can "
            "contain multiple values."
        ),
    )
    assert isinstance(result, FinalizerOutput)
    assert result.final_answer
    assert isinstance(result.key_points, list)
    assert isinstance(result.limitations, list)
    # The final answer should mention lists and their usefulness.
    combined = (result.final_answer + " " + " ".join(result.key_points)).lower()
    assert "list" in combined
