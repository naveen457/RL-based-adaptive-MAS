from app.config.settings import settings
from app.architecture.manager import ArchitectureManager
from app.graph.workflow import run_workflow


def print_architecture():
    """Display the current static MAS architecture."""
    mgr = ArchitectureManager.create_default_architecture()

    print("## Static MAS Architecture")
    print(f"Architecture ID: {mgr.get_architecture().architecture_id}")
    print(f"Agent count: {mgr.get_architecture().agent_count}")
    print(f"Active agents: {mgr.get_architecture().active_agent_count}")
    print()

    print("Agents:")
    for a in mgr.get_architecture().agents:
        status = "active" if a.active else "inactive"
        print(f"  * {a.agent_id}  (role: {a.role}, {status})")
        print(f"    capabilities: {', '.join(a.capabilities) or '(none)'}")
    print()

    print("Roles:")
    for a in mgr.get_architecture().agents:
        print(f"  {a.agent_id} -> {a.role}")
    print()

    print("Communication edges (agent -> agent):")
    for e in mgr.get_communication_edges():
        print(f"  {e.source} -> {e.target}")
    print()

    errors = mgr.validate()
    if errors:
        print("Validation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Architecture validation: PASSED")
    print()


def print_workflow_result(state):
    """Print the result of running the static MAS workflow."""
    plan = state.get("planner_output")
    if plan:
        print("PLANNER")
        print(f"  task_understanding: {plan.task_understanding}")
        print(f"  requires_research: {plan.requires_research}")
        print(f"  requires_coding: {plan.requires_coding}")
        print(f"  requires_verification: {plan.requires_verification}")
        print(f"  steps: {len(plan.steps)} step(s)")
        print()

    research = state.get("research_output")
    if research:
        print("RESEARCHER")
        for f in research.findings or []:
            print(f"  - {f}")
        print()

    coder = state.get("coder_output")
    if coder:
        print("CODER")
        print(f"  approach: {coder.approach}")
        print(f"  code:\r{code.code}")
        print(f"  explanation: {coder.explanation}")
        print()

    critic = state.get("critic_output")
    if critic:
        print("CRITIC")
        print(f"  overall_assessment: {critic.overall_assessment}")
        print(f"  verification_status: {critic.verification_status}")
        if critic.issues:
            print(f"  issues:")
            for i in critic.issues:
                print(f"    - {i}")
        if critic.corrections:
            print(f"  corrections:")
            for c in critic.corrections:
                print(f"    - {c}")
        print()

    final = state.get("final_answer")
    if final:
        print("FINALIZER")
        print(f"  final_answer: {final.final_answer}")
        if final.key_points:
            print(f"  key_points:")
            for k in final.key_points:
                print(f"    - {k}")
        if final.limitations:
            print(f"  limitations:")
            for l in final.limitations:
                print(f"    - {l}")
    print()


def main():
    print("Adaptive MAS - Static Baseline")
    print("=" * 48)

    if settings.openrouter_api_key:
        print("OpenRouter API key: Loaded")
    else:
        print("OpenRouter API key: NOT FOUND")

    print(f"OpenRouter model: {settings.openrouter_model or '(NOT SET)'}")

    if settings.langchain_api_key:
        print("LangSmith API key: Loaded")
    else:
        print("LangSmith API key: NOT FOUND")

    print(f"LangSmith project: {settings.langchain_project}")
    print()

    # Display architecture
    print_architecture()

    task = (
        "Write a Python function that checks whether a string is a "
        "palindrome and explain how it works."
    )

    print(f"TASK: {task}")
    print("-" * 48)
    print()

    state = run_workflow(task)

    plan = state.get("planner_output")
    print("PLANNER")
    print(f"  task_understanding: {plan.task_understanding}")
    print(f"  requires_research: {plan.requires_research}")
    print(f"  requires_coding: {plan.requires_coding}")
    print(f"  requires_verification: {plan.requires_verification}")
    print(f"  steps: {len(plan.steps)} step(s)")
    print()

    research = state.get("research_output")
    if research:
        print("RESEARCHER")
        for f in research.findings or []:
            print(f"  - {f}")
        print()

    coder = state.get("coder_output")
    if coder:
        print("CODER")
        print(f"  approach: {coder.approach}")
        print(f"  code:\r{coder.code}")
        print(f"  explanation: {coder.explanation}")
        print()

    critic = state.get("critic_output")
    if critic:
        print("CRITIC")
        print(f"  overall_assessment: {critic.overall_assessment}")
        print(f"  verification_status: {critic.verification_status}")
        if critic.issues:
            print(f"  issues:")
            for i in critic.issues:
                print(f"    - {i}")
        if critic.corrections:
            print(f"  corrections:")
            for c in critic.corrections:
                print(f"    - {c}")
        print()

    final = state.get("final_answer")
    if final:
        print("FINALIZER")
        print(f"  final_answer: {final.final_answer}")
        if final.key_points:
            print(f"  key_points:")
            for k in final.key_points:
                print(f"    - {k}")
        if final.limitations:
            print(f"  limitations:")
            for l in final.limitations:
                print(f"    - {l}")
    print()


if __name__ == "__main__":
    main()
