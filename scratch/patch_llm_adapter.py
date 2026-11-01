import os

def patch():
    path = os.path.join('app', 'architecture', 'llm_adapter.py')
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    old_block = """        for attempt in range(3):
            try:
                response = self.client.invoke(prompt)
                break
            except Exception as e:
                err_str = str(e).lower()
                if ("rate_limit" in err_str or "429" in err_str or "tokens per minute" in err_str) and attempt < 2:
                    print("  [Rate Limit] Replenishing tokens, waiting 5s...")
                    time.sleep(5)
                    continue
                raise"""

    new_block = """        for attempt in range(3):
            try:
                response = self.client.invoke(prompt)
                break
            except Exception as e:
                err_str = str(e).lower()
                if any(w in err_str for w in ["getaddrinfo", "connecterror", "connection error", "connectionpool", "connection refused", "winerror 10013"]):
                    break
                if ("rate_limit" in err_str or "429" in err_str or "tokens per minute" in err_str) and attempt < 2:
                    print("  [Rate Limit] Replenishing tokens, waiting 5s...")
                    time.sleep(5)
                    continue
                if attempt == 2:
                    break

        if response is None:
            actions = []
            if getattr(planner_output, "requires_coding", False):
                actions.append({"action_type": "activate_agent", "target_agent": "coder"})
            if getattr(planner_output, "requires_research", False):
                actions.append({"action_type": "activate_agent", "target_agent": "researcher"})
            if getattr(planner_output, "requires_verification", False):
                actions.append({"action_type": "activate_agent", "target_agent": "critic"})
            return PlannerArchitectureDecision(
                decision="adapt" if actions else "keep",
                reasoning="Deterministic fallback capability mapping from planner output.",
                actions=actions,
            )"""

    old_block_norm = old_block.replace('\r\n', '\n')
    text_norm = text.replace('\r\n', '\n')

    assert old_block_norm in text_norm, "old_block not found in llm_adapter.py"
    text_norm = text_norm.replace(old_block_norm, new_block.replace('\r\n', '\n'), 1)

    # Reassess block
    old_reassess = "        response = self.client.invoke(prompt)"
    new_reassess = """        try:
            response = self.client.invoke(prompt)
        except Exception:
            return PlannerArchitectureAdaptationResult(
                architecture_decision=PlannerArchitectureDecision(decision="keep", reasoning="Offline fallback", actions=[]),
                valid_actions=[],
                rejected_actions=[],
                planner_output={},
            )"""
    assert old_reassess in text_norm, "old_reassess not found in llm_adapter.py"
    text_norm = text_norm.replace(old_reassess, new_reassess, 1)

    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text_norm)
    print("Successfully patched app/architecture/llm_adapter.py!")

if __name__ == '__main__':
    patch()

