path = 'Research/main.tex'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Line 235
text = text.replace(
    r'\caption{Representative task execution walkthrough on benchmark query \texttt{res-001}',
    r'\caption{Representative task execution walkthrough on benchmark query Res-01'
)

# Line 383
text = text.replace(
    r'The two sub-optimal points correspond to \texttt{cross-001} (33.3\% coverage, 2,250 tokens) and \texttt{cross-002} (66.7\% coverage, 3,250 tokens), where aggressive agent pruning removed specialist roles required for secondary sub-goals.',
    r'The two sub-optimal points correspond to Tasks Cross-01 (33.3\% coverage, 2,250 tokens) and Cross-02 (66.7\% coverage, 3,250 tokens), where aggressive agent pruning removed specialist roles required for secondary sub-goals.'
)

# Line 424
target_424 = "As visualized in Fig.~\\ref{fig:neural_learning_curves}(a), code inspection reveals the structural root cause: the discrete action space mapper constrained role modifications to $\\texttt{role\\_options} = [\\text{'analysis'}, \\text{'planning'}]$. Consequently, when a held-out task required \\texttt{research} or \\texttt{coding} capabilities, the policy could not generate the necessary mutation action, locking compatibility scores at a fixed ceiling. Table~\\ref{tab:action_space_ablation} details this structural action space ablation."

replacement_424 = "As visualized in Fig.~\\ref{fig:neural_learning_curves}(a), structural ablation reveals the root cause: the discrete action space mapper constrained role mutation actions to $\\mathcal{R}_{\\mathrm{valid}} = \\{\\text{Analysis}, \\text{Planning}\\}$. Consequently, when a held-out evaluation task required research or coding capabilities, the policy could not generate the necessary role assignment, locking compatibility scores at a fixed ceiling. Table~\\ref{tab:action_space_ablation} details this structural action space ablation."

text = text.replace(target_424, replacement_424)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated lines 235, 383, 424 successfully.')

