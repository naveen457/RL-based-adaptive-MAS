path = 'Research/main.tex'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Abstract
text = text.replace(
    'dynamic LangGraph compilation engine.',
    'dynamic state-graph compilation engine.'
)

# Keywords
text = text.replace(
    'dynamic workflows, LangGraph, graph pruning',
    'dynamic workflows, state-graph compilation, graph pruning'
)

# Figure 2 caption
text = text.replace(
    'drive dynamic LangGraph compilation (Stage~3)',
    'drive dynamic state-graph compilation (Stage~3)'
)

# Line 360
text = text.replace(
    r'Table~\ref{tab:per_task_resources} details the per-task resource metrics measured directly from live LLM execution against the NVIDIA NIM API (\texttt{nvidia/nemotron-3.5-super-vl-120b-a12b}) across all 18 benchmark tasks.',
    r'Table~\ref{tab:per_task_resources} details the per-task resource metrics measured directly from live LLM execution across all 18 benchmark tasks against a 120B-parameter production model.'
)

# Line 364
text = text.replace(
    'The empirical live NVIDIA NIM measurements corroborate the theoretical model',
    'The empirical production LLM measurements corroborate the theoretical model'
)

# Line 369
text = text.replace(
    r'\caption{Empirical live resource consumption across the 18 benchmark tasks in Episode 2 measured on the NVIDIA NIM API.',
    r'\caption{Empirical resource consumption across the 18 benchmark tasks in Episode 2 measured under production LLM execution.'
)

# Line 386 & 391 & 395
text = text.replace(
    r'graph construction ($T_{\mathrm{build}}$) and LangGraph compilation ($T_{\mathrm{comp}}$).',
    r'graph construction ($T_{\mathrm{build}}$) and state-graph compilation ($T_{\mathrm{comp}}$).'
)

text = text.replace(
    r'\caption{Dynamic LangGraph compilation and construction overhead across functional task domains.',
    r'\caption{Dynamic state-graph compilation and construction overhead across functional task domains.'
)

text = text.replace(
    'and LangGraph compilation time is $5.58 \\pm 1.77$\\,ms',
    'and state-graph compilation time is $5.58 \\pm 1.77$\\,ms'
)

# Discussion lines 471 & 473
text = text.replace(
    'sub-15\\,ms dynamic LangGraph compilation latency',
    'sub-15\\,ms dynamic state-graph compilation latency'
)

text = text.replace(
    r'Under live production execution against the NVIDIA NIM API (\texttt{nvidia/nemotron-3.5-super-vl-120b-a12b}), AMAS reduces total measured tokens',
    r'Under live execution against a 120B-parameter production foundation model, AMAS reduces total measured tokens'
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print('Brand cleanups applied successfully!')

