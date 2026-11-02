"""
Generate a publication-grade two-column HTML preview of the paper.
Renders IEEEtran formatting, KaTeX math, embedded figures, tables, and references.
Allows instant browser viewing and printing to PDF (Ctrl+P -> Save as PDF).
"""

import re
from pathlib import Path

tex_file = Path("Research/main.tex")
bib_file = Path("Research/references.bib")
html_out = Path("Research/paper_preview.html")

tex_text = tex_file.read_text(encoding="utf-8")
bib_text = bib_file.read_text(encoding="utf-8")

# Parse BibTeX entries
bib_dict = {}
for match in re.finditer(r'@(\w+)\{([^,]+),\s*(.*?)\n\}', bib_text, re.DOTALL):
    b_type, b_key, b_body = match.groups()
    fields = {}
    for f_m in re.finditer(r'(\w+)\s*=\s*[\{"](.*?)[}"]', b_body, re.DOTALL):
        fields[f_m.group(1).lower()] = re.sub(r'\s+', ' ', f_m.group(2).strip())
    bib_dict[b_key.strip()] = fields

# Citation map
cited_keys = []
for m in re.finditer(r'\\cite\{([^}]+)\}', tex_text):
    for k in m.group(1).split(','):
        k = k.strip()
        if k and k not in cited_keys:
            cited_keys.append(k)

cite_to_num = {k: i + 1 for i, k in enumerate(cited_keys)}

# HTML structure
html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Task-Adaptive Architecture Optimization for LLM MAS - Paper Preview</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false},
        {left: '\\\\(', right: '\\\\)', display: false},
        {left: '\\\\[', right: '\\\\]', display: true}
    ]});"></script>
<style>
    @page { size: letter; margin: 0.75in; }
    body {
        font-family: 'Times New Roman', Times, serif;
        font-size: 10pt;
        line-height: 1.25;
        color: #111;
        background-color: #525659;
        margin: 0;
        padding: 20px;
    }
    .paper-sheet {
        background: #fff;
        max-width: 8.5in;
        min-height: 11in;
        margin: 0 auto;
        padding: 0.75in;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        box-sizing: border-box;
    }
    .header-bar {
        position: fixed;
        top: 10px;
        right: 20px;
        background: #2b5c8f;
        color: white;
        padding: 8px 16px;
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        z-index: 1000;
        font-family: sans-serif;
        font-size: 12px;
    }
    .header-bar button {
        background: white;
        color: #2b5c8f;
        border: none;
        padding: 4px 10px;
        border-radius: 3px;
        cursor: pointer;
        font-weight: bold;
        margin-left: 8px;
    }
    .title-banner { text-align: center; margin-bottom: 20px; }
    h1.paper-title { font-size: 18pt; font-weight: bold; margin-bottom: 12px; }
    .author-grid {
        display: flex;
        justify-content: space-around;
        margin-bottom: 20px;
        font-size: 9.5pt;
    }
    .author-block { text-align: center; max-width: 22%; }
    .author-name { font-weight: bold; }
    .author-inst { font-style: italic; font-size: 8.5pt; }
    .author-email { font-family: monospace; font-size: 8pt; color: #333; }
    .abstract-box {
        margin: 0 20px 24px 20px;
        padding: 10px 14px;
        background: #f9f9f9;
        border-left: 3px solid #2b5c8f;
        font-size: 9pt;
        text-align: justify;
    }
    .abstract-title { font-weight: bold; font-style: italic; }
    .keywords { margin-top: 6px; font-weight: bold; font-size: 8.5pt; }
    .keywords span { font-weight: normal; font-style: italic; }
    .two-column {
        column-count: 2;
        column-gap: 0.35in;
        column-rule: 0.5px solid #e0e0e0;
        text-align: justify;
    }
    h2.sec-heading {
        font-size: 10pt;
        text-transform: uppercase;
        text-align: center;
        margin-top: 14px;
        margin-bottom: 6px;
        border-bottom: 0.5px solid #222;
        padding-bottom: 2px;
        break-after: avoid;
    }
    h3.subsec-heading {
        font-size: 10pt;
        font-style: italic;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 4px;
        break-after: avoid;
    }
    p { margin-top: 0; margin-bottom: 6px; text-indent: 1.2em; }
    p.no-indent { text-indent: 0; }
    .figure-box {
        margin: 12px 0;
        text-align: center;
        break-inside: avoid;
    }
    .figure-box img {
        max-width: 100%;
        height: auto;
        border: 0.5px solid #ddd;
        border-radius: 2px;
    }
    .figure-caption {
        font-size: 8pt;
        text-align: justify;
        margin-top: 4px;
        color: #333;
    }
    .figure-caption strong { font-weight: bold; }
    .full-width-figure {
        column-span: all;
        margin: 16px 0;
        text-align: center;
    }
    .full-width-figure img {
        max-width: 98%;
        height: auto;
    }
    table.latex-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 8pt;
        margin: 10px 0;
        break-inside: avoid;
    }
    table.latex-table th, table.latex-table td {
        border-top: 0.5px solid #222;
        border-bottom: 0.5px solid #222;
        padding: 4px 6px;
        text-align: left;
    }
    table.latex-table th { font-weight: bold; border-bottom: 1.5px solid #222; }
    .table-caption {
        font-size: 8pt;
        font-weight: bold;
        text-align: center;
        margin-bottom: 4px;
    }
    .reference-list {
        font-size: 8pt;
        padding-left: 18px;
        margin-top: 6px;
    }
    .reference-list li { margin-bottom: 4px; }
    @media print {
        body { background: transparent; padding: 0; }
        .paper-sheet { box-shadow: none; margin: 0; width: 100%; max-width: none; }
        .header-bar { display: none; }
    }
</style>
</head>
<body>

<div class="header-bar">
    <span>IEEE Conference Paper Preview (AMAS)</span>
    <button onclick="window.print()">Print to PDF (Ctrl+P)</button>
</div>

<div class="paper-sheet">
    <div class="title-banner">
        <h1 class="paper-title">Task-Adaptive Architecture Optimization for LLM-Based Multi-Agent Systems via Reinforcement Learning and Dynamic Graph Construction</h1>
        <div class="author-grid">
            <div class="author-block">
                <div class="author-name">Naveen Tanikonda</div>
                <div class="author-inst">Vellore Institute of Technology – AP</div>
                <div class="author-email">naveen.23bce9787@vitapstudent.ac.in</div>
            </div>
            <div class="author-block">
                <div class="author-name">VeeraBhadra Jagarlamudi</div>
                <div class="author-inst">Vellore Institute of Technology – AP</div>
                <div class="author-email">bhadra.23bce9780@vitapstudent.ac.in</div>
            </div>
            <div class="author-block">
                <div class="author-name">Sriram Somarouthu</div>
                <div class="author-inst">Vellore Institute of Technology – AP</div>
                <div class="author-email">sriram.23bce20270@vitapstudent.ac.in</div>
            </div>
            <div class="author-block">
                <div class="author-name">Ravi Teja Perumalapalli</div>
                <div class="author-inst">Vellore Institute of Technology – AP</div>
                <div class="author-email">teja.23bce9392@vitapstudent.ac.in</div>
            </div>
        </div>
    </div>

    <div class="abstract-box">
        <span class="abstract-title">Abstract—</span>
        Large Language Model (LLM)-based multi-agent systems coordinate ensembles of specialized agents to solve complex reasoning, coding, and information retrieval tasks. However, conventional multi-agent frameworks typically deploy rigid, monolithic static topologies, enforcing identical agent configurations across heterogeneous tasks. This static paradigm incurs substantial computational redundancy, unnecessary communication overhead, prolonged sequential latency, and inflated inference costs for simple queries. In this paper, we investigate a task-adaptive architecture optimization framework for LLM-based multi-agent systems, where agent activations, operational roles, and directed communication topologies are dynamically tailored to input task requirements. We formulate the multi-agent topology selection problem as a Markov Decision Process (MDP) and present two clearly demarcated contributions. First, as our primary live runtime contribution (C2), we implement an active tabular Q-learning selector coupled with an invariant-checked architecture manager and a dynamic LangGraph compilation engine. Across an 18-task benchmark suite spanning six functional domains, the runtime Q-learning adapter achieves a 36.19% net token reduction ($p = 9.85 \\times 10^{-10}$), a 50.0% active agent pruning rate relative to a 6-agent roster (40.0% relative to a 5-agent baseline), and compresses the sequential critical execution path from 5.0 to 2.89 hops, while preserving 100% capability coverage across 16 of 18 tasks (94.44% macro mean) with sub-15 ms dynamic compilation latency. Second, as an experimental exploration (C1), we investigate task-conditioned cross-task neural adaptation using a shared policy trained across task environments and evaluated on held-out tasks. We rigorously document empirical boundaries: under a discrete action space restricted to planning and analysis, neural adaptation plateaus at zero gain on four tasks due to missing domain-specific roles, while exhibiting protective stability on summarization. Our findings provide reproducible empirical evidence and architectural principles for deploying parsimonious, dynamically compiled multi-agent workflows.
        <div class="keywords">Index Terms—<span>Multi-agent systems, large language models, reinforcement learning, adaptive architecture, dynamic workflows, LangGraph, graph pruning, critical path compression.</span></div>
    </div>

    <div class="two-column">
"""

# Helper to format citations
def replace_cites(text):
    def sub_cite(m):
        keys = [k.strip() for k in m.group(1).split(',')]
        nums = [str(cite_to_num.get(k, '?')) for k in keys]
        return f"[{', '.join(nums)}]"
    return re.sub(r'\\cite\{([^}]+)\}', sub_cite, text)

# Build body content
body_html = """
<h2 class="sec-heading">I. Introduction</h2>
<p>Large Language Models (LLMs) have demonstrated remarkable efficacy across natural language comprehension, mathematical reasoning, program synthesis, and open-domain problem solving. To overcome the reasoning and context limitations of single-agent prompting, multi-agent systems (MAS) have emerged as the dominant architecture for handling sophisticated decomposition tasks [1], [2], [3]. In these collaborative architectures, heterogeneous agents assume specialized functional roles—such as planning, information research, code synthesis, critique, and result finalization—communicating through structured control flows.</p>
<p>Despite their empirical success, conventional LLM multi-agent systems suffer from a fundamental architectural limitation: <em>structural rigidity</em>. Prevailing frameworks deploy static, predefined execution topologies that invoke the entire multi-agent team regardless of the intrinsic complexity or capability demands of the user request. Under this fixed paradigm, a trivial greeting or direct arithmetic calculation triggers the same heavyweight multi-agent pipeline as an intricate distributed systems design task, executing redundant planning, research, coding, and verification loops.</p>

<div class="full-width-figure">
    <img src="figures/figure_01_architecture_overview.png" alt="Figure 1: Architecture Overview">
    <div class="figure-caption"><strong>Fig. 1.</strong> System architecture schematic showing the four stages of the proposed adaptive multi-agent framework. Stage 1 extracts task context $c_t$. In Stage 2, the solid green runtime pathway (Contribution C2) utilizes tabular Q-learning and structural validation to drive dynamic LangGraph compilation (Stage 3) and agent execution (Stage 4). The dashed orange experimental pathway (Contribution C1) trains a shared task-conditioned neural policy on proxy evaluations offline and is strictly decoupled from runtime execution.</div>
</div>

<h2 class="sec-heading">II. Related Work</h2>
<p>Multi-agent collaboration has proven highly effective for complex problem decomposition. Foundational systems such as MetaGPT [1] demonstrated that encoding human software engineering Standard Operating Procedures (SOPs) into agent roles significantly reduces communication ambiguity. However, static communication channels incur severe latency and token overhead [2], [3], [4].</p>
<p>To eliminate agent redundancy, AgentDropout [5] explored dynamic agent deactivation during multi-turn interactions. Similarly, DyLAN [6] introduced peer-rating mechanisms to prune inactive agents between layers. Adaptive Graph Pruning (AGP) [7] and Multi-Agent Architecture Search (MaAS) [8] optimized communication topologies through evolutionary and supernet search. Our runtime contribution complements these works by combining instant tabular Q-learning selection with native LangGraph compilation, executing sub-15 ms dynamic graph generation with zero token search overhead.</p>

<div class="full-width-figure">
    <img src="figures/figure_02_dynamic_topologies.png" alt="Figure 2: Query-Centric Dynamic Topologies">
    <div class="figure-caption"><strong>Fig. 2.</strong> Representative task execution walkthrough on benchmark query <code>res-001</code> (&ldquo;Research the architectural differences between Transformer and Mamba SSM models&rdquo;). (a) The rigid static multi-agent baseline traverses the full 5-agent pipeline sequentially, routing through unneeded Coder and Critic nodes ($L_{\mathrm{crit}} = 5$ hops, 9,656 tokens, 65.8&thinsp;s latency). (b) AMAS dynamically prunes redundant agents and compiles a direct shortcut edge from Researcher to Finalizer in LangGraph ($L_{\mathrm{crit}} = 3$ hops, 3,642 tokens, 62.3% savings, 9.1&thinsp;s latency, 7.2&times; speedup).</div>
</div>

<h2 class="sec-heading">III. Problem Formulation</h2>
<p>Let the multi-agent system be represented as a directed graph $G = (V, E)$, where $V$ denotes the pool of $n = 6$ potential agent nodes (Planner, Tool, Researcher, Coder, Critic, Finalizer). An active configuration at step $t$ is $G_t = \\langle \\mathbf{v}_t, \\mathbf{r}_t, \\mathbf{A}_t \\rangle$, where $\\mathbf{v}_t \\in \\{0, 1\\}^n$ indicates active nodes, $\\mathbf{r}_t$ denotes role assignments, and $\\mathbf{A}_t$ is the directed adjacency matrix.</p>
<p>An incoming query is characterized by task context $c_t = \\langle \\kappa_t, d_t, \\mathcal{C}_t, \\mathbf{f}_t \\rangle$. The objective is to select an architecture $G_t^*$ that maximizes Pareto net utility $U_{\\mathrm{pareto}} = 0.5(1 - T_{\\mathrm{adapt}}/T_{\\mathrm{base}}) + 0.5(C_{\\mathrm{cov}}/100)$ subject to structural validity invariants $\\mathcal{V}(G) = \\text{True}$.</p>
<p>We explicitly distinguish three Agent Pruning Rate (APR) definitions: (1) $\\mathrm{APR}_{\\mathrm{pool6}} = (6 - \\|\\mathbf{v}_t\\|_1)/6 \\times 100\\%$; (2) $\\mathrm{APR}_{\\mathrm{base5}} = (5 - \\|\\mathbf{v}_t\\|_1)/5 \\times 100\\%$; and (3) $\\mathrm{APR}_{\\mathrm{invoked}} = (6 - |V_{\\mathrm{invoked}}|)/6 \\times 100\\%$.</p>

<h2 class="sec-heading">IV. Experimental Results</h2>
<p>Across the 18 benchmark tasks in canonical Episode 2, AMAS prunes redundant agents effectively, compressing the active agent count from 5.0 to a macro mean of $\\mathbf{3.00 \\pm 0.59}$ agents. This achieves an active Agent Pruning Rate of $\\mathbf{50.00\\% \\pm 9.90\\%}$ relative to the 6-agent roster, and $\\mathbf{40.00\\% \\pm 11.88\\%}$ relative to the 5-agent baseline.</p>
<p>Critical path length compresses from 5.0 to $\\mathbf{2.89 \\pm 0.47}$ hops ($42.2\\%$ sequential latency reduction, paired $t = 18.9482, p = 1.87 \\times 10^{-13}$). Capability coverage is $100\\%$ preserved for 16 of 18 tasks ($94.44\\%$ macro mean).</p>

<div class="full-width-figure">
    <img src="figures/figure_03_active_agents_and_coverage.png" alt="Figure 3: Active Agents and Coverage">
    <div class="figure-caption"><strong>Fig. 3.</strong> Active agent count and capability coverage across the 18 benchmark tasks in Episode 2. (a) Active agents pruned to 2–4 across domains. (b) Capability coverage is 100% preserved on 16 tasks, with 2 cross-domain tasks experiencing under-engineering.</div>
</div>

<p>Under live production LLM execution against the NVIDIA NIM API (<code>nvidia/nemotron-3.5-super-vl-120b-a12b</code>) across all 18 benchmark tasks, baseline static multi-agent token consumption of 211,078 tokens is reduced to 83,457 tokens under AMAS, achieving an empirical net token savings of $\mathbf{60.46\%}$ ($p = 0.0031$, paired $t$-test) and reducing financial expenditure from \$0.31662 to \$0.12517 (Table IV). Under theoretical DAG critical path modeling, baseline tokens of 63,000 are reduced to 40,200 tokens ($36.19\%$ savings, $p = 9.85 \times 10^{-10}$).</p>

<div class="full-width-figure">
    <img src="figures/figure_04_resource_consumption.png" alt="Figure 4: Resource Consumption">
    <div class="figure-caption"><strong>Fig. 4.</strong> Resource consumption across 18 benchmark tasks in Episode 2. (a) Token consumption reduced by 36.19% overall. (b) Estimated cost reduced from \\$0.0945 to \\$0.0602.</div>
</div>

<div class="figure-box">
    <img src="figures/figure_05_pareto_frontier.png" alt="Figure 5: Pareto Frontier">
    <div class="figure-caption"><strong>Fig. 5.</strong> Quality-efficiency Pareto frontier showing 16 Pareto-optimal topologies delivering 100% capability coverage with 13% to 59% fewer tokens.</div>
</div>

<div class="figure-box">
    <img src="figures/figure_09_runtime_overhead.png" alt="Figure 9: Runtime Overhead">
    <div class="figure-caption"><strong>Fig. 9.</strong> Dynamic LangGraph compilation ($5.58 \\pm 1.77$\\,ms) and construction ($6.19 \\pm 1.88$\\,ms) latency across domains, ensuring sub-15\\,ms total adaptation overhead.</div>
</div>

<div class="full-width-figure">
    <img src="figures/figure_08_action_distribution.png" alt="Figure 8: Action Distribution">
    <div class="figure-caption"><strong>Fig. 8.</strong> Reinforcement learning adaptation behavior. (a) Active RL policy hit rate reaches 72.2% (13/18 tasks) in Episode 2 with 0 tokens search overhead. (b) Distribution of accepted graph mutations.</div>
</div>

<h2 class="sec-heading">V. Neural Cross-Task Fast Adaptation</h2>
<p>Evaluation of Contribution C1 across 5 held-out tasks and 5 seeds revealed that adaptation gain was bounded at $\\Delta Q_{\\mathrm{adapt}} = 0.0000$ on 4 tasks due to structural action-space constraints (<code>role_options=['analysis', 'planning']</code>). On <code>task-coding-001</code>, exploratory mutation degraded scores from 1.00 to 0.60. However, on <code>task-summarization-001</code>, the pre-trained shared meta-policy preserved 100% stability across all seeds ($0.6000 \\pm 0.0000$), whereas the fresh scratch policy suffered destructive mutations in 20% of seeds ($0.4800 \\pm 0.2683$).</p>

<div class="full-width-figure">
    <img src="figures/figure_06_neural_learning_curves.png" alt="Figure 6: Neural Adaptation Curves">
    <div class="figure-caption"><strong>Fig. 6.</strong> Cross-task neural policy adaptation trajectories across 2 fine-tuning steps. (a) Research tasks plateau due to action-space role constraints. (b) Coding tasks suffer exploration degradation.</div>
</div>

<div class="figure-box">
    <img src="figures/figure_07_neural_stability_seeds.pdf" alt="Figure 7: Neural Stability">
    <div class="figure-caption"><strong>Fig. 7.</strong> Neural adaptation stability across 25 seed observations. Diamond markers indicate policy means. Pretrained policy acts as a protective regularizer against architectural drift.</div>
</div>

<h2 class="sec-heading">VI. Conclusion</h2>
<p>This paper investigated task-adaptive architecture optimization for LLM-based multi-agent systems via reinforcement learning and dynamic LangGraph compilation. The runtime Q-learning adapter reduced token consumption by 36.19% ($p = 9.85 \\times 10^{-10}$), achieved a 50.0% active agent pruning rate, compressed critical path latency by 42.2%, and preserved 100% capability coverage on 16 of 18 tasks with sub-15\\,ms compilation overhead. Furthermore, our neural adaptation investigation established that discrete action space restrictions create severe plateaus on held-out tasks while providing protective stability against destructive mutations.</p>

<h2 class="sec-heading">References</h2>
<ol class="reference-list">
"""

for k in cited_keys:
    num = cite_to_num[k]
    f = bib_dict.get(k, {})
    author = f.get('author', 'Unknown')
    title = f.get('title', 'Unknown Title')
    venue = f.get('booktitle') or f.get('journal') or 'arXiv preprint'
    year = f.get('year', '2026')
    body_html += f"<li>[{num}] {author}, &ldquo;{title},&rdquo; <em>{venue}</em>, {year}.</li>\n"

body_html += """
</ol>
</div>
</div>
</body>
</html>
"""

full_html = html + body_html
html_out.write_text(full_html, encoding="utf-8")
print(f"Generated {html_out} successfully ({len(full_html.splitlines())} lines)")
