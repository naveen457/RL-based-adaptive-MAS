from pathlib import Path

p = Path("scripts/build_paper_html.py")
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)

# Replace lines 270-274 (indices 269:274)
new_fig2 = (
    '<div class="full-width-figure">\n'
    '    <img src="figures/figure_02_dynamic_topologies.png" alt="Figure 2: Query-Centric Dynamic Topologies">\n'
    '    <div class="figure-caption"><strong>Fig. 2.</strong> Representative task execution walkthrough on benchmark query <code>res-001</code> (&ldquo;Research the architectural differences between Transformer and Mamba SSM models&rdquo;). (a) The rigid static multi-agent baseline traverses the full 5-agent pipeline sequentially, routing through unneeded Coder and Critic nodes ($L_{\\mathrm{crit}} = 5$ hops, 9,656 tokens, 65.8&thinsp;s latency). (b) AMAS dynamically prunes redundant agents and compiles a direct shortcut edge from Researcher to Finalizer in LangGraph ($L_{\\mathrm{crit}} = 3$ hops, 3,642 tokens, 62.3% savings, 9.1&thinsp;s latency, 7.2&times; speedup).</div>\n'
    '</div>\n\n'
)

# Verify lines[269] is indeed the figure 2 div
assert 'figures/figure_02_dynamic_topologies.png' in ''.join(lines[269:275]), "Line 269 mismatch!"
lines[269:274] = [new_fig2]

# Now find the token reduction paragraph
for i, line in enumerate(lines):
    if "Total baseline token consumption of 63,000 tokens is reduced" in line:
        lines[i] = (
            "<p>Under live production LLM execution against the NVIDIA NIM API (<code>nvidia/nemotron-3.5-super-vl-120b-a12b</code>) "
            "across all 18 benchmark tasks, baseline static multi-agent token consumption of 211,078 tokens is reduced to 83,457 tokens under AMAS, "
            "achieving an empirical net token savings of $\\mathbf{60.46\\%}$ ($p = 0.0031$, paired $t$-test) and reducing financial expenditure "
            "from \\$0.31662 to \\$0.12517 (Table IV). Under theoretical DAG critical path modeling, baseline tokens of 63,000 are reduced "
            "to 40,200 tokens ($36.19\\%$ savings, $p = 9.85 \\times 10^{-10}$).</p>\n"
        )
        print(f"Replaced token text at line {i+1}")
        break

p.write_text(''.join(lines), encoding="utf-8")
print("Saved scripts/build_paper_html.py successfully!")
