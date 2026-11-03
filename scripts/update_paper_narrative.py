"""Update main.tex files with live empirical numbers in Section VIII-B and Section X."""

for path_str in ["Research/main.tex", "Research_Paper/main.tex"]:
    with open(path_str, "r", encoding="utf-8") as f:
        text = f.read()

    # Section VIII-B replacement
    old_viii_b = r"In the unpruned static pipeline where all 5 specialist agents execute sequentially, cumulative token consumption reaches 208,639 tokens. Under AMAS topological adaptation, active agent pruning reduces total token consumption to 83,457 tokens, achieving an empirical net token savings of $\mathbf{60.00\%}$ and reducing live financial expenditure from \$0.31297 to \$0.12517."
    new_viii_b = r"In the unpruned static pipeline where the full multi-agent workflow executes across all tasks, cumulative live token consumption reaches 211,078 tokens. Under AMAS topological adaptation, active agent pruning reduces total live token consumption to 83,457 tokens, achieving an empirical net token savings of $\mathbf{60.46\%}$ and reducing live financial expenditure from \$0.31662 to \$0.12517."

    if old_viii_b in text:
        text = text.replace(old_viii_b, new_viii_b)
        print("Updated Section VIII-B in", path_str)

    # Section X replacement
    old_x = r"""\subsection{Hypothesis Testing on Resource Efficiency}
To test whether token reductions are statistically significant, we perform a paired two-tailed Student's t-test comparing baseline tokens against adapted tokens across the 18 paired benchmark tasks:
\begin{equation}
t = \frac{\bar{d}}{s_d / \sqrt{N}} = \frac{1266.67}{447.21 / \sqrt{18}} = 12.0167.
\end{equation}
With degrees of freedom $\mathrm{df} = 17$, this yields $p = \mathbf{9.8510 \times 10^{-10}}$, confirming statistical significance well beyond $\alpha = 0.001$. Paired t-testing on critical path compression ($5.0 \to 2.89$ hops) yields $t = 18.9482$ ($p = 1.87 \times 10^{-13}$)."""

    new_x = r"""\subsection{Hypothesis Testing on Resource Efficiency}
To test whether token reductions are statistically significant, we perform a paired two-tailed Student's t-test comparing empirical live static tokens against live dynamic adapted tokens across the 18 benchmark tasks on NVIDIA NIM:
\begin{equation}
t_{\mathrm{live}} = \frac{\bar{d}}{s_d / \sqrt{N}} = \frac{7090.06}{8721.43 / \sqrt{18}} = 3.4478.
\end{equation}
With degrees of freedom $\mathrm{df} = 17$, this yields $p = \mathbf{0.0031}$, confirming statistically significant token reduction ($p < 0.01$) under live production LLM execution. Under theoretical DAG critical path modeling ($T_{\mathrm{base}} = 3500$), paired t-testing yields $t = 12.0167$ ($p = 9.85 \times 10^{-10}$). Paired t-testing on critical path compression ($5.0 \to 2.89$ hops) yields $t = 18.9482$ ($p = 1.87 \times 10^{-13}$)."""

    if old_x in text:
        text = text.replace(old_x, new_x)
        print("Updated Section X in", path_str)

    with open(path_str, "w", encoding="utf-8") as f:
        f.write(text)

print("Synchronized narrative text in both main.tex files.")

