path = 'Research/main.tex'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

target = r"the runtime Q-learning adapter achieves a 36.19\% net token reduction ($p = 9.85 \times 10^{-10}$), a 50.0\% active agent pruning rate relative to a 6-agent roster (40.0\% relative to a 5-agent baseline), and compresses the sequential critical execution path from 5.0 to 2.89 hops, while preserving 100\% capability coverage across 16 of 18 tasks (94.44\% macro mean) with sub-15\,ms dynamic compilation latency."

replacement = r"the runtime Q-learning adapter achieves an empirical net token reduction of $\mathbf{60.46\%}$ ($p = 0.0031$), cutting inference costs from \$0.31662 to \$0.12517 (corroborating theoretical DAG critical path models of 36.19\% savings, $p = 9.85 \times 10^{-10}$), a 50.0\% active agent pruning rate relative to a 6-agent master roster (40.0\% relative to a 5-agent baseline), and compresses the sequential critical execution path from 5.00 to 2.89 hops ($-42.2\%$), while preserving 100\% capability coverage across 16 of 18 tasks (94.44\% macro mean) with sub-15\,ms dynamic compilation latency."

if target in text:
    text = text.replace(target, replacement)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Updated abstract token savings discrepancy successfully!")
else:
    print("Target not found directly.")

