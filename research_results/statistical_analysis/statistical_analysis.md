# Master Statistical Methodology and Summary

This document details the statistical methodology, sample sizes, and inferential tests across the two research contributions.

## 1. Statistical Methodology
1. **Sample Independence:** Individual task steps within the continuous benchmark session are sequential and correlated through Q-table updates. They are treated as paired within-episode observations, NOT independent samples.
2. **Multi-Seed Neural Evaluation:** The neural adaptation experiment employs 5 distinct pseudo-random seeds (`[42, 101, 202, 303, 404]`) evaluating identical initial perturbed states. For comparisons between `shared_meta_policy` and `fresh_scratch_policy`, two-tailed paired Student's t-tests are conducted across seeds per task.
3. **Confidence Intervals:** 95% confidence intervals are computed using the Student's t-distribution: $\text{CI}_{95} = \bar{x} \pm t_{0.025, n-1} \frac{s}{\sqrt{n}}$.
4. **Zero-Variance Integrity:** When all seeds produce identical performance due to structural discrete action ceilings, the variance is explicitly reported as $0.0000$ and p-values are reported as $1.0000$ without artificial perturbation.

## 2. Core Statistical Findings Summary
- **Benchmark Token Reduction:** 63,000 baseline $\to$ 40,200 adapted tokens (36.19% reduction, paired t = 12.017, p = 9.8510e-10).
- **Cumulative 2-Episode Tokens:** 126,000 baseline $\to$ 80,400 adapted tokens (36.19% cumulative reduction).
- **Agent Pruning Rate:** Mean APR = 50.00% $\pm$ 9.92%.
- **Critical Path Compression:** Mean $L_{crit}$ = 2.89 hops vs 5.0 hops baseline.
- **Neural Adaptation Gain:** Bounded at $0.0000 \pm 0.0000$ across 4 of 5 held-out tasks due to `role_options=['analysis', 'planning']` action space constraints.
- **Neural Stability:** Shared meta-policy maintained $0.6000 \pm 0.0000$ on summarization, whereas fresh policy suffered a destructive mutation ($0.4800 \pm 0.2400$, 20% destructive rate).
