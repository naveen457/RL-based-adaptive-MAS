# Statistical Analysis: Benchmark Task & Resource Efficiency

## 1. Experimental Overview
- **Suite:** 18 tasks across 6 distinct functional categories (greeting, math, coding, research, web_search, cross_domain).
- **Episodes:** 2 sequential continuous training episodes (36 task executions total).
- **Evaluation Mode:** Simulated DAG-based critical path & agent invocation proxy (`is_proxy: True`).

## 2. Resource Reduction & Token Efficiency (Episode 2 Convergence)

- **Total Baseline Tokens:** 63,000
- **Total Adapted Tokens:** 40,200
- **Net Token Reduction:** **36.19%**
- **Baseline Tokens per Task:** 3500.0 $\pm$ 0.0
- **Adapted Tokens per Task:** 2233.3 $\pm$ 447.2 [95% CI: \pm 222.4]
- **Paired Student's t-test (Baseline vs Adapted Tokens):** t = 12.0167, p = 9.8510e-10 (Statistically significant at $\alpha = 0.001$).

## 3. Domain-Specific Breakdown

| Category | Tasks | Mean Base Tok | Mean Adapt Tok | Savings % | Mean Agents | APR % | Mean L_crit | Coverage % | Net Utility |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `coding` | 3 | 3500.0 | 2516.7 | +28.1% | 3.33 | 44.4% | 3.33 | 100.0% | 0.823 |
| `cross_domain` | 3 | 3500.0 | 2683.3 | +23.3% | 3.67 | 38.9% | 3.0 | 66.7% | 0.483 |
| `greeting` | 3 | 3500.0 | 1450.0 | +58.6% | 2.0 | 66.7% | 2.0 | 100.0% | 0.930 |
| `math` | 3 | 3500.0 | 2250.0 | +35.7% | 3.0 | 50.0% | 3.0 | 100.0% | 0.850 |
| `research` | 3 | 3500.0 | 2250.0 | +35.7% | 3.0 | 50.0% | 3.0 | 100.0% | 0.850 |
| `web_search` | 3 | 3500.0 | 2250.0 | +35.7% | 3.0 | 50.0% | 3.0 | 100.0% | 0.850 |

## 4. Multi-Episode Cumulative Totals (All 36 Executions)

- **Cumulative Baseline Tokens:** 126,000
- **Cumulative Adapted Tokens:** 80,400
- **Cumulative Token Savings:** **36.19%**
- **Episode 1 Savings:** 36.19%
- **Episode 2 Savings:** 36.19%
