# Task-Adaptive Architecture Optimization for LLM Multi-Agent Systems
## Publication-Quality Paper Package (Overleaf Ready)

### Authors
- **Naveen Tanikonda** (naveen.23bce9787@vitapstudent.ac.in)
- **VeeraBhadra Jagarlamudi** (bhadra.23bce9780@vitapstudent.ac.in)
- **Sriram Somarouthu** (sriram.23bce20270@vitapstudent.ac.in)
- **Ravi Teja Perumalapalli** (teja.23bce9392@vitapstudent.ac.in)
*Vellore Institute of Technology – AP, Amaravathi, India*

---

### Directory Contents
- `main.tex`: Master IEEEtran conference paper (14 sections, two-column format).
- `references.bib`: Verified BibTeX bibliography (29 entries, 100% citation resolution).
- `figures/`: 9 publication figures in both PDF vector and 300 DPI PNG formats:
  - `figure_01_architecture_overview`: Decoupled research architecture diagram.
  - `figure_02_dynamic_topologies`: Dynamic before/after graph topologies.
  - `figure_03_active_agents_and_coverage`: Active agents & capability coverage.
  - `figure_04_resource_consumption`: Estimated tokens and cost reduction.
  - `figure_05_pareto_frontier`: Multi-objective Pareto frontier.
  - `figure_06_neural_learning_curves`: Cross-task neural adaptation trajectories.
  - `figure_07_neural_stability_seeds`: 25 individual seed observations & stability.
  - `figure_08_action_distribution`: Adaptation decision breakdown & mutation families.
  - `figure_09_runtime_overhead`: Dynamic LangGraph compilation & construction latency.
- `tables/`: 7 modular LaTeX table files (`table_01_*.tex` through `table_07_*.tex`).
- `supplementary/`: Canonical benchmark episode 2 CSV and source-of-truth manifest JSON.

---

### Overleaf Compilation Instructions
1. Download `Overleaf_Paper_Bundle.zip`.
2. On [Overleaf](https://www.overleaf.com), click **New Project** -> **Upload Project**.
3. Select `Overleaf_Paper_Bundle.zip`.
4. Ensure compiler is set to **pdfLaTeX** or **XeLaTeX** (default in Overleaf).
5. Click **Recompile**.
   - Zero compilation warnings, zero missing citations, zero undefined references.

---

### Local Reproducibility Instructions
To regenerate all experimental metrics, tables, and figures from scratch:
```bash
# 1. Recalculate metrics and build canonical manifest
python scripts/recalculate_all_metrics.py

# 2. Generate all 9 publication figures
python scripts/generate_publication_figures.py

# 3. Build all 7 LaTeX tables
python scripts/build_latex_tables.py

# 4. Assemble complete research paper
python scripts/build_complete_paper.py
```
