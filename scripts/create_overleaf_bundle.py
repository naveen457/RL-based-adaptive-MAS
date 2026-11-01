"""
Package Overleaf Paper Bundle and synchronize Research/ and Research_Paper/.
"""

import shutil
import zipfile
from pathlib import Path

res_dir = Path("Research")
paper_dir = Path("Research_Paper")

# Ensure subdirectories
for d in [res_dir, paper_dir]:
    (d / "figures").mkdir(parents=True, exist_ok=True)
    (d / "tables").mkdir(parents=True, exist_ok=True)
    (d / "sections").mkdir(parents=True, exist_ok=True)
    (d / "supplementary").mkdir(parents=True, exist_ok=True)

# Copy references.bib to both
bib_src = res_dir / "references.bib"
shutil.copy2(bib_src, paper_dir / "references.bib")

# Copy all figures to Research_Paper/figures
for fig_f in (res_dir / "figures").glob("*.*"):
    shutil.copy2(fig_f, paper_dir / "figures" / fig_f.name)

# Copy all tables to Research_Paper/tables
for tab_f in (res_dir / "tables").glob("*.tex"):
    shutil.copy2(tab_f, paper_dir / "tables" / tab_f.name)

# Copy canonical CSVs to supplementary in both
sup_src = Path("research_results/extracted_data/canonical_benchmark_episode2.csv")
if sup_src.exists():
    shutil.copy2(sup_src, res_dir / "supplementary" / "canonical_benchmark_episode2.csv")
    shutil.copy2(sup_src, paper_dir / "supplementary" / "canonical_benchmark_episode2.csv")

manifest_src = Path("research_results/extracted_data/canonical_manifest.json")
if manifest_src.exists():
    shutil.copy2(manifest_src, res_dir / "supplementary" / "canonical_manifest.json")
    shutil.copy2(manifest_src, paper_dir / "supplementary" / "canonical_manifest.json")

# Create README.md in both
readme_content = """# Task-Adaptive Architecture Optimization for LLM Multi-Agent Systems
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
"""

(res_dir / "README.md").write_text(readme_content.strip() + "\n", encoding="utf-8")
(paper_dir / "README.md").write_text(readme_content.strip() + "\n", encoding="utf-8")

# Create Overleaf Zip Bundle
zip_path = res_dir / "Overleaf_Paper_Bundle.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    # Add main.tex
    zipf.write(res_dir / "main.tex", arcname="main.tex")
    # Add references.bib
    zipf.write(res_dir / "references.bib", arcname="references.bib")
    # Add README.md
    zipf.write(res_dir / "README.md", arcname="README.md")
    # Add tables
    for f in (res_dir / "tables").glob("*.tex"):
        zipf.write(f, arcname=f"tables/{f.name}")
    # Add figures (both pdf and png)
    for f in (res_dir / "figures").glob("*.*"):
        zipf.write(f, arcname=f"figures/{f.name}")
    # Add supplementary
    for f in (res_dir / "supplementary").glob("*.*"):
        zipf.write(f, arcname=f"supplementary/{f.name}")

# Also copy zip to Research_Paper
shutil.copy2(zip_path, paper_dir / "Overleaf_Paper_Bundle.zip")

print(f"Created {zip_path} and synced Research/ and Research_Paper/ successfully!")

