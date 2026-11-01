import os
import zipfile

def create_bundle():
    zip_path = os.path.join('Research_Paper', 'Overleaf_Paper_Bundle.zip')
    tex_path = os.path.join('Research_Paper', 'main.tex')
    bib_path = os.path.join('Research_Paper', 'thebibliography.bib')
    
    readme_content = """========================================================================
TASK-ADAPTIVE ARCHITECTURE OPTIMIZATION FOR MULTI-AGENT SYSTEMS
Paper Package for Overleaf Compilation
Authors: Naveen Tanikonda, VeeraBhadra Jagarlamudi, 
         Sriram Somarouthu, Ravi Teja Perumalapalli
Vellore Institute of Technology - Amaravathi (VIT-AP)
========================================================================

HOW TO VIEW & COMPILE IN OVERLEAF:
1. Go to https://www.overleaf.com and log into your account.
2. Click "New Project" -> "Upload Project".
3. Upload this zip file ("Overleaf_Paper_Bundle.zip").
4. Overleaf will automatically detect "main.tex" and compile the paper.
5. In the Overleaf editor:
   - Compiler: pdfLaTeX or XeLaTeX
   - Main document: main.tex
6. Click "Recompile" (Ctrl+Enter) to view the rendered 2-column IEEE PDF.

PACKAGE CONTENTS:
- main.tex: Complete IEEEtran conference paper (all sections, equations, TikZ figures)
- thebibliography.bib: Full BibTeX bibliography (24 references)
- README.txt: This instruction file
========================================================================
"""
    readme_path = os.path.join('Research_Paper', 'README_OVERLEAF.txt')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(tex_path, arcname='main.tex')
        zipf.write(bib_path, arcname='thebibliography.bib')
        zipf.write(readme_path, arcname='README.txt')

    print(f"Created {zip_path} successfully (Size: {os.path.getsize(zip_path)} bytes).")

if __name__ == '__main__':
    create_bundle()

