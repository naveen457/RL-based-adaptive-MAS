import pypdfium2 as pdfium
from pathlib import Path
import re

papers = sorted(list(Path("Research/Research_Papers").glob("*.pdf")))
print(f"Total papers in folder: {len(papers)}")

details = []
for p in papers:
    text = ""
    try:
        doc = pdfium.PdfDocument(p)
        text = doc[0].get_textpage().get_text_range()
    except Exception as e:
        text = f"Error: {e}"
    
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    details.append({"filename": p.name, "lines": lines[:8]})

for d in details:
    print("="*70)
    print(f"FILE: {d['filename']}")
    for l in d['lines'][:4]:
        safe_l = l.encode("ascii", errors="replace").decode("ascii")
        print(f"   {safe_l}")

