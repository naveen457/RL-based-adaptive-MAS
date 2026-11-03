import hashlib
import os
from pathlib import Path
import pypdfium2 as pdfium

papers_dir = Path("Research/Research_Papers")
files = list(papers_dir.glob("*.pdf"))
print(f"Total PDF files found: {len(files)}")

hashes = {}
duplicates = []
file_info = []

for f in files:
    h = hashlib.sha256()
    with open(f, "rb") as fp:
        while chunk := fp.read(8192):
            h.update(chunk)
    digest = h.hexdigest()
    
    # Try to extract title from first page
    title = ""
    try:
        doc = pdfium.PdfDocument(f)
        if len(doc) > 0:
            text = doc[0].get_textpage().get_text_range()
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            title = " | ".join(lines[:3])[:100]
    except Exception as e:
        title = f"Error reading PDF: {e}"
        
    size = f.stat().st_size
    file_info.append({"path": f, "name": f.name, "size": size, "hash": digest, "title": title})
    
    if digest in hashes:
        duplicates.append((f, hashes[digest]))
    else:
        hashes[digest] = f

print(f"\nExact Hash Duplicates Found: {len(duplicates)}")
for dup, orig in duplicates:
    print(f"  DUPLICATE: {dup.name}  ==>  ORIGINAL: {orig.name}")

# Also check for near-duplicates based on filename patterns (like " (1)")
print("\nFilename / Content Comparison:")
for info in sorted(file_info, key=lambda x: x["name"]):
    print(f"{info['name']:<50} | {info['size']/1024/1024:.2f} MB | {info['title'][:70]}")

