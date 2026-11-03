import hashlib
from pathlib import Path
import pypdfium2 as pdfium

papers_dir = Path("Research/Research_Papers")
files = sorted(list(papers_dir.glob("*.pdf")))

output_path = Path("Research/papers_inventory.txt")

with open(output_path, "w", encoding="utf-8") as out:
    out.write(f"Total PDFs found: {len(files)}\n\n")
    
    seen_hashes = {}
    duplicates = []
    
    for f in files:
        h = hashlib.sha256()
        with open(f, "rb") as fp:
            while chunk := fp.read(65536):
                h.update(chunk)
        digest = h.hexdigest()
        
        # Extract title and first 300 words
        full_text = ""
        try:
            doc = pdfium.PdfDocument(f)
            if len(doc) > 0:
                full_text = doc[0].get_textpage().get_text_range()
        except Exception as e:
            full_text = f"Error: {e}"
            
        lines = [l.strip() for l in full_text.split("\n") if l.strip()]
        # first 4 non-empty lines usually have title
        header = " // ".join(lines[:6])
        
        if digest in seen_hashes:
            duplicates.append((f, seen_hashes[digest], "Exact Hash Match"))
        seen_hashes[digest] = f
        
        out.write(f"FILE: {f.name}\n")
        out.write(f"Size: {f.stat().st_size / (1024*1024):.2f} MB | SHA256: {digest[:16]}...\n")
        out.write(f"Header: {header[:150]}\n")
        out.write("-" * 80 + "\n")

    out.write("\nEXACT DUPLICATES:\n")
    for dup, orig, reason in duplicates:
        out.write(f"  REMOVE: {dup.name} (duplicate of {orig.name}) [{reason}]\n")

print(f"Inventory written to {output_path}")

