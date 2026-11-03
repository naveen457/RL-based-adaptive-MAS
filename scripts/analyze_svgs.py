"""
Detailed analysis of the three SVGs in Research/figures/
"""

import xml.etree.ElementTree as ET
from pathlib import Path

files = [
    ("Untitled - Visual 1.svg", "Visual 1 (Overall Task-Adaptive Architecture)"),
    ("Untitled - Visual 1 (1).svg", "Visual 2 (Runtime Q-Learning Dynamic Architecture)"),
    ("Untitled - Visual 1 (2).svg", "Visual 3 (Experimental Task-Conditioned Neural Policy)")
]

for fname, desc in files:
    p = Path("Research/figures") / fname
    tree = ET.parse(p)
    root = tree.getroot()
    w = root.attrib.get("width", "unknown")
    h = root.attrib.get("height", "unknown")
    viewBox = root.attrib.get("viewBox", "unknown")
    
    # Extract unique texts
    texts = []
    for elem in root.iter():
        if elem.text and elem.text.strip():
            t = elem.text.strip()
            if not t.startswith("AAAx") and not t.startswith("@") and not t.startswith(".") and not t.startswith("/*"):
                if t not in texts:
                    texts.append(t)
    
    # Count shapes and paths
    paths = list(root.iter("{http://www.w3.org/2000/svg}path")) + list(root.iter("path"))
    rects = list(root.iter("{http://www.w3.org/2000/svg}rect")) + list(root.iter("rect"))
    circles = list(root.iter("{http://www.w3.org/2000/svg}circle")) + list(root.iter("circle"))
    images = list(root.iter("{http://www.w3.org/2000/svg}image")) + list(root.iter("image"))

    print("="*70)
    print(f"FILE: {fname}")
    print(f"Description: {desc}")
    print(f"Canvas Size: {w} x {h} | viewBox: {viewBox}")
    print(f"Vector elements: {len(paths)} paths, {len(rects)} rects, {len(circles)} circles, {len(images)} embedded images")
    print(f"Distinct Text Labels ({len(texts)}):")
    for t in texts:
        # safe print for unicode
        safe_t = t.encode("ascii", errors="replace").decode("ascii")
        print(f"  * {safe_t}")
