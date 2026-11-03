import xml.etree.ElementTree as ET
from pathlib import Path

files = [
    'Untitled - Visual 1.svg',
    'Untitled - Visual 1 (1).svg',
    'Untitled - Visual 1 (2).svg'
]

with open('Research/figures/svg_text_dump.txt', 'w', encoding='utf-8') as out:
    for name in files:
        p = Path('Research/figures') / name
        tree = ET.parse(p)
        root = tree.getroot()
        out.write('='*80 + '\n')
        out.write(f'FILE: {name}\n')
        out.write(f'Dimensions: {root.attrib.get("width")} x {root.attrib.get("height")} (viewBox: {root.attrib.get("viewBox")})\n')
        
        seen = set()
        for elem in root.iter():
            tag = elem.tag.split('}')[-1]
            if tag in ('text', 'tspan'):
                txt = ''.join(elem.itertext()).strip()
                if txt and not txt.startswith('AAAx') and not txt.startswith('@'):
                    if txt not in seen:
                        seen.add(txt)
                        out.write(f'  [{tag}] {txt}\n')
