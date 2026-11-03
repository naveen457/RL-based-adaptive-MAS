import xml.etree.ElementTree as ET
from pathlib import Path

files = [
    'Untitled - Visual 1.svg',
    'Untitled - Visual 1 (1).svg',
    'Untitled - Visual 1 (2).svg'
]

for name in files:
    p = Path('Research/figures') / name
    tree = ET.parse(p)
    root = tree.getroot()
    print('='*80)
    print(f'FILE: {name}')
    print(f'Width: {root.attrib.get("width")}, Height: {root.attrib.get("height")}, ViewBox: {root.attrib.get("viewBox")}')
    
    # Collect styles or classes
    styles = []
    for s in root.iter('{http://www.w3.org/2000/svg}style'):
        if s.text:
            styles.append(s.text[:200] + '...')
    print(f'Styles count: {len(styles)}')
    
    # Text elements
    texts = []
    for elem in root.iter():
        tag = elem.tag.split('}')[-1]
        if elem.text and elem.text.strip():
            txt = elem.text.strip()
            if not txt.startswith('@') and not txt.startswith('.') and not txt.startswith('/*') and not txt.startswith('font-family'):
                texts.append((tag, elem.attrib, txt))
    
    print(f'Total text nodes: {len(texts)}')
    for tag, attr, txt in texts:
        safe_txt = txt.encode('ascii', errors='replace').decode('ascii')
        print(f'  <{tag} {attr}> : {safe_txt}')
