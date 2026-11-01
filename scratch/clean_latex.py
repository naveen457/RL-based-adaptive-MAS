import os

def clean():
    path = os.path.join('Research_Paper', 'main.tex')
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    clean_lines = []
    removed = 0
    for l in lines:
        s = l.strip()
        if s.startswith('```'):
            removed += 1
            continue
        clean_lines.append(l)
        
    print(f'Original: {len(lines)}, Cleaned: {len(clean_lines)}, Removed: {removed}')
    assert removed == 7, f'Expected 7 removals, got {removed}'
    
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(clean_lines)
    print('Cleaned main.tex successfully!')

if __name__ == '__main__':
    clean()

