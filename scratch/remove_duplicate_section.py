import os
import re

def remove_duplicate_section():
    path = os.path.join('Research_Paper', 'main.tex')
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Verify markers
    assert 'as evidence about the other.' in lines[876]
    assert '\\caption{Proposed task-adaptive architecture' in lines[877]
    assert 'learning or selection.' in lines[1223]
    assert '% 5. MDP FORMULATION' in lines[1225]
    
    # Remove lines 877 to 1224 (inclusive)
    cleaned_lines = lines[:877] + lines[1224:]
    
    # Also clean the instructional comment at lines 350-356
    text = ''.join(cleaned_lines)
    comment_pattern = (
        r'% 4\. PROPOSED ARCHITECTURE  \(replaces the old "System Architecture"\)\n'
        r'% Replace everything from \\section\{System Architecture\} up to, but not\n'
        r'% including, the "5\. MDP FORMULATION" banner with this file\'s contents\.\n'
        r'% No extra packages or TikZ libraries are required\.\n'
        r'% Figures: fig:architecture \(overview\), fig:neural \(contribution 1\),\n'
        r'%\s+fig:runtime \(contribution 2\)\.\n'
    )
    text = re.sub(comment_pattern, '', text)
    
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
        
    print('Updated main.tex cleanly.')

if __name__ == '__main__':
    remove_duplicate_section()

