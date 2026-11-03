path = 'Research/main.tex'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

target = r"""\usepackage{array}
\usepackage{url}"""

replacement = r"""\usepackage{array}
\usepackage{tabularx}
\usepackage{makecell}
\usepackage{url}"""

if target in text:
    text = text.replace(target, replacement)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Added tabularx and makecell to main.tex")
else:
    print("Target not found directly, checking normalized...")
    text = text.replace('\\usepackage{array}\r\n\\usepackage{url}', '\\usepackage{array}\r\n\\usepackage{tabularx}\r\n\\usepackage{makecell}\r\n\\usepackage{url}')
    text = text.replace('\\usepackage{array}\n\\usepackage{url}', '\\usepackage{array}\n\\usepackage{tabularx}\n\\usepackage{makecell}\n\\usepackage{url}')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Added tabularx and makecell successfully!")

