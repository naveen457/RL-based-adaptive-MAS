path = 'Research/main.tex'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace(
    '\\usepackage{makecell}',
    '\\providecommand{\\makecell}[2][c]{\\begin{tabular}[#1]{@{}#1@{}}#2\\end{tabular}}'
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Replaced \\usepackage{makecell} with robust native macro.")

