with open('scripts/generate_architecture_diagram.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace all potential problematic mathtext
text = text.replace(r'C_{\\mathrm{cov}} \\ge C_{\\min}', r'C_{cov} \geq C_{min}')
text = text.replace(r'C_{\mathrm{cov}} \ge C_{\min}', r'C_{cov} \geq C_{min}')
text = text.replace(r'R_t = 0.4 C_{\\mathrm{cov}} + 0.3\\,\\mathrm{APR} + 0.3\\,L_{\\mathrm{crit}} - P_{\\mathrm{rej}}', r'R_t = 0.4 C_{cov} + 0.3 APR + 0.3 L_{crit} - P_{rej}')
text = text.replace(r'\\mathbf{x}_t = E_G(s_t) \\in \\mathbb{R}^{32}', r'x_t = E_G(s_t) in R^{32}')
text = text.replace(r'\\mathbf{z}_t = E_C(\\mathbf{c}_t) \\in \\mathbb{R}^{16}', r'z_t = E_C(c_t) in R^{16}')
text = text.replace(r'[\\mathbf{x}_t \\parallel \\mathbf{z}_t]', r'[x_t, z_t]')
text = text.replace(r'q_\\theta', r'q_\theta')
text = text.replace(r'y_t = R_t + \\gamma \\max_{a\'} Q(s\', a\')', r'y_t = R_t + \gamma \max Q(s\', a\')')
text = text.replace(r'\\Delta Q = 0.00 \\pm 0.00', r'\Delta Q = 0.00 \pm 0.00')
text = text.replace(r'\\mathbf{c}_t', r'c_t')
text = text.replace(r'\\kappa_t', r'\kappa_t')
text = text.replace(r'\\mathcal{C}_t', r'C_t')
text = text.replace(r'\\arg\\max_a', r'\mathrm{argmax}_a')
text = text.replace(r'|V_{\\mathrm{act}}| = 3.00 \\pm 0.59', r'|V_{act}| = 3.00 \pm 0.59')
text = text.replace(r'L_{\\mathrm{crit}} = 2.89', r'L_{crit} = 2.89')
text = text.replace(r'\\pm', r'\pm')

with open('scripts/generate_architecture_diagram.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated script with robust mathtext!")
