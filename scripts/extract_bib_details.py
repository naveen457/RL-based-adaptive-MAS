import pypdfium2 as pdfium
from pathlib import Path

papers = [
    ("24048-AAAI26.YangL-MS.pdf", "BAMAS"),
    ("28430-AAAI26.YangW-MS.pdf", "LearningToDeliberate"),
    ("30196-AAAI26.ZhouZ-NLP.pdf", "ResMAS"),
    ("4584_Stronger_MAS_Multi_Agent_.pdf", "StrongerMAS"),
    ("NeurIPS-2025-rema-learning-to-meta-think-for-llms-with-multi-agent-reinforcement-learning-Paper-Conference.pdf", "ReMA"),
    ("Survey_on_Large_Language_Model-Enhanced_Reinforcement_Learning_Concept_Taxonomy_and_Methods.pdf", "TNNLSSurvey")
]

for filename, tag in papers:
    p = Path("Research/Research_Papers") / filename
    if p.exists():
        doc = pdfium.PdfDocument(p)
        txt = doc[0].get_textpage().get_text_range()
        lines = [l.strip() for l in txt.split("\n") if l.strip()]
        print(f"=== {tag} ({filename}) ===")
        for l in lines[:6]:
            print(" ", l.encode("ascii", errors="replace").decode("ascii"))

