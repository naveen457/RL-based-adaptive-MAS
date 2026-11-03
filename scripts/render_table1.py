import pypdfium2 as pdfium
from PIL import Image

pdf = pdfium.PdfDocument("Research/main.pdf")

page = pdf[3] # 0-indexed: page 4
bitmap = page.render(scale=3)
image = bitmap.to_pil()
image.save("Research/page_04_rendered.png")

# Crop top half where Table I is located
table_crop = image.crop((0, int(image.height * 0.03), image.width, int(image.height * 0.40)))
table_crop.save("Research/table_01_rendered.png")
print("Successfully rendered Table I from page 4!")

