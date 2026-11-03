import pypdfium2 as pdfium
from PIL import Image

pdf = pdfium.PdfDocument("Research/main.pdf")
print(f"Total pages: {len(pdf)}")

page = pdf[2]
bitmap = page.render(scale=3)
image = bitmap.to_pil()
image.save("Research/page_03_rendered.png")
print(f"Saved page_03_rendered.png (Size: {image.size})")

table_crop = image.crop((0, int(image.height * 0.04), image.width, int(image.height * 0.42)))
table_crop.save("Research/table_01_rendered.png")
print("Saved table_01_rendered.png")
