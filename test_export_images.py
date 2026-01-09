import fitz  # pip install pymupdf
import os

pdf_path = "example_pdfs/normal4.pdf"
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

def is_scanned_pdf(doc, area_threshold=0.6, text_threshold=10):
    scanned_pages = 0

    for page in doc:
        images = page.get_images(full=True)
        page_area = page.rect.width * page.rect.height
        big_images = 0

        for img in images:
            info = doc.extract_image(img[0])
            if (info["width"] * info["height"]) / page_area > area_threshold:
                big_images += 1

        text_len = len(page.get_text().strip())

        if big_images >= 1 and text_len < text_threshold:
            scanned_pages += 1

    return scanned_pages / len(doc) > 0.7


doc = fitz.open(pdf_path)
print(is_scanned_pdf(doc))

img_id = 0
for page_index in range(len(doc)):
    page = doc[page_index]
    image_list = page.get_images(full=False)

    for img in image_list:
        xref = img[0]
        print(f"Page rect: {page.get_image_rects(xref)}")
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]

        img_id += 1
        with open(f"{output_dir}/img_{img_id}.{image_ext}", "wb") as f:
            f.write(image_bytes)

print("完成圖片匯出")
