import os
import json

from converter import PdfInfo

PDF_STORE_PATH = "./example_pdfs"
def get_pdf_files(directory):
    pdf_files = []
    # 遍歷目錄中的所有檔案
    for filename in os.listdir(directory):
        # 檢查檔案是否為 PDF
        if filename.endswith('.pdf'):
            pdf_files.append(filename)
    return pdf_files

pdf_files = get_pdf_files(PDF_STORE_PATH)
pdf_store_ids = []

for pdf_file in pdf_files:
    pdf = PdfInfo(os.path.join(PDF_STORE_PATH, pdf_file), gpu=True)
    pdf_store_ids.append(pdf.pdf_uid)
    pdf.to_images(dpi=300)
    pdf.label_layout()
    pdf.label_images(optimize_resolution=True, optimize_dpi=400, use_xref=False)
    pdf.extract_image_description(export=True)
    
with open("pdf_store_ids.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(pdf_store_ids, ensure_ascii=False, indent=4))
    f.close()