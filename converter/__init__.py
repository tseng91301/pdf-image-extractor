import os
import fitz  # PyMuPDF
from PIL import Image

from paddleocr import LayoutDetection

from .tools import random_uid
from .img_data import ImgData

class PdfInfo:
    pdf_path: str
    pdf_uid: str
    tmp_files_path = "tmp/PdfInfo/"
    pdf_doc: fitz.Document = None
    
    scanned_to_images = False
    pdf_img_paths = []
    
    pdf_layouts = []
    pdf_imgdatas = []
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pdf_uid = random_uid.generate()
        self.tmp_files_path = os.path.join(self.tmp_files_path, self.pdf_uid)
        os.makedirs(self.tmp_files_path, exist_ok=True)
        self.pdf_doc = fitz.open(self.pdf_path)
        
    def to_images(self, dpi: int = 100, get_output_path: bool = False):
        """
            將 PDF 每頁渲染成圖片後，再重組成「掃描型（影像型）」PDF。
        """
        temp_images_path = os.path.join(self.tmp_files_path, "extracted_images")
        os.makedirs(temp_images_path, exist_ok=True)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        image_paths = []
        for i in range(len(self.pdf_doc)):
            page = self.pdf_doc[i]
            pix = page.get_pixmap(matrix=mat, alpha=False)  # alpha=False 避免透明通道
            img_path = os.path.join(temp_images_path, f"page_{i+1:04d}.png")
            pix.save(img_path)
            image_paths.append(img_path)
        
        self.scanned_to_images = True
        self.pdf_img_paths = image_paths
        if get_output_path:
            print(f"PDF is now converted to images, stored in {temp_images_path}.")
            return temp_images_path
        
    def label_layout(self):
        if not self.scanned_to_images:
            self.to_images()
            pass
        self.pdf_layouts = []
        model = LayoutDetection(model_name="PP-DocLayout_plus-L")
        for img_path in self.pdf_img_paths:
            self.pdf_layouts.append(model.predict(img_path, batch_size=1, layout_nms=True)[0])
        return self.pdf_layouts
    
    def label_images(self):
        if not self.scanned_to_images:
            self.to_images()
            pass
        self.pdf_imgdatas = []
        for l in self.pdf_layouts:
            for i, v in enumerate(l['boxes']):
                if v["label"] == "image":
                    self.pdf_imgdatas.append(ImgData(
                        image=l['input_img'],
                        page_boxes=l['boxes'],
                        image_box_index=i
                    ))
        return self.pdf_imgdatas
    
    def __del__(self):
        self.pdf_doc.close()
        if os.path.exists(self.tmp_files_path):
            try:
                os.removedirs(self.tmp_files_path)
            except Exception as e:
                print(f"Please try manual delete {self.tmp_files_path}")