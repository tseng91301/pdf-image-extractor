import os
import fitz  # PyMuPDF
from PIL import Image
import cv2
import json

from paddleocr import LayoutDetection

from .tools import random_uid
from .img_data import ImgData

class PdfInfo:
    pdf_path: str
    pdf_uid: str
    tmp_files_path = "tmp/PdfInfo/"
    pdf_doc: fitz.Document = None
    use_gpu = False
    
    scanned_to_images = False
    pdf_img_paths = []
    
    pdf_layouts = []
    pdf_imgdatas = []
    def __init__(self, pdf_path, gpu=False):
        self.pdf_path = pdf_path
        self.pdf_uid = random_uid.generate()
        self.tmp_files_path = os.path.join(self.tmp_files_path, self.pdf_uid)
        os.makedirs(self.tmp_files_path, exist_ok=True)
        self.pdf_doc = fitz.open(self.pdf_path)
        self.use_gpu = gpu
        
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
        
    def label_layout(self, output=False):
        """
        將 pdf 的各種 Layout 標記出來，會使用 PP-DocLayout_plus-L 模型來進行偵測
        """
        if not self.scanned_to_images:
            self.to_images()
            pass
        self.pdf_layouts = []
        model = LayoutDetection(model_name="PP-DocLayout_plus-L")
        for i, img_path in enumerate(self.pdf_img_paths):
            p = model.predict(img_path, batch_size=1, layout_nms=True)
            self.pdf_layouts.append(p[0])
            
        if output:
            for i, v in enumerate(self.pdf_layouts):
                v.save_to_img(f"output/{self.pdf_uid}/layout_detection/{i+1}.png")
        return self.pdf_layouts
    
    def label_images(self):
        """
        將已經進行 Layout Labeling 的資料下去尋找圖片，並將所有找到的圖片放到 ImgData 物件裡面進行分析
        """
        if not self.scanned_to_images:
            self.to_images()
            pass
        self.pdf_imgdatas = []
        for i0, l in enumerate(self.pdf_layouts):
            for i, v in enumerate(l['boxes']):
                if v["label"] == "image":
                    i_d = ImgData(
                        image=l['input_img'],
                        page_boxes=l['boxes'],
                        image_box_index=i,
                        gpu=self.use_gpu
                    )
                    i_d.img_page = i0
                    i_d.raw_pdf_path = self.pdf_path
                    self.pdf_imgdatas.append(i_d)
        return self.pdf_imgdatas
    
    def extract_image_description(self, export=False):
        """
        將所有圖片周圍的 title, text 的文字資料提取出來
        方法: 先嘗試使用一般的 pdf 文字提取，若提取不到或是提取出來為亂碼會自動使用 OCR 來取得文字
        """
        n = len(self.pdf_imgdatas)
        for i, img in enumerate(self.pdf_imgdatas):
            print(f"Getting surrounding texts and figure_title of image {i+1}/{n}, reading page {img.img_page+1}")
            img.get_surroundings(cv2.imread(self.pdf_img_paths[img.img_page]))
        if export:
            self.export_all_image_datas()
            
    def export_all_image_datas(self, path: str=None):
        if path is None:
            path = f"output/{self.pdf_uid}/image_datas/"
        os.makedirs(path, exist_ok=True)
        print(f"Exporting data into {path}...")
        for i, img_data in enumerate(self.pdf_imgdatas):
            name = f"image_{i:04d}"
            description = {}
            description["figure_title"] = ""
            if img_data.image_has_figure_title:
                description["figure_title"] = img_data.image_figure_title_text
            description["surrounding_texts"] = img_data.image_surrounding_texts
            Image.fromarray(img_data.image).save(os.path.join(path, f"{name}.png"))
            open(os.path.join(path, f"{name}.json"), "w").write(json.dumps(description, ensure_ascii=False, indent=4))
        print(f"Export successfully!")
            
    def export_all_images_and_image_descriptions(self):
        """
        Docstring for export_all_images_and_image_descriptions
        
        :param self: Description
        說明:
            將上面的方法全部結合起來，一個指令直接提取 pdf 中所有圖片以及可能的說明文字
        """
        self.to_images(dpi=100)
        self.label_layout()
        self.label_images()
        self.extract_image_description(export=True)
        
    def __del__(self):
        self.pdf_doc.close()
        if os.path.exists(self.tmp_files_path):
            try:
                os.removedirs(self.tmp_files_path)
            except Exception as e:
                print(f"Please try manual delete {self.tmp_files_path}")