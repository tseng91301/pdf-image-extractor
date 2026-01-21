import numpy as np
from PIL import Image
import fitz
import os

from .distance import box_distance, normalize_box
from .img2text import ImgOcr
from .tools.coordinates import map_bbox
from .tools.text_validation import is_garbled_text
from .tools import random_uid

class ImgData:
    uid: str
    store_path = "tmp/ImgData"
    coordinate: list
    image_diagonal_length: float
    raw_pdf_path: str # 原始的 PDF 文件所在位置，提取圖片周圍文字時會使用到
    img_page: int # 使用 pdf 解析功能時，需要知道圖片所在的頁數
    
    image_has_figure_title: bool
    image_figure_title_box: dict
    image_surrounding_text_boxes: list
    
    image_figure_title_text: str
    image_surrounding_texts: list
    
    use_gpu = False
    
    def __init__(self, image: np.ndarray, page_boxes: list, image_box_index: int, figure_title_threshold: float = 0.05, gpu=False):
        """_summary_

        Args:
            image (np.ndarray): 輸入的圖像 (numpy array) (通常為完整的頁面)
            page_boxes (list): 其他 Layout 元素的座標位置 [x1, y1, x2, y2]
            image_box_index (int): 圖片的 box index
        
        Description: 
            功能: 找出圖片本身以及圖片的標題或提示字
            方法: 
                1. 先尋找距離圖片最近、box_type 是 figure_title 以及位置在圖片底下的 box，若距離大於一個指定距離(這邊使用圖片對角線長度的 percentage 計算)則不採計
                2. 尋找距離圖片最近的 3 個 text 區塊
        """
        image_coordinate = normalize_box(page_boxes[image_box_index]['coordinate'])
        self.uid = random_uid.generate()
        
        # 將圖片透過座標方框擷取出來並暫存
        x1, y1, x2, y2 = map(int, image_coordinate)
        self.store_path = os.path.join(self.store_path, self.uid)
        os.makedirs(self.store_path, exist_ok=True)
        self.store_path = os.path.join(self.store_path, "image.png")
        image = image[y1:y2, x1:x2]
        Image.fromarray(image).save(self.store_path)
        
        self.image_diagonal_length = np.linalg.norm(
            np.array(image_coordinate[2:]) - np.array(image_coordinate[:2])
        )
        self.image_has_figure_title = False
        self.image_figure_title_box = None
        self.image_surrounding_text_boxes = []
        self.coordinate = image_coordinate
        
        self.use_gpu = gpu
        
        # 偵測圖片周圍的可用 boxes
        min_figure_title_distance = self.image_diagonal_length * figure_title_threshold
        min_figure_title: dict = None
        text_distances = []
        for box in page_boxes:
            d = box_distance(image_coordinate, box['coordinate'], type="min")
            if box["label"] == "figure_title":
                if d < min_figure_title_distance:
                    min_figure_title = box
                    min_figure_title_distance = d
            elif box["label"] == "text":
                text_distances.append((box, d))
        
        if min_figure_title is not None:
            self.image_has_figure_title = True
            self.image_figure_title_box = min_figure_title
        
        text_distances.sort(key=lambda x: x[1])
        for box, d in text_distances[:3]:
            self.image_surrounding_text_boxes.append(box)
    
    def update_image(self, image: np.ndarray):
        """
        更新物件所儲存的圖片本體 (圖片本體不會再被使用到，因此可以被更新)
        輸入: 純圖片的 numpy 陣列
        動作: 更新儲存於暫存區的圖片檔案
        """
        Image.fromarray(image).save(self.store_path)
            
    def get_surroundings(self, raw_image: np.ndarray):
        """
        功能: 從 __init__ 中提取出的 text box 以及 figure_title box 去 OCR 出文字
        """
        # 取得 raw pdf
        doc = fitz.open(self.raw_pdf_path)
        page = doc[self.img_page]
        raw_width = page.rect.width
        raw_height = page.rect.height
        (img_height, img_width) = raw_image.shape[:2]
        # print(f"raw_width={raw_width}, raw_height={raw_height}, img_width={img_width}, img_height={img_height}")
        
        # figure_title 部分
        if self.image_has_figure_title:
            x1, y1, x2, y2 = map(int, self.image_figure_title_box['coordinate'])
            # 嘗試直接提取自元
            (x1_t, y1_t, x2_t, y2_t) = map_bbox(x1, y1, x2, y2, img_width, img_height, raw_width, raw_height)
            rect = fitz.Rect(int(x1_t), int(y1_t), int(x2_t), int(y2_t))
            text = page.get_text("text", clip=rect)
            if is_garbled_text(text) or len(text) == 0:
                ocr = ImgOcr(raw_image[y1:y2, x1:x2], gpu=self.use_gpu)
                self.image_figure_title_text = ocr.extracted_text
            else:
                self.image_figure_title_text = text
        # text boxes 部分
        self.image_surrounding_texts = []
        for box in self.image_surrounding_text_boxes:
            x1, y1, x2, y2 = map(int, box['coordinate'])
            # 嘗試直接提取自元
            (x1_t, y1_t, x2_t, y2_t) = map_bbox(x1, y1, x2, y2, img_width, img_height, raw_width, raw_height)
            # print(f"Extract words from ({x1_t}, {y1_t}, {x2_t}, {y2_t}) at page {self.img_page+1}, image coordinates=({x1}, {y1}, {x2}, {y2})")
            rect = fitz.Rect(int(x1_t), int(y1_t), int(x2_t), int(y2_t))
            text = page.get_text("text", clip=rect)
            if is_garbled_text(text) or len(text) == 0:
                ocr = ImgOcr(raw_image[y1:y2, x1:x2], gpu=self.use_gpu)
                self.image_surrounding_texts.append(ocr.extracted_text)
            else:
                self.image_surrounding_texts.append(text)
        pass
    
    def save_image(self, path: str):
        with open(path, "wb") as f:
            with open(self.store_path, "rb") as r:
                f.write(r.read())
                r.close()
                f.close()
        