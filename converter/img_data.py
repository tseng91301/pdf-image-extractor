import numpy as np
from PIL import Image

from .distance import box_distance, normalize_box
from .img2text import ImgOcr

class ImgData:
    image: np.ndarray
    coordinate: list
    image_diagonal_length: float
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
            image (np.ndarray): 輸入的圖像 (numpy array)
            page_boxes (list): 其他Layout 元素的座標位置 [x1, y1, x2, y2]
            image_box_index (int): 圖片的 box index
        
        Description: 
            功能: 找出圖片本身以及圖片的標題或提示字
            方法: 
                1. 先尋找距離圖片最近、box_type 是 figure_title 以及位置在圖片底下的 box，若距離大於一個指定距離(這邊使用圖片對角線長度的 percentage 計算)則不採計
                2. 尋找距離圖片最近的 3 個 text 區塊
        """
        image_coordinate = normalize_box(page_boxes[image_box_index]['coordinate'])
        x1, y1, x2, y2 = map(int, image_coordinate)
        self.image = image[y1:y2, x1:x2]
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
            
    def get_surroundings(self, raw_image: np.ndarray):
        """
        功能: 從 __init__ 中提取出的 text box 以及 figure_title box 去 OCR 出文字
        """
        # figure_title 部分
        if self.image_has_figure_title:
            x1, y1, x2, y2 = map(int, self.image_figure_title_box['coordinate'])
            ocr = ImgOcr(raw_image[y1:y2, x1:x2], gpu=self.use_gpu)
            self.image_figure_title_text = ocr.extracted_text
        # text boxes 部分
        self.image_surrounding_texts = []
        for box in self.image_surrounding_text_boxes:
            x1, y1, x2, y2 = map(int, box['coordinate'])
            ocr = ImgOcr(raw_image[y1:y2, x1:x2], gpu=self.use_gpu)
            self.image_surrounding_texts.append(ocr.extracted_text)
        pass
    
    def save_image(self, path: str):
        Image.fromarray(self.image).save(path)