import numpy as np
from PIL import Image
import fitz
import os
import re
from typing import List, Optional, Dict, Any

from .distance import (
    box_distance,
    normalize_box,
    box_iou,
    horizontal_overlap_ratio,
    column_aware_distance
)
from .img2text import ImgOcr
from .tools.coordinates import map_bbox
from .tools.text_validation import (
    is_garbled_text,
    is_masthead_or_metadata,
    is_page_number_or_header,
    clean_extracted_text,
    is_caption_pattern,
    extract_figure_identifier
)
from .tools import random_uid


class ImgData:
    uid: str
    store_path: str = "tmp/ImgData"
    coordinate: list
    image_diagonal_length: float
    raw_pdf_path: str  # 原始的 PDF 文件所在位置
    img_page: int  # 圖片所在的頁數 (0-indexed)
    page_shape: tuple  # (height, width)
    page_boxes: list  # 當前頁面的所有版面偵測方框
    
    image_has_figure_title: bool
    image_figure_title_box: Optional[dict]
    image_surrounding_text_boxes: list
    
    image_figure_title_text: str
    image_caption_score: float
    image_surrounding_texts: list
    cross_references: list  # 全文跨段正文引用 (Cross-References)
    
    use_gpu: bool = False
    
    def __init__(
        self,
        image: np.ndarray,
        page_boxes: list,
        image_box_index: int,
        figure_title_threshold: float = 0.18,
        gpu: bool = False,
        raw_pdf_path: Optional[str] = None,
        img_page: int = 0
    ):
        """
        初始化單張圖片的資料物件，並初步關聯候選 Caption 與周圍段落方框。

        Args:
            image (np.ndarray): 完整的頁面圖像陣列 (H, W, C)
            page_boxes (list): PP-DocLayout 偵測出的所有頁面元素方框
            image_box_index (int): 本圖片在 page_boxes 中的索引
            figure_title_threshold (float): 圖表標題最大搜尋半徑 (對角線比例，預設放寬至 0.18)
            gpu (bool): 是否使用 GPU 進行 OCR
            raw_pdf_path (str): 原始 PDF 路徑
            img_page (int): 圖片所在頁數 (0-indexed)
        """
        page_height, page_width = image.shape[:2]
        self.page_shape = (page_height, page_width)
        self.page_boxes = page_boxes
        self.raw_pdf_path = raw_pdf_path
        self.img_page = img_page
        
        image_coordinate = normalize_box(page_boxes[image_box_index]['coordinate'])
        self.uid = random_uid.generate()
        self.coordinate = image_coordinate
        self.use_gpu = gpu
        self.cross_references = []
        self.image_caption_score = -1.0
        
        # 裁切並儲存圖片本體至暫存目錄
        x1, y1, x2, y2 = map(int, image_coordinate)
        # 防止座標超出圖像邊界
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(page_width, x2), min(page_height, y2)
        
        self.store_path = os.path.join("tmp", "ImgData", self.uid)
        os.makedirs(self.store_path, exist_ok=True)
        self.store_path = os.path.join(self.store_path, "image.png")
        cropped_img = image[y1:y2, x1:x2]
        Image.fromarray(cropped_img).save(self.store_path)
        
        # 計算對角線長度作為相對距離基準
        self.image_diagonal_length = float(np.linalg.norm(
            np.array(image_coordinate[2:]) - np.array(image_coordinate[:2])
        ))
        
        self.image_has_figure_title = False
        self.image_figure_title_box = None
        self.image_figure_title_text = ""
        self.image_surrounding_text_boxes = []
        self.image_surrounding_texts = []
        
        # 初步幾何關聯 (供尚未呼叫 get_surroundings 時使用)
        self._initial_box_association(figure_title_threshold)

    def _initial_box_association(self, figure_title_threshold: float):
        """根據放寬後的幾何距離與欄位感知，進行初版方框關聯"""
        page_height = self.page_shape[0]
        header_limit = page_height * 0.05
        footer_limit = page_height * 0.92
        
        max_title_dist = self.image_diagonal_length * figure_title_threshold
        min_title_dist = float("inf")
        best_title_box = None
        
        text_candidates = []
        
        for idx, box in enumerate(self.page_boxes):
            b_coord = normalize_box(box['coordinate'])
            b_y1, b_y2 = b_coord[1], b_coord[3]
            b_cent_y = (b_y1 + b_y2) / 2.0
            
            # 過濾頁首與頁尾
            if b_cent_y < header_limit or b_cent_y > footer_limit:
                continue
                
            # 過濾與圖片重疊度過高的方框 (如圖內浮水印或嵌字)
            if box_iou(self.coordinate, b_coord) > 0.4:
                continue
                
            label = box.get("label", "")
            if label == "figure_title":
                d = box_distance(self.coordinate, b_coord, type="min")
                if d < max_title_dist and d < min_title_dist:
                    min_title_dist = d
                    best_title_box = box
            elif label in ["text", "paragraph"]:
                # 使用欄位感知距離計算段落關聯
                cad = column_aware_distance(self.coordinate, b_coord, col_penalty=3.0)
                text_candidates.append((box, cad))
                
        if best_title_box is not None:
            self.image_has_figure_title = True
            self.image_figure_title_box = best_title_box
            
        # 依欄位感知距離排序取最近的 3 個候選段落
        text_candidates.sort(key=lambda x: x[1])
        for b, _ in text_candidates[:3]:
            self.image_surrounding_text_boxes.append(b)

    def update_image(self, image: np.ndarray):
        """更新物件所儲存的圖片本體 (例如替換為高解析度裁切版本)"""
        Image.fromarray(image).save(self.store_path)

    def get_surroundings(self, raw_image: np.ndarray, nl: bool = False):
        """
        從 PDF 向量文字層與 PaddleOCR 雙軌提取圖表標題與周圍段落，
        並進行 Caption 正則回退判斷、多欄位校正與刊頭/版權雜訊過濾。
        """
        if not self.raw_pdf_path or not os.path.exists(self.raw_pdf_path):
            raise FileNotFoundError(f"PDF file not found: {self.raw_pdf_path}")

        doc = fitz.open(self.raw_pdf_path)
        try:
            page = doc[self.img_page]
            raw_width = page.rect.width
            raw_height = page.rect.height
            img_height, img_width = raw_image.shape[:2]
            
            # 輔助函式：從 PDF 座標裁切並提取文字
            def _extract_box_text(b_coord: list, try_ocr: bool = True, is_title: bool = False) -> str:
                bx1, by1, bx2, by2 = map(int, b_coord)
                bx1_t, by1_t, bx2_t, by2_t = map_bbox(
                    bx1, by1, bx2, by2, img_width, img_height, raw_width, raw_height
                )
                clip_rect = fitz.Rect(int(bx1_t), int(by1_t), int(bx2_t), int(by2_t))
                txt = page.get_text("text", clip=clip_rect).strip()
                
                min_len = 2 if is_title else 5
                if is_garbled_text(txt, min_length=min_len) or len(txt) == 0:
                    if try_ocr and (by2 > by1) and (bx2 > bx1):
                        crop_patch = raw_image[max(0, by1):min(img_height, by2), max(0, bx1):min(img_width, bx2)]
                        if crop_patch.size > 0:
                            ocr = ImgOcr(crop_patch, gpu=self.use_gpu, nl=nl)
                            txt = ocr.extracted_text
                return clean_extracted_text(txt, nl=nl)

            # ========================================================
            # 1. 雙軌圖表標題 (Caption) 偵測與正則回退機制
            # ========================================================
            # 候選評分標準：結合 PP-DocLayout 標籤、Caption 正則匹配、方位 (上下對齊) 與距離
            best_caption_score = -1.0
            best_caption_box = None
            best_caption_text = ""
            
            # 標題搜索範圍：放寬至對角線 22% (垂直下方優先)
            max_caption_dist = self.image_diagonal_length * 0.22
            header_limit = img_height * 0.05
            footer_limit = img_height * 0.92

            for box in self.page_boxes:
                b_coord = normalize_box(box['coordinate'])
                b_cent_y = (b_coord[1] + b_coord[3]) / 2.0
                
                # 排除頁首頁尾與與圖片本身高度重疊之方框
                if b_cent_y < header_limit or b_cent_y > footer_limit:
                    continue
                if box_iou(self.coordinate, b_coord) > 0.4:
                    continue
                    
                d = box_distance(self.coordinate, b_coord, type="min")
                if d > max_caption_dist:
                    continue
                    
                label = box.get("label", "")
                overlap_x = horizontal_overlap_ratio(self.coordinate, b_coord)
                
                # 方位判斷：緊鄰下方 (最常見) 或緊鄰上方
                is_below = (b_coord[1] >= self.coordinate[3] - 15)
                is_above = (b_coord[3] <= self.coordinate[1] + 15)
                
                # 提取初步文字進行正則檢驗
                box_text = _extract_box_text(b_coord, try_ocr=True, is_title=True)
                if not box_text:
                    continue
                    
                # 評分系統
                score = 0.0
                has_pattern = is_caption_pattern(box_text)
                
                if has_pattern:
                    score += 15.0  # 符合「圖1/Figure 1」正則特徵，給予最高權重
                if label == "figure_title":
                    score += 8.0   # 模型辨識為 figure_title
                if is_below and overlap_x >= 0.25:
                    score += 6.0   # 正下方水平對齊，典型 Caption 排版
                elif is_above and overlap_x >= 0.25:
                    score += 3.0   # 正上方水平對齊
                    
                # 距離扣分項
                score -= (d / (self.image_diagonal_length + 1e-6)) * 10.0
                
                # 門檻要求：若有 Caption 正則特徵只要 >= 10 分即可；若無正則特徵則需至少為 figure_title 標籤 (>= 6 分)
                threshold = 10.0 if has_pattern else 6.0
                if score > threshold and score > best_caption_score:
                    best_caption_score = score
                    best_caption_box = box
                    best_caption_text = box_text

            if best_caption_box is not None:
                self.image_has_figure_title = True
                self.image_figure_title_box = best_caption_box
                self.image_figure_title_text = best_caption_text
                self.image_caption_score = best_caption_score
            else:
                self.image_has_figure_title = False
                self.image_figure_title_box = None
                self.image_figure_title_text = ""
                self.image_caption_score = -1.0

            # ========================================================
            # 2. 周圍段落 (Surrounding Texts) 欄位感知篩選與排版過濾
            # ========================================================
            candidate_texts = []
            for box in self.page_boxes:
                # 排除已被選為 Caption 的方框，避免內文與圖題重複
                if best_caption_box is not None and box is best_caption_box:
                    continue
                    
                b_coord = normalize_box(box['coordinate'])
                b_cent_y = (b_coord[1] + b_coord[3]) / 2.0
                
                # 排除頁首頁尾
                if b_cent_y < header_limit or b_cent_y > footer_limit:
                    continue
                # 排除與圖片重疊的嵌字
                if box_iou(self.coordinate, b_coord) > 0.4:
                    continue
                    
                label = box.get("label", "")
                if label in ["text", "paragraph"]:
                    # 計算欄位感知距離 (跨欄懲罰 3.0x)
                    cad = column_aware_distance(self.coordinate, b_coord, col_penalty=3.0)
                    candidate_texts.append((box, cad))
                    
            # 依欄位感知距離由近至遠排序，取前 6 個候選段落進行文字品質檢驗
            candidate_texts.sort(key=lambda x: x[1])
            
            self.image_surrounding_texts = []
            self.image_surrounding_text_boxes = []
            
            for b, dist in candidate_texts[:6]:
                txt = _extract_box_text(b['coordinate'], try_ocr=True, is_title=False)
                
                # 品質過濾機制：
                # A. 排除版權頁、專訊刊頭、編輯委員名單與通訊地址等刊印雜訊
                if is_masthead_or_metadata(txt):
                    continue
                # B. 排除單純頁碼或無意義流水號
                if is_page_number_or_header(txt):
                    continue
                # C. 排除極短破碎字串
                if len(txt) < 8:
                    continue
                # D. 排除與圖題完全相同的重複段落
                if self.image_has_figure_title and txt == self.image_figure_title_text:
                    continue
                # E. 排除已加入的重複段落
                if txt in self.image_surrounding_texts:
                    continue
                    
                self.image_surrounding_texts.append(txt)
                self.image_surrounding_text_boxes.append(b)
                
                # 最多取 3 個最高品質的周圍段落
                if len(self.image_surrounding_texts) >= 3:
                    break

        finally:
            doc.close()

    def save_image(self, path: str):
        """儲存圖片至目標路徑"""
        with open(path, "wb") as f:
            with open(self.store_path, "rb") as r:
                f.write(r.read())
