import os
import shutil
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# 自動檢測並過濾相容之 GPU 架構 (排除超過 SM 9.0 之 Blackwell，鎖定 RTX A6000)
from .gpu_config import setup_paddle_gpu_environment
setup_paddle_gpu_environment()

import fitz  # PyMuPDF
from PIL import Image
import cv2
import json
import re
import numpy as np

import paddle
from paddleocr import LayoutDetection

from .tools import random_uid
from .tools.coordinates import map_bbox
from .tools.text_validation import (
    extract_figure_identifier,
    is_masthead_or_metadata,
    clean_extracted_text
)
from .img_data import ImgData

def match_xref_for_rect(page: fitz.Page, rect_pdf: fitz.Rect):
    """
    回傳：best_xref, best_image_rect, best_coverage
    best_coverage = intersection_area / rect_area
    """
    rect_pdf = rect_pdf & page.rect  # 保險：裁到頁面內
    if rect_pdf.is_empty:
        return (None, None, 0.0)

    best = (None, None, 0.0)

    for img in page.get_images(full=True):
        xref = img[0]
        rects = page.get_image_rects(xref)  # 這張 xref 在頁面上出現的位置(可能多個)

        for r in rects:
            inter = r & rect_pdf
            if inter.is_empty:
                continue

            coverage = inter.get_area() / (rect_pdf.get_area() + 1e-9)
            if coverage > best[2]:
                best = (xref, r, coverage)

    return best

class PdfInfo:
    _layout_model = None  # 用來快取單例的 LayoutDiscovery 模型
    pdf_path: str
    pdf_uid: str
    pdf_name: str
    tmp_files_path = "tmp/PdfInfo/"
    pdf_doc: fitz.Document = None
    use_gpu = False
    
    scanned_to_images = False
    pdf_img_paths = []
    
    pdf_layouts = []
    pdf_imgdatas = []
    def __init__(self, pdf_path, gpu=None):
        self.pdf_path = pdf_path
        self.pdf_uid = random_uid.generate()
        self.tmp_files_path = os.path.join(self.tmp_files_path, self.pdf_uid)
        self.pdf_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
        os.makedirs(self.tmp_files_path, exist_ok=True)
        self.pdf_doc = fitz.open(self.pdf_path)
        
        # GPU Check
        if gpu is None:
            gpu = paddle.device.is_compiled_with_cuda()
            if not gpu:
                print("\033[93m[WARNING] [PdfInfo] CUDA is not available. Falling back to CPU.\033[0m")
        elif gpu and not paddle.device.is_compiled_with_cuda():
            print("\033[93m[WARNING] [PdfInfo] GPU requested but CUDA is not available. Falling back to CPU.\033[0m")
            gpu = False
            
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
        if PdfInfo._layout_model is None:
            print("[PdfInfo] 初始化 LayoutDetection 模型 (PP-DocLayout_plus-L)...")
            PdfInfo._layout_model = LayoutDetection(model_name="PP-DocLayout_plus-L")
        model = PdfInfo._layout_model
        for i, img_path in enumerate(self.pdf_img_paths):
            p = model.predict(img_path, batch_size=1, layout_nms=True)
            p[0]['input_img'] = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) # 使用原本擷取出來的圖片
            self.pdf_layouts.append(p[0])
            
        if output:
            for i, v in enumerate(self.pdf_layouts):
                v.save_to_img(f"output/{self.pdf_uid}/layout_detection/{i+1}.png")
            print(f"Layout results saved into output/{self.pdf_uid}/layout_detection/")
        return self.pdf_layouts
    
    def label_images(self, optimize_resolution=False, optimize_dpi=500, use_xref=True):
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
                        gpu=self.use_gpu,
                        raw_pdf_path=self.pdf_path,
                        img_page=i0
                    )
                    if optimize_resolution:
                        # 從原版的 pdf 文件用較高畫質擷取圖片範圍
                        now_img_data_index = len(self.pdf_imgdatas)
                        page = self.pdf_doc[i0]
                        pdf_raw_width = page.rect.width
                        pdf_raw_height = page.rect.height
                        (pdf_bitmap_height, pdf_bitmap_width) = l['input_img'].shape[:2]
                        x1, y1, x2, y2 = map(int, l['boxes'][i]['coordinate'])
                        (x1_t, y1_t, x2_t, y2_t) = map_bbox(x1, y1, x2, y2, pdf_bitmap_width, pdf_bitmap_height, pdf_raw_width, pdf_raw_height)
                        rect = fitz.Rect(int(x1_t), int(y1_t), int(x2_t), int(y2_t))
                        # 先使用 xref 尋找可能符合的圖片
                        use_render = not use_xref
                        if use_xref:
                            (xref, _, cov) = match_xref_for_rect(page, rect)
                            if xref is not None and cov >= 0.6:
                                info = self.pdf_doc.extract_image(xref)
                                img_bytes = info["image"]          # bytes
                                img = cv2.cvtColor(cv2.imdecode(
                                    np.frombuffer(img_bytes, np.uint8),
                                    cv2.IMREAD_COLOR               # BGR, uint8
                                ), cv2.COLOR_BGR2RGB)
                                print(f"Image {now_img_data_index+1} using xref")
                            else:
                                use_render = True
                        if use_render:
                            # 直接渲染出來並剪取圖片
                            pix = page.get_pixmap(dpi=optimize_dpi, clip=rect, colorspace=fitz.csRGB)  # 高dpi裁切
                            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                            print(f"Image {now_img_data_index+1} using render")
                        i_d.update_image(img)
                        pass
                    self.pdf_imgdatas.append(i_d)
        return self.pdf_imgdatas
    
    def resolve_cross_references(self, max_refs_per_image: int = 3, max_page_distance: int = 2):
        """
        跨頁面/跨段落正文引用關聯 (Cross-Reference Resolution):
        1. 針對每張已辨識出圖表編號 (例如「圖1」、「Figure 2」、「表 3」) 的圖片，
        2. 全文檢索所有 PDF 頁面中的正文段落，尋找對該圖的引用 (例如「如圖1」、「見圖 1 所示」、「(Figure 2)」)，
        3. 將最相關的引用段落標註為 [正文引用 第X頁] 並加入到該圖片的 surrounding_texts 與 cross_references，
        4. 嚴格加入 max_page_distance 篇章邊界約束（預設前後 2 頁內），防止專訊/期刊跨不同獨立文章大串味。
        """
        if not self.pdf_imgdatas:
            return

        # 1. 建立 figure_key -> list of (img_data, fig_info) 映射
        fig_map = {}
        for img_data in self.pdf_imgdatas:
            if not img_data.image_has_figure_title or not img_data.image_figure_title_text:
                continue
            fig_info = extract_figure_identifier(img_data.image_figure_title_text)
            if fig_info:
                key = fig_info["raw_key"]
                if key not in fig_map:
                    fig_map[key] = []
                fig_map[key].append((img_data, fig_info))

        if not fig_map:
            return

        print(f"[PdfInfo] 啟動全文正文引用關聯分析 (Cross-Reference Resolution)，針對 {len(fig_map)} 組圖表編號...")

        # 2. 為每個 figure key 預先編譯正則表達式
        compiled_patterns = {}
        for key, items in fig_map.items():
            fig_info = items[0][1]
            prefix = fig_info["prefix"]
            num = fig_info["number"]

            # 正則引用特徵：
            # (A) 括弧引用: (如圖1), （見圖 1）, (Fig. 1), (Figure 1A)
            # (B) 正文動詞引導: 如圖1, 見圖1, 參見圖1, 如圖 1 所示, 詳見圖1
            # (C) 主詞引導: 圖1為, 圖 1 是, 圖1顯示, 圖1表示, Figure 1 shows, Fig. 1 illustrates
            pat_str = (
                rf"(?:"
                rf"[（\(]\s*(?:如|見|參見|參閱|詳見|請參照|如所附|see)?\s*(?:{re.escape(prefix)})\s*{re.escape(num)}\b[^\)）]{{0,30}}[）\)]"
                rf"|"
                rf"(?:如|見|參見|參閱|詳見|請參照)\s*(?:{re.escape(prefix)})\s*{re.escape(num)}(?=[^\d]|$)"
                rf"|"
                rf"(?:{re.escape(prefix)})\s*{re.escape(num)}\s*(?:為|是|顯示|表示|顯示出|所示|中可見|中可知|呈現|指|係指)"
                rf"|"
                rf"(?:Figure|Fig\.?)\s*{re.escape(num)}\s+(?:shows|illustrates|depicts|demonstrates|presents|indicates|reveals)"
                rf")"
            )
            compiled_patterns[key] = re.compile(pat_str, re.IGNORECASE)

        # 3. 掃描 PDF 全文所有頁面之文字塊 (Text Blocks)
        total_linked = 0
        for page_idx in range(len(self.pdf_doc)):
            page = self.pdf_doc[page_idx]
            blocks = page.get_text("blocks")
            for block in blocks:
                if len(block) < 7 or block[6] != 0:  # block_type == 0 為純文字
                    continue

                raw_text = block[4].strip()
                if len(raw_text) < 15:
                    continue

                clean_para = clean_extracted_text(raw_text, nl=False)
                if is_masthead_or_metadata(clean_para):
                    continue

                for key, pattern in compiled_patterns.items():
                    if pattern.search(clean_para):
                        for img_data, fig_info in fig_map[key]:
                            # 限制正文引用必須在圖片所在頁面的前後 max_page_distance 頁以內 (預設 2 頁)
                            # 避免跨越期刊/專訊中不同技術文章造成嚴重的跨領域語意污染
                            if abs(page_idx - img_data.img_page) > max_page_distance:
                                continue

                            # 避免將 Caption 本身當作引用
                            if (
                                img_data.img_page == page_idx and
                                img_data.image_figure_title_text and
                                img_data.image_figure_title_text in clean_para and
                                len(clean_para) - len(img_data.image_figure_title_text) < 20
                            ):
                                continue

                            ref_entry = f"[正文引用 第{page_idx+1}頁] {clean_para}"
                            if ref_entry not in img_data.cross_references:
                                if len(img_data.cross_references) < max_refs_per_image:
                                    img_data.cross_references.append(ref_entry)
                                    if ref_entry not in img_data.image_surrounding_texts:
                                        img_data.image_surrounding_texts.append(ref_entry)
                                        total_linked += 1

        print(f"[PdfInfo] 全文正文引用關聯分析完成，共成功鏈結 {total_linked} 處正文引用至對應圖片。")

    def extract_image_description(self, export=False, nl=False, resolve_refs=True):
        """
        將所有圖片周圍的 title, text 的文字資料提取出來
        方法: 先嘗試使用一般的 pdf 文字提取，若提取不到或是提取出來為亂碼會自動使用 OCR 來取得文字。
        接著執行 resolve_cross_references() 全文正文引用關聯分析。
        """
        n = len(self.pdf_imgdatas)
        for i, img in enumerate(self.pdf_imgdatas):
            print(f"Getting surrounding texts and figure_title of image {i+1}/{n}, reading page {img.img_page+1}")
            img.get_surroundings(cv2.imread(self.pdf_img_paths[img.img_page]), nl=nl)

        # 同一頁面共同圖題防搶奪去重 (Single-Caption Claim per page)
        # 若同頁多張圖片搶奪同一個 Caption 方框，僅保留評分最高者，避免同頁多圖撞名
        imgs_by_page = {}
        for img in self.pdf_imgdatas:
            imgs_by_page.setdefault(img.img_page, []).append(img)

        for page_idx, page_imgs in imgs_by_page.items():
            title_to_imgs = {}
            for img in page_imgs:
                if img.image_has_figure_title and img.image_figure_title_text:
                    title_to_imgs.setdefault(img.image_figure_title_text, []).append(img)

            for title_text, claimed_imgs in title_to_imgs.items():
                if len(claimed_imgs) > 1:
                    # 依據標題評分由高至低排序，最高分者勝出
                    claimed_imgs.sort(key=lambda x: getattr(x, 'image_caption_score', 0.0), reverse=True)
                    # 其餘圖片撤銷圖題，避免同頁圖片產生假性圖題碰撞
                    for other_img in claimed_imgs[1:]:
                        other_img.image_has_figure_title = False
                        other_img.image_figure_title_text = ""
                        other_img.image_figure_title_box = None
                        other_img.image_caption_score = -1.0
            
        if resolve_refs:
            self.resolve_cross_references()
            
        if export:
            self.export_all_image_datas()
            
    def export_all_image_datas(self, path: str=None, doc_name: str=None):
        metadata = {}
        metadata["name"] = doc_name if doc_name is not None else self.pdf_name
        metadata["uid"] = self.pdf_uid
        metadata["imgs"] = []
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
            description["cross_references"] = getattr(img_data, "cross_references", [])
            img_data.save_image(os.path.join(path, f"{name}.png"))
            metadata["imgs"].append({
                "name": name,
                "page": img_data.img_page+1,
                "coordinate": img_data.coordinate,
                "figure_title": description["figure_title"],
                "surrounding_texts": description["surrounding_texts"],
                "cross_references": description["cross_references"]
            })
        open(os.path.join(path, "metadata.json"), "w").write(json.dumps(metadata, ensure_ascii=False, indent=4))
        print(f"Export successfully!")
            
    def export_all_images_and_image_descriptions(self):
        """
        Docstring for export_all_images_and_image_descriptions
        
        說明:
            將上面的方法全部結合起來，一個指令直接提取 pdf 中所有圖片以及可能的說明文字
        """
        self.to_images(dpi=100)
        self.label_layout()
        self.label_images()
        self.extract_image_description(export=True)
        
    def cleanup(self):
        if hasattr(self, "pdf_doc") and self.pdf_doc is not None:
            try:
                self.pdf_doc.close()
            except Exception:
                pass
        if hasattr(self, "tmp_files_path") and os.path.exists(self.tmp_files_path):
            try:
                shutil.rmtree(self.tmp_files_path, ignore_errors=True)
            except Exception:
                pass
        for img_data in getattr(self, "pdf_imgdatas", []):
            if hasattr(img_data, "store_path") and os.path.exists(img_data.store_path):
                try:
                    shutil.rmtree(os.path.dirname(img_data.store_path), ignore_errors=True)
                except Exception:
                    pass

    def __del__(self):
        self.cleanup()
