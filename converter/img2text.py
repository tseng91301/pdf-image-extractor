import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

class ImgOcr:
    # 私有類別變數，用來快取初始化好的模型
    _ocr_gpu = None
    _ocr_cpu = None
    
    raw_image: np.ndarray
    extracted_text: str
    result = None

    def __init__(self, imgInput: np.ndarray, nl=False, gpu=False):
        # 取得對應的模型實例
        ocr = self._get_ocr(gpu)
        
        # 執行預測
        res = ocr.predict(imgInput)
        self.result = res
        self.extracted_text = ""
        
        for v in res[0]["rec_texts"]:
            self.extracted_text += v
            if nl:
                self.extracted_text += "\n"
        self.raw_image = imgInput

    @classmethod
    def _get_ocr(cls, gpu):
        """根據 GPU 設定，只在第一次調用時初始化模型"""
        if gpu:
            if cls._ocr_gpu is None:
                print("[ImgOcr] 初始化 GPU OCR 模型...")
                cls._ocr_gpu = PaddleOCR(
                    text_detection_model_name="PP-OCRv5_server_det",
                    text_recognition_model_name="PP-OCRv5_server_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device="gpu"
                )
            return cls._ocr_gpu
        else:
            if cls._ocr_cpu is None:
                print("[ImgOcr] 初始化 CPU OCR 模型...")
                cls._ocr_cpu = PaddleOCR(
                    text_detection_model_name="PP-OCRv5_server_det",
                    text_recognition_model_name="PP-OCRv5_server_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            return cls._ocr_cpu
