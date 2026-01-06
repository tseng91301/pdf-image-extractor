import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

class ImgOcr:
    raw_image: np.ndarray
    extracted_text: str
    result = None
    def __init__(self, imgInput: np.ndarray, nl=False, gpu=False):
        if gpu:
            ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="gpu"
            )
        else:
            ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        res = ocr.predict(imgInput)
        self.result = res
        self.extracted_text = ""
        for v in res[0]["rec_texts"]:
            self.extracted_text += v
            if nl:
                self.extracted_text += "\n"
        self.raw_image = imgInput