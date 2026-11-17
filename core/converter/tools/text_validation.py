import re
from typing import Optional, Dict

# 1. 判斷是否包含 Unicode 私用區字元（PDF 亂碼常見元兇）
def has_private_use(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if (
            0xE000 <= code <= 0xF8FF or
            0xF0000 <= code <= 0xFFFFD or
            0x100000 <= code <= 0x10FFFD
        ):
            return True
    return False


# 2. 計算「可讀字元比例」（中 + 英 + 數字 + 常見標點）
def valid_char_ratio(text: str) -> float:
    if not text:
        return 0.0

    valid_chars = re.findall(
        r'[\u4e00-\u9fffA-Za-z0-9，。！？,.()\[\]:\-/%]',
        text
    )
    return len(valid_chars) / len(text)


# 3. 核心：判斷文字是否為亂碼（中英混合適用）
def is_garbled_text(
    text: str,
    valid_threshold: float = 0.6,
    min_length: int = 5
) -> bool:
    # 太短或空白，直接視為不可用
    if not text or len(text.strip()) < min_length:
        return True

    # 有私用區字元，幾乎可確定是 PDF 字型映射亂碼
    if has_private_use(text):
        return True

    # 可讀字元比例過低
    if valid_char_ratio(text) < valid_threshold:
        return True

    return False


# 4. 版權頁、出版資訊、通訊處與刊頭雜訊過濾 (Masthead / Publisher Metadata)
def is_masthead_or_metadata(text: str) -> bool:
    """
    判斷段落是否為版權頁、專訊刊頭、編輯委員名單、出版單位或聯絡資訊等排版雜訊。
    通用於各類政府出版品、學術期刊、技術專訊及推廣手冊。
    """
    if not text:
        return False
    
    s = text.strip()
    
    masthead_indicators = [
        "發行人", "總編輯", "編輯委員", "執行編輯", "美術編輯", "出版機關", "出版單位", "發行所",
        "印刷所", "印刷單位", "工本費", "定價", "版權所有", "著作權所有", "翻印必究",
        "ISSN", "ISBN", "GPN", "All rights reserved", "Published by", "Printed in"
    ]
    contact_indicators = ["電話", "傳真", "地址", "E-mail", "email", "網址", "聯絡資訊"]
    
    # 命中 2 個以上刊頭/出版關鍵詞，高確信度判定為雜訊
    hit_masthead = sum(1 for ind in masthead_indicators if ind in s)
    if hit_masthead >= 2:
        return True
    
    # 命中 1 個出版關鍵詞且伴隨聯絡電話/地址/傳真
    if hit_masthead >= 1 and any(c in s for c in contact_indicators):
        return True
    
    # 純聯絡通訊區塊 (電話 + 傳真/地址/網址)
    hit_contact = sum(1 for c in contact_indicators if c in s)
    if hit_contact >= 2 and any(kw in s for kw in ["號", "路", "市", "區", "街", "樓", "http", "www", "com", "gov", "org", "tw"]):
        return True
        
    return False


# 5. 頁碼、眉批、簡易頁尾標籤判斷
def is_page_number_or_header(text: str) -> bool:
    """
    判斷段落是否僅為單純頁碼、頁首頁尾流水號或極短符號。
    例如: '12', '- 12 -', 'Page 12', '第 12 頁', '12 / 45'
    """
    s = text.strip()
    if not s:
        return True
    
    # 純數字或頁碼格式
    if re.match(r"^[-—~第\s]*\d+[\s\-—~頁\/of\d]*$", s, re.IGNORECASE):
        return True
    
    # 極短文字且無實質詞彙（小於等於 3 個字元且不含中文字或字母）
    if len(s) <= 3 and not any(ch.isalpha() or '\u4e00' <= ch <= '\u9fa5' for ch in s):
        return True
        
    return False


# 6. 文字清理與空白正規化
def clean_extracted_text(text: str, nl: bool = False) -> str:
    """清理文字中的連續空白與無效換行字符"""
    if not text:
        return ""
    if not nl:
        text = re.sub(r"[\r\n]+", " ", text)
    else:
        text = re.sub(r"[\r\n]{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# 7. 圖表標題 (Caption) 正則模式檢查
def is_caption_pattern(text: str) -> bool:
    """
    判斷段落開頭是否符合圖表 Caption 特徵。
    支援: 圖1、圖 1、圖一、Figure 1、Fig. 2、表 3、附圖 4 等。
    """
    if not text:
        return False
    
    # 匹配以 圖/表/Figure/Fig 開頭並緊接編號與分隔符（冒號、頓號、點、空格或緊接中文）
    pattern = (
        r"^\s*(?:圖|表|附圖|附表|照片|Figure|Fig\.?|FIG\.?|Plate|Photo|Table)"
        r"\s*([0-9]+|[一二三四五六七八九十]+|[IVXLCDM]+)(?:[-–—_][0-9A-Za-z]+|\([a-zA-Z]\))?"
        r"(?:[\s:：、\.\-—]|(?=[\u4e00-\u9fa5]))"
    )
    return bool(re.match(pattern, text, re.IGNORECASE))


# 8. 從標題文字提取圖表前綴與編號
def extract_figure_identifier(text: str) -> Optional[Dict[str, str]]:
    """
    從標題文字提取圖表編號與前綴。
    回傳 dict: {"prefix": "圖", "number": "1", "raw_key": "圖1"} 或 None
    """
    if not text:
        return None
        
    pattern = (
        r"^\s*(圖|表|附圖|附表|照片|Figure|Fig\.?|FIG\.?|Plate|Photo|Table)"
        r"\s*([0-9]+|[一二三四五六七八九十]+|[IVXLCDM]+)(?:[-–—_][0-9A-Za-z]+|\([a-zA-Z]\))?"
    )
    m = re.match(pattern, text, re.IGNORECASE)
    if m:
        prefix = m.group(1).strip()
        num = m.group(2).strip()
        return {
            "prefix": prefix,
            "number": num,
            "raw_key": f"{prefix}{num}"
        }
    return None
