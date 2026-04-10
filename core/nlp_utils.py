import re
import jieba
import jieba.analyse

# 定義一個正規表達式：抓取 1~4 個中文字，加上這些農業常見特定結尾
# 例如：炭疽病、斜紋夜蛾、紅蜘蛛蟎... 等等
AGRICULTURAL_PATTERN = re.compile(r'[\u4e00-\u9fa5]{1,4}(?:病|蟲|害|菌|蛾|蟎|蝨|蠅|蝶|農藥|肥料)')

def extract_keywords(text: str, topK: int = 8, allowPOS: tuple = ('n', 'nz', 'vn', 'ns')) -> list:
    """
    統一關鍵字提取邏輯，結合 Jieba 與農業專有正則表達式。
    
    Args:
        text (str): 待提取的文字
        topK (int): Jieba 提取的關鍵字數量
        allowPOS (tuple): Jieba 提取的詞性過濾
        
    Returns:
        list: 提取出的關鍵字列表（已去重）
    """
    if not text or not text.strip():
        return []
    
    # 1. 使用 Jieba 提取關鍵字
    # 使用 try-except 以防 jieba 或其字典未正確載入
    try:
        jieba_keywords = jieba.analyse.extract_tags(
            text, 
            topK=topK, 
            allowPOS=allowPOS
        )
    except Exception:
        # 如果提取失敗，退而求其次使用一般斷詞或空列表
        jieba_keywords = []

    # 2. 使用正則表達式抓取特定農業名詞
    special_matches = AGRICULTURAL_PATTERN.findall(text)
    
    # 3. 合併並去重
    final_keywords = list(set(jieba_keywords + special_matches))
    
    return final_keywords
