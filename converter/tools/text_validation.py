import re

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