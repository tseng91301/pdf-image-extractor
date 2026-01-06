import random
import string

def generate(length=10) -> str:
    # 定義可用的字符集（英文字母和數字）
    characters = string.ascii_letters + string.digits

    # 隨機選擇 10 個字符
    random_string = ''.join(random.choices(characters, k=length))
    return random_string