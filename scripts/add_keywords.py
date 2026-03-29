import json
import os
import shutil
import re

try:
    import jieba
    import jieba.analyse
except ImportError:
    print("請先安裝 jieba: pip install jieba")
    exit(1)

DB_META_PATH = "agriculture_db/meta.json"
DB_META_BAK = "agriculture_db/meta.json.bak.1"

def main():
    if not os.path.exists(DB_META_PATH):
        print(f"找不到資料庫的 meta 檔案: {DB_META_PATH}")
        return

    print(f"安全起見，正在備份 {DB_META_PATH} 到 {DB_META_BAK} ...")
    shutil.copy2(DB_META_PATH, DB_META_BAK)
    
    with open(DB_META_PATH, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
        
    print(f"總共需要處理 {len(meta_data)} 筆紀錄。")
    extracted_count = 0
    
    # 定義一個正規表達式：抓取 1~4 個中文字，加上這些農業常見特定結尾
    # 例如：炭疽病、斜紋夜蛾、紅蜘蛛蟎... 等等
    special_pattern = re.compile(r'[\u4e00-\u9fa5]{1,4}(?:病|蟲|害|菌|蛾|蟎|蝨|蠅|蝶|農藥|肥料)')
    
    for i, item in enumerate(meta_data):
        text_source = item.get("figure_title", "") + " "
        text_source += " ".join(item.get("sur_text_list", []))
        text_source = text_source.strip()
        
        if not text_source:
            item["keywords"] = []
            continue
            
        # 1. 使用 Jieba，並加上 allowPOS 過濾 (n: 名詞, nz: 其他專名, vn: 動名詞)
        jieba_keywords = jieba.analyse.extract_tags(
            text_source, 
            topK=8, 
            allowPOS=('n', 'nz', 'vn', 'ns')
        )
        
        # 2. 強制抓取特殊結尾的專有名詞
        special_matches = special_pattern.findall(text_source)
        
        # 3. 把兩者合併，並使用 set 來去除重複
        final_keywords = list(set(jieba_keywords + special_matches))
        
        item["keywords"] = final_keywords
        
        # 您可以保留印出的測試來觀看結果，正式跑的時候可以把這行註解掉
        # print(f"提取出: {final_keywords}")
        
        extracted_count += 1
        
        if (i + 1) % 50 == 0:
            print(f"進度: {i + 1}/{len(meta_data)} ...")
            
    # 寫回 JSON 檔案
    print(f"已完成關鍵字提取，正在寫入檔案...")
    with open(DB_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)
        
    print("🎉 程式執行完畢！")

if __name__ == "__main__":
    main()
