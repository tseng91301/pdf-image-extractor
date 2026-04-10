import json
import os
import shutil
from core.nlp_utils import extract_keywords

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
    for i, item in enumerate(meta_data):
        text_source = item.get("figure_title", "") + " "
        text_source += " ".join(item.get("sur_text_list", []))
        text_source = text_source.strip()
        
        # 使用統一的關鍵字提取邏輯
        item["keywords"] = extract_keywords(text_source, topK=8)
        
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
