import os
import json
from converter import PdfInfo
from clip_faiss import MultiModalRetriever

# 設定路徑
# SOURCE_DIR = "agriculture_tech_docs"
SOURCE_DIR = "example_pdfs_2"
DB_SAVE_PATH = "agriculture_db"
PROCESSED_LOG = "processed_docs.json"
SAVE_INTERVAL = 5  # 每 5 個文件存檔一次

def load_processed_log():
    if os.path.exists(PROCESSED_LOG):
        try:
            with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_processed_log(processed_set):
    with open(PROCESSED_LOG, "w", encoding="utf-8") as f:
        json.dump(list(processed_set), f, ensure_ascii=False, indent=4)

def build_agriculture_database():
    # 1. 搜尋所有 PDF 文件
    pdf_files = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                pdf_files.append(full_path)
    
    print(f"找到 {len(pdf_files)} 個 PDF 文件。")

    # 2. 初始化或載入現有資料庫
    processed_docs = load_processed_log()
    if os.path.exists(DB_SAVE_PATH) and os.path.exists(os.path.join(DB_SAVE_PATH, "meta.json")):
        print(f"偵測到現有資料庫，正在載入續傳...")
        retriever = MultiModalRetriever.load(DB_SAVE_PATH)
    else:
        print(f"建立新的資料庫...")
        retriever = MultiModalRetriever()

    # 3. 逐一處理 PDF
    newly_processed_count = 0
    
    for i, pdf_path in enumerate(pdf_files):
        rel_path = os.path.relpath(pdf_path, SOURCE_DIR)
        
        # 檢查是否已處理過
        if rel_path in processed_docs:
            print(f"[{i+1}/{len(pdf_files)}] 跳過已處理文件: {rel_path}")
            continue
            
        print(f"\n[{i+1}/{len(pdf_files)}] 正在處理: {rel_path}")
        
        try:
            pdf = PdfInfo(pdf_path, gpu=True)
            pdf.to_images(dpi=300)
            pdf.label_layout()
            pdf.label_images(optimize_resolution=True, optimize_dpi=400, use_xref=False)
            pdf.extract_image_description(export=False)
            
            output_dir = f"output/{pdf.pdf_uid}/image_datas"
            pdf.export_all_image_datas(output_dir)
            
            indexed_count = retriever.add_folder(output_dir, doc_name_override=rel_path)
            
            # 更新處理記錄
            processed_docs.add(rel_path)
            newly_processed_count += 1
            print(f"成功索引 {indexed_count} 張圖片來自 {rel_path}")
            
            # 定期存檔快照
            if newly_processed_count % SAVE_INTERVAL == 0:
                print(f"\n💾 正在儲存快照 (已累計處理 {newly_processed_count} 個新文件)...")
                retriever.save(DB_SAVE_PATH)
                save_processed_log(processed_docs)

        except Exception as e:
            print(f"處理文件 {rel_path} 時發生錯誤: {e}")
        except KeyboardInterrupt:
            print(f"\n🛑 使用者中斷。正在緊急存檔目前進度...")
            retriever.save(DB_SAVE_PATH)
            save_processed_log(processed_docs)
            raise

    # 4. 最終儲存
    if newly_processed_count > 0:
        retriever.save(DB_SAVE_PATH)
        save_processed_log(processed_docs)
        print(f"\n✅ 所有新文件處理完成！資料庫已更新至: {DB_SAVE_PATH}")
    else:
        print("\nℹ️ 沒有新的文件需要處理。")

if __name__ == "__main__":
    build_agriculture_database()
