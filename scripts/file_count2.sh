#!/bin/bash

TARGET_DIR="${1:-.}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "錯誤: 路徑 '$TARGET_DIR' 不存在。"
    exit 1
fi

declare -A counts

# 使用 find 找出剛好在第二層的目錄 (例如 ./季刊/臺南場)
# -mindepth 2 -maxdepth 2 確保只抓到單位層級
while IFS= read -r -d '' dir; do
    # 取得單位的名稱 (例如 臺南場)
    base_name=$(basename "$dir")
    
    # 計算該單位目錄下「所有層級」的檔案數量
    file_count=$(find "$dir" -type f | wc -l)
    
    # 加總到對應的單位名稱
    counts["$base_name"]=$(( ${counts["$base_name"]:-0} + file_count ))

done < <(find "$TARGET_DIR" -mindepth 2 -maxdepth 2 -type d -print0)

# 輸出結果
echo "統計路徑: $(realpath "$TARGET_DIR")"
echo "-------------------------------------------"
printf "%-20s %s\n" "單位名稱" "總檔案數量 (含各類刊物)"
echo "-------------------------------------------"

# 按檔案數量由大到小排序輸出
for name in "${!counts[@]}"; do
    printf "%-20s %d\n" "$name" "${counts[$name]}"
done | sort -rn -k2

echo "-------------------------------------------"