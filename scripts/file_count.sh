#!/bin/bash

echo "-----------------------------------"
printf "%-30s %s\n" "目錄名稱" "檔案數量"
echo "-----------------------------------"

# 遍歷當前目錄下的子目錄
for dir in /home/nas2/Workspace/Jesse/LLM/organized_data/*; do
    # 移除目錄名稱後面的斜線 /
    dir_name=${dir%/}
    # 計算檔案數量 (含子目錄下的檔案)
    count=$(find "$dir" -type f | wc -l)
    
    # 格式化輸出，%-30s 代表左對齊並預留 30 個字元寬度
    printf "%-30s %d\n" "$dir_name" "$count"
done

echo "-----------------------------------"