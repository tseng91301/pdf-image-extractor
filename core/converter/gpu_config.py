"""
GPU environment configuration and architecture compatibility checker for PaddlePaddle.
"""
import os
import subprocess
import logging

logger = logging.getLogger("gpu_config")

def setup_paddle_gpu_environment():
    """
    自動檢測並過濾 PaddlePaddle 支援之 GPU 架構。
    當前官方 PaddlePaddle 預編譯 wheel 支援的 CUDA 架構為: 61, 70, 75, 80, 86, 89, 90。
    若系統中存在高於 SM 9.0 之顯卡 (例如 NVIDIA Blackwell 12.0)，Paddle 初始化時會丟出
    'RuntimeError: Unsupported GPU architecture'。
    
    本函式會在載入 Paddle 前自動偵測顯卡算力，並將 CUDA_VISIBLE_DEVICES 限制在
    與 PaddlePaddle 相容的顯卡上 (例如伺服器上的 RTX A6000 / SM 8.6)，
    確保能在全 GPU 加速下穩定運行。
    """
    # 關閉 PaddleX 連線檢查，加速初始化
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    
    # 若使用者已手動指定 CUDA_VISIBLE_DEVICES，則尊重使用者設定
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return

    try:
        # 使用 nvidia-smi 查詢所有 GPU 的 PCI 序號、名稱與 Compute Capability
        cmd = ["nvidia-smi", "--query-gpu=index,name,compute_cap", "--format=csv,noheader,nounits"]
        out = subprocess.check_output(cmd, encoding="utf-8", stderr=subprocess.DEVNULL)
        
        supported_gpus = []
        unsupported_gpus = []
        
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                idx, name, cap = parts[0], parts[1], parts[2]
                try:
                    cap_val = float(cap)
                    # PaddlePaddle 支援上限為 SM 9.0 (Hopper)
                    if cap_val <= 9.0:
                        supported_gpus.append((idx, name, cap))
                    else:
                        unsupported_gpus.append((idx, name, cap))
                except ValueError:
                    pass

        # 若存在不相容之新架構顯卡 (例如 Blackwell SM 12.0)
        if unsupported_gpus:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            if supported_gpus:
                chosen_devices = ",".join(g[0] for g in supported_gpus)
                os.environ["CUDA_VISIBLE_DEVICES"] = chosen_devices
                print(f"\033[93m[GPU Config] 偵測到架構超過 Paddle 支援之顯卡: {[g[1] for g in unsupported_gpus]} (SM > 9.0)\033[0m")
                print(f"\033[92m[GPU Config] 自動將 Paddle 切換至相容顯卡: CUDA_VISIBLE_DEVICES={chosen_devices} ({supported_gpus[0][1]})\033[0m")
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                print("\033[93m[GPU Config] 系統中無相容之 Paddle GPU，已自動降級為 CPU 模式。\033[0m")
    except Exception:
        # 若無 nvidia-smi 或執行失敗，不中斷主流程
        pass

