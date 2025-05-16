import os
import shutil
import random

input_dir = "frames"  # 請將此處替換為您的輸入目錄
train_dir = "train"   # 訓練資料集目錄
val_dir = "val"       # 驗證資料集目錄
split_ratio = 0.8     # 訓練與驗證的分割比例
img_type = ".jpg"     # 圖檔類型
operation = "copy"    # 選擇 "move" 搬移檔案 或 "copy" 複製檔案

def split_dataset(input_dir, train_dir, val_dir,
                  split_ratio=0.8, img_type='.jpg',
                  operation='move'):
    # 確保輸出目錄存在
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    # 取得所有圖檔和對應的 .txt 檔
    files = [f for f in os.listdir(input_dir) if f.endswith(img_type)]
    # 隨機打亂檔案順序
    random.shuffle(files)
    # 計算分割點
    split_point = int(len(files) * split_ratio)
    # 分割資料集
    train_files = files[:split_point]
    val_files = files[split_point:]    
    # 依據 operation 參數選擇是搬移還是複製檔案
    def transfer_files(files, source_dir, target_dir):
        for f in files:
            src = os.path.join(source_dir, f)
            dst = os.path.join(target_dir, f)
            if operation == "move":
                shutil.move(src, dst)
            elif operation == "copy":
                shutil.copy(src, dst)
            txt_file = f.replace(img_type, ".txt")
            src_txt = os.path.join(source_dir, txt_file)
            dst_txt = os.path.join(target_dir, txt_file)
            if os.path.exists(src_txt):
                if operation == "move":
                    shutil.move(src_txt, dst_txt)
                elif operation == "copy":
                    shutil.copy(src_txt, dst_txt)
    
    # 執行檔案轉移
    transfer_files(train_files, input_dir, train_dir)
    transfer_files(val_files, input_dir, val_dir)
    
    print(f"使用 {operation} 操作處理 {input_dir} 目錄的資料集")
    print(f"資料集分割成 {train_dir} 和 {val_dir} 目錄的資料集")
    print(f"使用的分割比例: {split_ratio}")

split_dataset(input_dir, train_dir, val_dir,
              split_ratio, img_type, operation)

