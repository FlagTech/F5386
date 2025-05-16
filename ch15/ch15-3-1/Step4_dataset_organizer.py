import os
import shutil

train_dir = "data/train"   # 訓練資料集目錄 
val_dir = "data/val"       # 驗證資料集目錄

def organize_files(directory):
    # 建立 images 和 labels 子目錄
    images_dir = os.path.join(directory, "images")
    labels_dir = os.path.join(directory, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    # 取得目錄下所有的檔案
    files = os.listdir(directory)
    # 走訪所有檔案來進行分類
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            shutil.move(os.path.join(directory, file),
                        os.path.join(images_dir, file))
        elif file.endswith(".txt"):
            shutil.move(os.path.join(directory, file),
                        os.path.join(labels_dir, file))
    
    print(f"目錄的檔案已經分配至 '{images_dir}' 和 '{labels_dir}' 目錄")

# 組織YOLO訓練資料
organize_files(train_dir)
organize_files(val_dir)