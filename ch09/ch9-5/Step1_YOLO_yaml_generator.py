import os

data_dir = "data"      # 訓練資料所在的子目錄
classes = ['hamster']    # 分類名稱串列['hamster'], 或classes.txt路徑

# 判斷分類是否是串列, 如果不是, 就讀取檔案內容
if not isinstance(classes, list):
    # 分類檔案所在的子目錄
    classes_file = classes
    # 讀取類別名稱
    with open(classes_file, 'r') as file:
        classes = [line.strip() for line in file.readlines()]
# 判斷驗證資料集子目錄是val或valid
val_path = data_dir + "/valid/images" 
val_dir = "/val/images"
# 判斷vvalidb路徑是否存在, 否則就是bval
if os.path.exists(val_path):
    val_dir = "/valid/images"
# 判斷是否有測試資料集    
test_dataset = False    
test_path = data_dir + "/test/images"  # 測試資料集目錄
# 判斷 test 路徑是否存在
if os.path.exists(test_path):
    test_dataset = True   # 有測試資料集
# 建立YAML檔案的內容
if test_dataset:     # 有測試資料集
    # YAML 檔案內容
    data_yaml_content = f"""
train: {data_dir}/train/images
val: {data_dir}{val_dir}
test: {data_dir}/test/images

nc: {len(classes)}
names: {classes}

"""
else:                # 沒有測試資料集
    data_yaml_content = f"""
train: {data_dir}/train/images
val: {data_dir}{val_dir}

nc: {len(classes)}
names: {classes}

"""    
# YAML 檔案路徑
yaml_path = os.path.join(os.getcwd(), "data.yaml")
# 寫入 YAML 檔案
with open(yaml_path, "w") as file:
    file.write(data_yaml_content.strip())

print(f"YOLO 的 data.yaml 已經建立在 {yaml_path}")
