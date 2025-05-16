# 定義分類串列
class_list = ["Millennium Falcon", "Tie Fighter"]
# 定義要寫入的檔案路徑
file_path = "YOLO_labelImg/data/predefined_classes.txt"

# 將串列的每個元素寫入檔案，每一行一個元素
with open(file_path, "w") as file:
    for item in class_list:
        file.write(f"{item}\n")

print(f"成功寫入 {len(class_list)} 分類到檔案 {file_path}")
