import cv2
import os

image_folder = "data/train/images"    # 圖檔路徑
label_folder = "data/train/labels"    # 標註檔路徑
class_names = ['hamster']             # 分類名稱

# 讀取目錄下的所有圖檔和排序
image_files = sorted(os.listdir(image_folder))
# 走訪每一個圖檔
for image_file in image_files:
    if image_file.endswith(".jpg") or image_file.endswith(".png"):
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder,
                         os.path.splitext(image_file)[0] + ".txt")
        # 讀取圖檔
        img = cv2.imread(image_path)
        print(image_path)
        if os.path.exists(label_path):        
          # 讀取標註文檔
          with open(label_path, "r") as file:
            for line in file.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                # 計算方框座標
                img_height, img_width = img.shape[:2]
                x1 = int((x_center - width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                x2 = int((x_center + width / 2) * img_width)
                y2 = int((y_center + height / 2) * img_height)
                # 繪製方框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 設定文字內容
                text = f'{class_names[class_id]}'
                # 計算文字大小
                (text_width, text_height), baseline = cv2.getTextSize(text,
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                # 計算文字背景框的左上角和右下角座標
                top_left = (x1, y1 - text_height - baseline)
                bottom_right = (x1 + text_width, y1)
                # 繪製文字背景框
                cv2.rectangle(img, top_left, bottom_right,
                              (255, 255, 255), cv2.FILLED)
                # 繪製文字
                cv2.putText(img, text, (x1, y1 - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # 顯示影像
        cv2.imshow("Labeled Image", img)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # ESC 鍵
            break

cv2.destroyAllWindows()
