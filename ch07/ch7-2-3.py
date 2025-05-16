from ultralytics import YOLO
import cv2

model = YOLO("yolo12n.pt") 

image_path = "images/road.png"
image = cv2.imread(image_path)
results = model(image)
if results:
    boxes = results[0].boxes
    for box in boxes:
        # 取得邊界框座標, 這是使用map()函數轉換成整數座標
        x1, y1, x2, y2 = map(int, box.xyxy[0])  
        conf = box.conf[0].item()  # 取得物體偵測的信心指數
        conf = round(conf*100, 2)
        class_id = int(box.cls[0])  # 物體的分類 ID, 因為使用int()轉換成整數, 可以不用.item()
        class_name = model.names[class_id]  # 獲取分類名稱

        cv2.rectangle(image, (x1, y1), (x2, y2),
                      (0, 255, 0), 2)  # 綠色邊界框
        # 黃色文字
        label = class_name + " " + str(conf) + "%"        
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 2)

cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
