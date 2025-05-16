from ultralytics import YOLO
import cv2

model = YOLO("yolo12n.pt") 

results = model.predict("images/cars.jpg", conf=0.8)
annotated_image = results[0].plot()
result = results[0]
print("===========================")
for box in result.boxes:
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    class_id = int(box.cls[0].item())
    conf = box.conf[0].item()
    conf = round(conf*100, 2)
    print("分類索引:", class_id)
    print("分類名稱:", model.names[class_id])
    print("邊界框座標:", cords)
    print("可能性:", conf, "%")
    print("------------------------")

cv2.imshow("Detected Objects", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

