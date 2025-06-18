from ultralytics import YOLO
import cv2
import os

output_path = "output"
os.makedirs(output_path, exist_ok=True)   # 確保輸出目錄存在
model = YOLO("yolo11n-seg.pt")

image_path = "images/dogs.jpg"
results = model(image_path)
annotated_image = results[0].plot()
cv2.imshow("Segmented Objects", annotated_image)

result = results[0]
img = cv2.imread(image_path)
for idx,box in enumerate(result.boxes.xyxy):
    # box.cpu()是將box資料從GPU(若用CUDA）移到CPU，因為NumPy不支援GPU張量。
    x1,y1,x2,y2 = box.cpu().numpy().astype(int)
    cv2.imwrite(f"{output_path}/image{idx}.png",
                img[y1:y2,x1:x2,:])

cv2.waitKey(0)
cv2.destroyAllWindows()
