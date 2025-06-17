from ultralytics import YOLO
import cv2
import os
import numpy as np

output_path = "output2"
os.makedirs(output_path, exist_ok=True)   # 確保輸出目錄存在
model = YOLO("yolo11n-seg.pt")

image_path = "images/dogs.jpg"
results = model(image_path)
annotated_image = results[0].plot()
cv2.imshow("Segmented Objects", annotated_image)

result = results[0]
for idx,polygon in enumerate(result.masks.xy):
    polygon = polygon.astype(np.int32)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
    mask = np.zeros_like(img,dtype=np.int32)
    cv2.fillPoly(mask,[polygon],color=(255, 255, 255))
    x1,y1,x2,y2 = result.boxes.xyxy[idx].cpu().numpy().astype(int)
    img = cv2.bitwise_and(img, img, mask=mask[:,:,0].astype('uint8'))
    cv2.imwrite(f"{output_path}/image{idx}.png", img[y1:y2,x1:x2,:])

cv2.waitKey(0)
cv2.destroyAllWindows()
