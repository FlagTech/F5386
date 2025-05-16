from ultralytics import YOLO
import random
import cv2
import numpy as np

random.seed(11)
model = YOLO("yolo11n-seg.pt")
img = cv2.imread("images/dog_person.jpg")
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
results = model.predict(img)
colors = [random.choices(range(256), k=3) for _ in classes_ids]

result = results[0]
for mask, box in zip(result.masks.xy, result.boxes):
    points = np.int32([mask])
    color_number = classes_ids.index(int(box.cls[0]))
    # 繪出多邊形外框
    cv2.polylines(img, points, isClosed=True,
                  color=colors[color_number], thickness=2)

cv2.imshow("Segmented Objects", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
