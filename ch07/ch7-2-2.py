from ultralytics import YOLO
import cv2

model = YOLO("yolo12n.pt") 
print(model.names)   # 80種分類名稱
results = model("images/cars.jpg")
annotated_image = results[0].plot()
print("===========================")
print(results[0])

cv2.imshow("Detected Objects", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

