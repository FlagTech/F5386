from ultralytics import YOLO
import cv2

model = YOLO("yolo12n.pt") 

results = model.predict("images/cars.jpg", conf=0.8)
annotated_image = results[0].plot()
print("===========================")
result = results[0]
print(result.boxes.data)

cv2.imshow("Detected Objects", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

