from ultralytics import YOLO
import cv2

model = YOLO("yolo12n.pt")  

image = cv2.imread("images/road.png")
results = model.predict(image, conf=0.84)
annotated_image = results[0].plot()

cv2.imshow("Detected Objects", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
