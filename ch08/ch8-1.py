from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-cls.pt")

image = cv2.imread("images/cat.jpg")
results = model(image)
annotated_image = results[0].plot()

cv2.imshow("Image Classification", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


