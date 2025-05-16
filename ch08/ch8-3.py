from ultralytics import YOLO
import cv2

model = YOLO("yolo11m-pose.pt") 

image_path = "images/pose.jpg"
results = model(image_path)
annotated_image = results[0].plot()

cv2.imshow("Pose Estimation", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()