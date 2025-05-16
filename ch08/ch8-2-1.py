from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-seg.pt")

results = model("images/dog_person.jpg")
annotated_image = results[0].plot()

cv2.imshow("Segmented Objects", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
