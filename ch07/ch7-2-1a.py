from ultralytics import YOLO
import cv2

model = YOLO("yolo12n.pt")  

results = model("images/road.png", classes=[0])
#results = model("images/road.png", classes=[0, 2])
annotated_image = results[0].plot()

cv2.imshow("Detected Objects", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
