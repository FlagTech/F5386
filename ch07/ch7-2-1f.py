from ultralytics import YOLO
import cv2

model = YOLO("yolo12n.pt")  

image_paths = ["images/road.png", "images/persons.jpg"]
results = model.predict(image_paths)
annotated_image = results[0].plot()
cv2.imshow("Road", annotated_image)
annotated_image = results[1].plot()
cv2.imshow("Persons", annotated_image)    
cv2.waitKey(0)
cv2.destroyAllWindows()
