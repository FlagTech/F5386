from ultralytics import YOLO
import cv2

model = YOLO("yolo12n.pt")  

image_paths = ["images/road.png", "images/persons.jpg"]
results = model(image_paths)
for result in results:
    annotated_image = result.plot()
    cv2.imshow(result.path, annotated_image)
    
cv2.waitKey(0)
cv2.destroyAllWindows()
