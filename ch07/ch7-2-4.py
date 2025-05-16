from ultralytics import YOLO
import cv2

model = YOLO("yolo12n.pt")

cap = cv2.VideoCapture("media/highway.mp4")
while True:
    success, frame  = cap.read()
    if not success:
        break
    results = model(frame)
    annotated_image = results[0].plot()
               
    cv2.imshow("Detected Objects", annotated_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()   
cv2.destroyAllWindows()