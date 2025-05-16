from ultralytics import YOLO
import cv2
import imutils

model = YOLO("yolo11m-pose.pt") 

cap = cv2.VideoCapture("media/demo.mp4")
while True:
    success, frame  = cap.read()
    if not success:
        break
    frame = imutils.resize(frame, width=640)
    results = model(frame)
    annotated_image = results[0].plot()
               
    cv2.imshow("Pose Estimation", annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()   
cv2.destroyAllWindows()
