from ultralytics import YOLO
import cv2

model = YOLO("yolo12n.pt")

cap = cv2.VideoCapture("media/highway.mp4")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.resize(frame, (640, 480))
    results = model.track(frame, persist=True,
                          verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Object Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()