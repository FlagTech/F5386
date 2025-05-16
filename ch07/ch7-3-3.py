from ultralytics import YOLO
import cv2, imutils

model = YOLO("yolo12n.pt")

cap = cv2.VideoCapture("media/highway.mp4")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = imutils.resize(frame, width=640)
    results = model.track(frame, persist=True,
                          verbose=False, classes=[2])
    if results[0].boxes.id is None:
        continue
    print("追蹤數:", len(results[0].boxes.data))
    print("第1個:", results[0].boxes.data[0])
    # tensor([570.8726, 289.0330, 594.5842, 318.3900,   1.0000,   0.6356,   2.0000])
    #          x1,        y1,       x2       ,y2,       track_id,  conf,   class_id     
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Object Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()