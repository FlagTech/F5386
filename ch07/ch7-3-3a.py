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
    print("===============================")
    print("追蹤數:", len(results[0].boxes.data))
    print("第1個:", results[0].boxes.data[0])
    boxes = results[0].boxes.xywh.tolist()
    print(boxes[0])
    track_ids = results[0].boxes.id.int().tolist()
    print(track_ids[0])
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Object Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()