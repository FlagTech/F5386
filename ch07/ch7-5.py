from collections import defaultdict
from ultralytics import YOLO
import cv2
import numpy as np
import imutils

model = YOLO("yolo12n.pt")
cap = cv2.VideoCapture("media/highway.mp4")
track_history = defaultdict(lambda: [])

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = imutils.resize(frame, width=640)
    results = model.track(frame, persist=True,
                          verbose=False, classes=[2])
    if results[0].boxes.id is None:
        continue
    boxes = results[0].boxes.xywh
    track_ids = results[0].boxes.id.int().tolist()
    annotated_frame = results[0].plot()
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # 存入中心點
        if len(track) > 90:                 # 只保留最後 90 frames的追蹤記錄
            track.pop(0)
        # 繪出追蹤線
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False,
                      color=(255, 255, 0), thickness=5)
    
    cv2.imshow("YOLO Object Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print(track_history[2])  # track_id 2的追蹤記錄, 即90個frames的中心點座標
print(len(track_history[2]))
cap.release()
cv2.destroyAllWindows()