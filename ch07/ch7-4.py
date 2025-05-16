from ultralytics import YOLO
import cv2

model = YOLO("yolo12n.pt")

cap = cv2.VideoCapture("media/demo.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    results = model(frame)
    # 計算檢測到的 "person" 數量
    person_count = sum(1 for box in results[0].boxes
                       if model.names[int(box.cls[0])] == "person")
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, "Count: " + str(person_count),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("YOLO Person Counting", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
