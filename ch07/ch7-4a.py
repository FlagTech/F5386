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
    counter_list = []
    for box in results[0].boxes: 
        class_index = int(box.cls[0])
        class_name = model.names[class_index]
        if class_name == "person":
            counter_list.append(1)
    person_count = sum(counter_list)

    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, "Count: " + str(person_count),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("YOLO Person Counting", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
