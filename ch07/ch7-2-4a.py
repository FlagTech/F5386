from ultralytics import YOLO
import cv2

model = YOLO("yolo12n.pt")

cap = cv2.VideoCapture("media/highway.mp4")
while True:
    success, frame  = cap.read()
    if not success:
        break
    results = model(frame)
    if results:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            conf = round((box.conf[0].item()*100), 2)
            cls = box.cls[0].item()
            class_name = results[0].names[int(cls)]            
            # 繪製矩形邊界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 寫上分類標籤和可能性
            label = class_name + " " + str(conf) + "%"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    cv2.imshow("Detected Objects", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
