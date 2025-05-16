import cv2
import time
import torch
from ultralytics import YOLO

# 初始化網絡攝影機或開啟影片檔
cap = cv2.VideoCapture("media/emotion2.mp4")
smile_duration = 1  # 微笑持續時間, 秒
start_time = None
# 載入YOLO情緒辨識模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("face_emotion.pt")
model.to(device)
# 分類名稱的索引，索引值 3 是 happy
emotion_labels = {3: "happy"}
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 保留原始影格供儲存使用
    original_frame = frame.copy()
    # 使用YOLO模型偵測情緒
    results = model.predict(source=frame, conf=0.5,                             
                            device=device, verbose=False)    
    emotion = None
    for result in results:
        for box in result.boxes:
            # 取得情緒分類索引
            class_id = int(box.cls[0].item())
            # 取得信心指數
            confidence = box.conf[0].item()            
            # 繪製邊界框和標籤
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)            
            # 檢查是否為happy情緒
            if class_id == 3:  # happy的索引是3
                emotion = "happy"
                label = f"Happy: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)    
    # 偵測微笑
    if emotion == 'happy':
        if start_time is None:
            start_time = time.time()
        elif time.time() - start_time >= smile_duration:
            # 微笑持續超過設定時間, 拍照並儲存
            filename = f"smile_{int(time.time())}.jpg"
            cv2.imwrite(filename, original_frame)
            print(f"拍照並儲存為 {filename}")
            start_time = None  # 重置計時器
    else:
        start_time = None  # 重置計時器    
    # 顯示影格
    cv2.imshow("YOLO Emotion", frame)    
    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 釋放資源
cap.release()
cv2.destroyAllWindows()