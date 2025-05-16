import cv2
import time
from deepface import DeepFace

# 初始化網絡攝影機或開啟影片檔
cap = cv2.VideoCapture("media/emotion1.mp4")
smile_duration = 1  # 微笑持續時間, 秒
start_time = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 保留原始影格供儲存使用
    original_frame = frame.copy()
    # 使用DeepFace偵測情緒
    result = DeepFace.analyze(frame, actions=['emotion'],
                              enforce_detection=False)
    # 取得情緒和臉部座標
    emotion = result[0]['dominant_emotion']
    face_region = result[0]['region']
    # 取得臉部區域座標
    x, y = face_region['x'], face_region['y']
    w, h = face_region['w'], face_region['h']
    # 在臉部周圍繪製邊界框
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)    
    # 在邊界框上方顯示情緒文字
    emotion_text = f"Emotion: {emotion}"
    cv2.putText(frame, emotion_text, (x, y-10),
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
    # 顯示偵測到的所有臉部情緒
    all_emotions = result[0]['emotion']
    y_position = 30
    for emotion_type, score in all_emotions.items():
        emotion_info = f"{emotion_type}: {score:.2f}%"
        cv2.putText(frame, emotion_info, (10, y_position), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        y_position += 20    
    # 顯示影格
    cv2.imshow("Deepface Emotion", frame)    
    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 釋放資源
cap.release()
cv2.destroyAllWindows()