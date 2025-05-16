import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2
import math

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 指定圖檔路徑
image_path = "images/pointing_up.jpg"
# 建立手勢辨識物件
base_options = python.BaseOptions(
               model_asset_path='models/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# 讀取輸入影像
image = mp.Image.create_from_file(image_path)
# 執行手勢辨識
recognition_result = recognizer.recognize(image)
# 取得辨識分數最高的手勢和手部關鍵點資訊
top_gesture = recognition_result.gestures[0][0]
multi_hand_landmarks = recognition_result.hand_landmarks

# 顯示影像並繪製結果
annotated_image = image.numpy_view().copy()  # 轉換成 NumPy 陣列
# 取得手勢分類名稱和信心指數的分數
title = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
# 繪製手部關鍵點和連接線
if multi_hand_landmarks:
    for hand_landmarks in multi_hand_landmarks:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z) for landmark in hand_landmarks
        ])
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
# 使用 OpenCV 顯示影像
image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
cv2.imshow(title, image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()