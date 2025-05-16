from mediapipe import solutions  # 匯入 Mediapipe 的 solutions 模組
from mediapipe.framework.formats import landmark_pb2  # 處理關鍵點座標格式
import numpy as np  # 用於數值運算
import mediapipe as mp  # MediaPipe 核心函式庫
from mediapipe.tasks import python  # MediaPipe 任務模組
from mediapipe.tasks.python import vision  # MediaPipe 視覺模組
import cv2  # OpenCV 用於影像處理

# 設定模型路徑和偵測參數，允許最多 2 隻手
base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')  # 設定模型路徑
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)  # 設定偵測參數，最多允許2隻手
detector = vision.HandLandmarker.create_from_options(options)  # 建立手部標記偵測器
# 載入輸入影像
image = mp.Image.create_from_file("images/hands.jpg")  # 從檔案讀取影像並轉換為 Mediapipe 影像格式
# 偵測影像中的手部標記點
detection_result = detector.detect(image)  # 使用偵測器對影像進行手部標記偵測
# 處理偵測結果
if len(detection_result.hand_landmarks) > 0:  # 如果偵測到手部標記點
    annotated_image = np.copy(image.numpy_view())  # 建立影像副本
    for hand_landmarks in detection_result.hand_landmarks:  # 遍歷所有偵測到的手部標記點
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()  # 建立 Mediapipe 格式的標記點資料結構
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])  # 填充標記點資料
        solutions.drawing_utils.draw_landmarks(
            annotated_image,  # 要繪製的影像
            hand_landmarks_proto,  # 手部標記點資料
            solutions.hands.HAND_CONNECTIONS,  # 預設的手部標記點連線規則
            solutions.drawing_styles.get_default_hand_landmarks_style()  # 預設的標記點樣式
        )  # 使用 Mediapipe 的繪圖工具繪製手部標記點與連接線

cv2.imshow("MediaPipe Hands", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))  # 使用 OpenCV 顯示影像，轉換為 BGR 色彩模式
cv2.waitKey(0)  # 等待使用者按下任意鍵後關閉視窗
cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗
