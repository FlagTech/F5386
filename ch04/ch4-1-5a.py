import mediapipe as mp  # 匯入 Mediapipe 核心函式庫
from mediapipe import solutions  # 匯入 Mediapipe 的 solutions 模組
from mediapipe.framework.formats import landmark_pb2  # 匯入 Mediapipe 的關鍵點格式
from mediapipe.tasks import python  # 匯入 Mediapipe 的 Python 任務 API
from mediapipe.tasks.python import vision  # 匯入 Mediapipe 的視覺處理模組
import numpy as np  # 匯入 NumPy 進行影像處理
import cv2  # 匯入 OpenCV 進行影像顯示

# 初始化 Mediapipe 姿勢偵測器
base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_full.task')  # 設定模型路徑
options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True)  # 設定偵測參數
detector = vision.PoseLandmarker.create_from_options(options)  # 建立姿勢偵測器
# 讀取輸入影像
mp_image = mp.Image.create_from_file("images/pose.jpg")  # 從檔案讀取影像並轉換為 Mediapipe 影像格式
# 進行姿勢偵測
detection_result = detector.detect(mp_image)  # 使用偵測器對影像進行分析
# 處理偵測結果
if len(detection_result.pose_landmarks) > 0:  # 如果偵測到姿勢關鍵點
    print("偵測到的姿勢數量:", len(detection_result.pose_landmarks))  # 印出偵測到的姿勢數量
    annotated_image = np.copy(mp_image.numpy_view())  # 建立影像副本
    for pose_landmarks in detection_result.pose_landmarks:  # 遍歷所有偵測到的姿勢關鍵點
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()  # 建立 Mediapipe 格式的關鍵點資料結構
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])  # 填充關鍵點資料
        solutions.drawing_utils.draw_landmarks(
            annotated_image,  # 要繪製的影像
            pose_landmarks_proto,  # 姿勢關鍵點資料
            solutions.pose.POSE_CONNECTIONS,  # 預設的姿勢關鍵點連線規則
            solutions.drawing_styles.get_default_pose_landmarks_style()  # 預設的關鍵點樣式
        )  # 使用 Mediapipe 的繪圖工具繪製姿勢關鍵點與連接線

cv2.imshow('Image', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))  # 使用 OpenCV 顯示影像，轉換為 BGR 色彩模式
cv2.waitKey(0)  # 等待使用者按下任意鍵後關閉視窗
cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗
