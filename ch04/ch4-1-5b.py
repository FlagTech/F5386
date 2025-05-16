import mediapipe as mp
from mediapipe import solutions  # 匯入 Mediapipe 的 solutions 模組
from mediapipe.framework.formats import landmark_pb2  # 匯入 Mediapipe 的關鍵點格式
from mediapipe.tasks import python  # 匯入 Mediapipe 的 Python 任務 API
from mediapipe.tasks.python import vision  # 匯入 Mediapipe 的視覺處理模組
import numpy as np  # 匯入 NumPy 進行影像處理
import cv2  # 匯入 OpenCV 進行影像讀取與顯示

# === 函數: 在影像上繪製人體姿勢關鍵點 ===
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks  # 取得所有偵測到的人體姿勢關鍵點
    annotated_image = np.copy(rgb_image)  # 建立影像副本，避免修改原始影像
    print("偵測到的姿勢數量:", len(pose_landmarks_list))  # 印出偵測到的姿勢數量    
    # 逐一處理偵測到的每個人體姿勢
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]  # 取得第 idx 個人體姿勢的關鍵點
        
        # 將關鍵點資料轉換為 Mediapipe 格式
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        print("第", idx + 1, "組姿勢關鍵點:", pose_landmarks)  # 印出該人體姿勢的關鍵點資訊        
        # 使用 Mediapipe 的繪圖工具繪製姿勢關鍵點與連接線
        solutions.drawing_utils.draw_landmarks(
            annotated_image,  # 要繪製的影像
            pose_landmarks_proto,  # 姿勢關鍵點資料
            solutions.pose.POSE_CONNECTIONS,  # 預設的姿勢關鍵點連線規則
            solutions.drawing_styles.get_default_pose_landmarks_style()  # 預設的關鍵點樣式
        )
    
    return annotated_image  # 回傳繪製完的影像

# === 初始化 Mediapipe 姿勢偵測器 ===
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

base_options = BaseOptions(model_asset_path='models/pose_landmarker_full.task')  # 指定模型
options = PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.VIDEO  # 設定為影片模式
)

# === 使用 OpenCV 讀取影片並進行姿勢偵測 ===
with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture("media/Squat_clip.mp4")  # 讀取影片
    frame_index = 0
    video_file_fps = 30  # 設定影片幀率    
    while True:
        success, img = cap.read()  # 讀取影片幀
        if not success:
            break        
        img = cv2.resize(img, (640, 480))  # 調整影像大小
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)  # 轉換為 Mediapipe 影像格式        
        # 計算當前幀的時間戳記（毫秒）
        frame_timestamp_ms = int(1000 * frame_index / video_file_fps)
        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)  # 進行姿勢偵測        
        # 繪製人體姿勢關鍵點
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)        
        cv2.imshow('Image', annotated_image)  # 顯示結果影像
        frame_index += 1        
        # 按 'q' 退出
        if cv2.waitKey(1) == ord('q'):
            break
    
cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗

