from mediapipe import solutions  # 匯入 Mediapipe 的 solutions 模組
from mediapipe.framework.formats import landmark_pb2  # 處理關鍵點座標格式
import numpy as np  # 用於數值運算
import mediapipe as mp  # MediaPipe 核心函式庫
from mediapipe.tasks import python  # MediaPipe 任務模組
from mediapipe.tasks.python import vision  # MediaPipe 視覺模組
import cv2  # OpenCV 用於影像處理

# 建立 MediaPipe 臉部偵測器 (FaceLandmarker)
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')  # 設定模型路徑
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,  # 啟用 Blendshapes 偵測
    output_facial_transformation_matrixes=True,  # 啟用 3D 變形矩陣輸出
    num_faces=1  # 限制偵測 1 張臉
)
detector = vision.FaceLandmarker.create_from_options(options)  # 初始化偵測器
# 讀取圖片並進行臉部偵測
image = mp.Image.create_from_file("images/face.jpg")  # 載入影像
detection_result = detector.detect(image)  # 執行偵測
# 取得臉部特徵點資料
face_landmarks_list = detection_result.face_landmarks  # 取得臉部特徵點資料
# 判斷是否有偵測到臉部
if face_landmarks_list:
    annotated_image = np.copy(image.numpy_view())  # 複製影像，避免修改原圖
    # 遍歷所有偵測到的臉部 (支援多張臉)
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]  # 取得第 idx 張臉的特徵點
        # 建立 MediaPipe 標準的 Landmark 格式
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
        # 在影像上繪製臉部網格 (Tessellation)
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        )
        # 在影像上繪製臉部輪廓 (Contours)
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
        )
        # 在影像上繪製虹膜 (Irises)
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
        )

# 使用 OpenCV 顯示標記後的影像
cv2.imshow("MediaPipe FaceMesh", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)  # 等待按鍵以關閉視窗
cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗
