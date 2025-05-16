import cv2
import dlib
import numpy as np
import math

image_path = "images/mary.jpg"

# 載入 Dlib 人臉偵測器和特徵點模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
            "models/shape_predictor_68_face_landmarks.dat")
# 替嘴唇塗上口紅
def apply_lipstick(image, landmarks):
    # 嘴唇的特徵點索引（基於68個特徵點模型）
    lips_points = list(range(48, 61))  # 包括上下唇
    hull = cv2.convexHull(np.array(
        [landmarks[i] for i in lips_points]))
    # 建立一個遮罩
    mask = np.zeros_like(image[:, :, 0])
    cv2.fillConvexPoly(mask, hull, 255)
    # 選擇紅色的口紅顏色
    lipstick_color = np.zeros_like(image)
    lipstick_color[:, :] = (0, 0, 255)  # BGR 的紅色
    # 塗抹口紅
    output = image.copy()
    lipstick_applied = cv2.addWeighted(output, 0.5,
                                  lipstick_color, 0.5, 0)
    output[mask == 255] = lipstick_applied[mask == 255]

    return output

# 讀取圖檔
image = cv2.imread(image_path)
# 轉換成灰階影像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 偵測人臉
faces = detector(gray)
for face in faces:
    # 取得臉部特徵點
    shape = predictor(gray, face)
    landmarks = [(p.x, p.y) for p in shape.parts()]
    # 塗上口紅
    image = apply_lipstick(image, landmarks)

# 顯示結果
cv2.imshow("Lipstick", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
