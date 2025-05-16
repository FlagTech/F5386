import cv2
import dlib

image = cv2.imread("images/face4.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hog_face_detector = dlib.get_frontal_face_detector()
# 下載預訓練模型檔
shape_predictor = dlib.shape_predictor(
                  "models/shape_predictor_68_face_landmarks.dat")
faces = hog_face_detector(image_gray, 1)
for face in faces:
    # 取得人臉 68 個特徵點
    landmarks = shape_predictor(image_gray, face)
    # 定義連接主要面部特徵的區域
    key_regions = [
        range(0, 17),   # 下巴線
        range(17, 22),  # 左眉毛
        range(22, 27),  # 右眉毛
        range(27, 36),  # 鼻子
        [31, 27], [30, 35],
        range(36, 42),  # 左眼
        range(42, 48),  # 右眼
        range(48, 60),  # 外唇
        range(60, 68)   # 內唇
    ]
    # 繪製特徵點
    for region in key_regions:
        prev_x, prev_y = None, None
        for n in region:
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            if prev_x is not None and prev_y is not None:
                cv2.line(image, (prev_x, prev_y), (x, y),
                         (0, 255, 0), 1)
            prev_x, prev_y = x, y
        # 連接區域的第一點和最後一點
        if len(region) >= 3:
            x0 = landmarks.part(region[0]).x
            y0 = landmarks.part(region[0]).y
            cv2.line(image, (x, y), (x0, y0), (0, 255, 0), 1)
# 顯示結果
cv2.imshow("Face Mesh", image)
cv2.waitKey(0)
cv2.destroyAllWindows()