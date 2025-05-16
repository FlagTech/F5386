import dlib
import cv2

img = cv2.imread("images/faces2.jpg")
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 載入Dlib預訓練模型
cnn_face_detector = dlib.cnn_face_detection_model_v1(
                    "models/mmod_human_face_detector.dat")
# 執行影像的人臉偵測
results = cnn_face_detector(imgRGB, 0)
height, width = img.shape[:2]
for face in results:
    bbox = face.rect
    x1 = bbox.left()   # 取得邊界框的座標
    y1 = bbox.top()
    x2 = bbox.right()
    y2 = bbox.bottom()
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow("Detected Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
