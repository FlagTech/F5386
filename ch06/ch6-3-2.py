import dlib
import cv2

img = cv2.imread("images/faces.jpg")
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 載入預訓練 HoG 人臉偵測器
hog_face_detector = dlib.get_frontal_face_detector()
# 執行影像的人臉偵測
results = hog_face_detector(imgRGB, 0)
for bbox in results:
    x1 = bbox.left()    # 左上角x坐標
    y1 = bbox.top()     # 左上角y坐標
    x2 = bbox.right()   # 右下角x坐標
    y2 = bbox.bottom()  # 右下角y坐標
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
cv2.imshow("Detected Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
