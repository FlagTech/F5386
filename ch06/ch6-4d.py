import face_recognition
import cv2
import numpy as np

img = cv2.imread("images/faces2.jpg")
faces = face_recognition.face_locations(img)
landmarks = face_recognition.face_landmarks(img, face_locations=faces)
features = ["chin",            # 下巴
            "left_eyebrow",    # 左眉
            "right_eyebrow",   # 右眉
            "nose_bridge",     # 鼻樑
            "nose_tip",        # 鼻尖
            "left_eye",        # 左眼
            "right_eye",       # 右眼
            "top_lip",         # 上嘴唇
            "bottom_lip"]      # 下嘴唇
for landmark in landmarks:
    for feature in features:
        points = np.array(landmark[feature], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(img, [points], False, (255, 255, 0), 2)        

cv2.imshow("Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
