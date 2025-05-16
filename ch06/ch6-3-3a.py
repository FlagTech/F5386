import dlib
import cv2
import numpy as np

image = cv2.imread("images/Happy.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
            "models/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1(
          "models/dlib_face_recognition_resnet_model_v1.dat")
faces = detector(gray)
for face in faces:
    shape = predictor(gray, face)
    # 計算人臉128維特徵向量
    face_descriptor = facerec.compute_face_descriptor(image, shape)
    # 將特徵向量轉換為Numpy陣列
    face_descriptor_np = np.array(face_descriptor)
    print("128維特徵向量：\n", face_descriptor_np)
