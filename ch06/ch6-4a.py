import face_recognition
import cv2

img = cv2.imread("images/faces2.jpg")
faces = face_recognition.face_locations(img)
print("臉數=", len(faces))
img_encodings = face_recognition.face_encodings(img,
                known_face_locations=faces, num_jitters=10)
print("128維度的編碼=", len(img_encodings))
print(img_encodings[0])
