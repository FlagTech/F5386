import face_recognition
import cv2

img = cv2.imread("images/mary.jpg")
known_encoding = face_recognition.face_encodings(img)[0]
new_img = cv2.imread("images/mary2.jpg")
new_encoding = face_recognition.face_encodings(new_img)[0]

result = face_recognition.compare_faces([known_encoding],
                                new_encoding, tolerance=0.6)
print(result)
