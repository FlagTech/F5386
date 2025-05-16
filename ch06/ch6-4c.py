import face_recognition
import cv2

img = cv2.imread("images/mary.jpg")
known1_encoding = face_recognition.face_encodings(img)[0]
img = cv2.imread("images/jane.jpg")
known2_encoding = face_recognition.face_encodings(img)[0]
new_img = cv2.imread("images/mary2.jpg")
new_encoding = face_recognition.face_encodings(new_img)[0]

know_encodings = [known1_encoding, known2_encoding]
distance = face_recognition.face_distance(know_encodings,
                                          new_encoding)
print(distance)
