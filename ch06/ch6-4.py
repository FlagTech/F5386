import face_recognition
import cv2

img = cv2.imread("images/faces2.jpg")
faces = face_recognition.face_locations(img,
                         number_of_times_to_upsample=1,
                         model="hog")
print("臉數=", len(faces))
for face in faces:
    top, right, bottom, left = face
    cv2.rectangle(img, (left, top), (right, bottom),
                                    (0, 0, 255), 3)

cv2.imshow("Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
