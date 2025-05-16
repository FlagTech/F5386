from cvzone3d.FaceDetectionModule import FaceDetector
import cv2

img = cv2.imread("images/face.jpg")
detector = FaceDetector()
img, faces = detector.findFaces(img)
if faces:
    for face in faces:
        x, y, w, h = face["bbox"]
        face = img[y:y+h, x:x+w]
        cv2.imshow("Non Padded", face)
        padding = 20
        padded_face = img[y-padding:y+h+padding,x-padding:x+w+padding] 
        cv2.imshow("Padded", padded_face)
        cv2.waitKey(0)

cv2.destroyAllWindows()
