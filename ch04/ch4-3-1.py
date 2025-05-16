from cvzone3d.FaceDetectionModule import FaceDetector
import cv2

img = cv2.imread("images/faces.jpg")
detector = FaceDetector(minDetectionCon=0.5)
img, faces = detector.findFaces(img)
if faces:
    print("偵測到人臉數:", len(faces))
cv2.imshow("Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
