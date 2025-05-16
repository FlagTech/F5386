from cvzone3d.FaceDetectionModule import FaceDetector
import cv2

img = cv2.imread("images/faces.jpg")
detector = FaceDetector()
img, faces = detector.findFaces(img)
if faces:
    print("偵測到人臉數:", len(faces))
    face = faces[0]
    print("id:", face["id"])
    print("bbox:", face["bbox"])
    print("score:", face["score"])
    print("center:", face["center"])
    