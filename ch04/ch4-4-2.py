from cvzone3d.FaceMeshModule import FaceMeshDetector
import cv2
import numpy as np

img = cv2.imread("images/face.jpg")
detector = FaceMeshDetector(maxFaces=2)
img, faces = detector.findFaceMesh(img, draw=False)
if faces:
    face_oval = detector.getFacePart(img, 0, "FACE_OVAL")
    points = np.array(face_oval, np.int32)
    cv2.polylines(img, [points], True, (0, 0, 255), 1)  
    face_leye = detector.getFacePart(img, 0, "LEFT_EYE")
    points = np.array(face_leye, np.int32)
    cv2.polylines(img, [points], True, (0, 255, 255), 1)     
  
cv2.imshow("Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
