from cvzone3d.FaceMeshModule import FaceMeshDetector
import cv2
import numpy as np

img = cv2.imread("images/face.jpg")
detector = FaceMeshDetector(maxFaces=2)
img, faces = detector.findFaceMesh(img, draw=False)
if faces:
    face_lips = detector.getFacePart(img, 0, "LIPS")
    height_lips, width_lips = detector.getFacePartSize(face_lips)
    print(width_lips, height_lips)
    face_oval = detector.getFacePart(img, 0, "FACE_OVAL")
    height_oval, width_oval = detector.getFacePartSize(face_oval)
    print(width_oval, height_oval)
    if (height_lips/height_oval)*100 > 15:
        msg = 'Mouth OPEN'
    else:
        msg = 'Mouth CLOSE'
    points = np.array(face_lips, np.int32)
    cv2.polylines(img, [points], True, (0, 255, 255), 1) 
    points = np.array(face_oval, np.int32)
    cv2.polylines(img, [points], True, (0, 0, 255), 1) 
    cv2.putText(img, msg, (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
cv2.imshow("Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 