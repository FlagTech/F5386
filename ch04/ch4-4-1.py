from cvzone3d.FaceMeshModule import FaceMeshDetector
import cv2

img = cv2.imread("images/face4.jpg")
detector = FaceMeshDetector(maxFaces=2, minDetectionCon=0.5,
                            minTrackCon=0.5)
img, faces = detector.findFaceMesh(img)
if faces:
   print(len(faces[0]))
   
cv2.imshow("Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
