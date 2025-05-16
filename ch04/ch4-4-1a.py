from cvzone3d.FaceMeshModule import FaceMeshDetector
import cv2

img = cv2.imread("images/face4.jpg")
detector = FaceMeshDetector(maxFaces=2)
img, faces = detector.findFaceMesh(img, draw=False)
if faces:
   print(len(faces[0]))
   for point in faces[0]:
       cv2.circle(img, point, 1, (255, 0, 255), cv2.FILLED)    
   
cv2.imshow("Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
