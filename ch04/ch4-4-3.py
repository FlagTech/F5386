from cvzone3d.FaceMeshModule import FaceMeshDetector
import cv2

img = cv2.imread("images/face4.jpg")
detector = FaceMeshDetector(maxFaces=2)
img, faces = detector.findFaceMesh(img, draw=False)
if faces:
   face = faces[0]    
   leftEyePoint = face[386]
   rightEyePoint = face[159]
   cv2.circle(img, leftEyePoint, 5, (255, 0, 255), cv2.FILLED)
   cv2.circle(img, rightEyePoint, 5, (0, 255, 0), cv2.FILLED)
   EyeDistance, info = detector.findDistance(leftEyePoint, rightEyePoint)
   text = "Distance : " + str(int(EyeDistance))
   cv2.putText(img, text, (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0 , 255, 0), 2)  
   
cv2.imshow("Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
