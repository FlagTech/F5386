from cvzone3d.FaceMeshModule import FaceMeshDetector
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = FaceMeshDetector(maxFaces=2)
while cap.isOpened():
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img)
    if faces:
        print(faces[0])
    cv2.imshow("Faces", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
