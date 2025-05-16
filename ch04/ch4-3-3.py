from cvzone3d.FaceDetectionModule import FaceDetector
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = FaceDetector()
while cap.isOpened():
    success, img = cap.read()
    img, faces = detector.findFaces(img)
    if faces:
        center = faces[0]["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
        
    cv2.imshow("Faces", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
