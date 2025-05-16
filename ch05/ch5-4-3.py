from cvzone3d.PoseModule import PoseDetector
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = PoseDetector()
while cap.isOpened():
    success, img = cap.read()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, draw=False)
    if lmList:
        x1, y1, w, h = bboxInfo["bbox"]
        cv2.rectangle(img, (x1, y1),
                           (x1 + w, y1 + h),
                           (255, 0, 255), 2)
        center = bboxInfo["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Pose", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
