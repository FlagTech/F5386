from cvzone3d.HandTrackingModule import HandDetector
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(detectionCon=0.5, maxHands=2)
while cap.isOpened():
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand1 = hands[0]
        centerPoint1 = hand1["center"]
        cv2.circle(img, centerPoint1, 10, (0, 255, 255),
                   cv2.FILLED)
        if len(hands) == 2:
            hand2 = hands[1]
            centerPoint2 = hand2["center"]
            cv2.circle(img, centerPoint2, 10, (0, 255, 255),
                       cv2.FILLED)
            
    cv2.imshow("Hands", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
cap.release()
cv2.destroyAllWindows()

