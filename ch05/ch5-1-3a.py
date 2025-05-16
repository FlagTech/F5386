from cvzone3d.HandTrackingModule import HandDetector
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(detectionCon=0.5, maxHands=1)
while cap.isOpened():
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        bbox = hand["bbox"]
        print(bbox)
        fingers = detector.fingersUp(hand)
        totalFingers = fingers.count(1)
        msg = "Fingers:" + str(totalFingers)
        cv2.putText(img, msg, (bbox[0]+100, bbox[1]-30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Hand", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
cap.release()
cv2.destroyAllWindows()