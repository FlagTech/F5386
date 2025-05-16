from cvzone3d.HandTrackingModule import HandDetector
import cv2

img = cv2.imread("images/hand2.jpg")
detector = HandDetector(detectionCon=0.5, maxHands=1)
hands, img = detector.findHands(img)
if hands:
    hand = hands[0]
    bbox = hand["bbox"]        
    fingers = detector.fingersUp(hand)
    print(fingers)
    totalFingers = fingers.count(1)
    msg = "Fingers:" + str(totalFingers)
    cv2.putText(img, msg, (bbox[0]+100,bbox[1]-30),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
cv2.imshow("Hand", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
