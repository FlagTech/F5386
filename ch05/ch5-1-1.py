from cvzone3d.HandTrackingModule import HandDetector
import cv2

img = cv2.imread("images/hand.jpg")
detector = HandDetector(detectionCon=0.5, maxHands=1)
hands, img = detector.findHands(img)
if hands:
    hand1 = hands[0]   # 第1隻手
    centerPoint1 = hand1["center"]
    print(centerPoint1)
    cv2.circle(img, centerPoint1, 10, (0, 255, 255), cv2.FILLED)
    handType1 = hand1["type"]
    print(handType1)
    cv2.putText(img, handType1, (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
cv2.imshow("Hand", img)
detector.plotHandLandmarks3D()
cv2.waitKey(0)
cv2.destroyAllWindows()
