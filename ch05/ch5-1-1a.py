from cvzone3d.HandTrackingModule import HandDetector
import cv2

img = cv2.imread("images/hands.jpg")
detector = HandDetector(detectionCon=0.5, maxHands=2)
hands, img = detector.findHands(img)
if hands:
    hand1 = hands[0]   # 第1隻手
    centerPoint1 = hand1["center"]
    cv2.circle(img, centerPoint1, 10, (0, 255, 255), cv2.FILLED)
    if len(hands) == 2:
        hand2 = hands[1]   # 第2隻手
        centerPoint2 = hand2["center"]
        print(centerPoint2)
        cv2.circle(img, centerPoint2, 10, (0, 255, 255), cv2.FILLED)
        
cv2.imshow("Hands", img)
detector.plotHandLandmarks3D(box_aspect=[3, 1, 1])
cv2.waitKey(0)
cv2.destroyAllWindows()
