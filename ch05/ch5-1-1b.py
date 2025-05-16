from cvzone3d.HandTrackingModule import HandDetector
import cv2

img = cv2.imread("images/hand.jpg")
detector = HandDetector(detectionCon=0.5, maxHands=2)
hands = detector.findHands(img, draw=False)
print(hands)
if hands:
    hand1 = hands[0]   # 第1隻手
    bbox1 = hand1["bbox"]
    x, y, w, h = bbox1
    cv2.rectangle(img, (x, y), (x+w, y+h),
                              (0, 0, 255), 2)
    lmList1 = hand1["lmList"]
    for point in lmList1:
        x, y, z = point
        cv2.circle(img, (x, y), 3, (0, 255, 255), cv2.FILLED)    

cv2.imshow("Hand", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
