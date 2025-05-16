from cvzone3d.HandTrackingModule import HandDetector
import cv2

img = cv2.imread("images/hands3.jpg")
detector = HandDetector(detectionCon=0.5, maxHands=2)
hands, img = detector.findHands(img)
if hands:
    hand1 = hands[0]
    lmList1 = hand1["lmList"]
    if len(hands) == 2:
        hand2 = hands[1]
        lmList2 = hand2["lmList"]
        bbox2 = hand2["bbox"]
        length, info, img = detector.findDistance3D(lmList1[8], lmList2[8], img,
                                                    color=(255, 255, 0), scale=10)
        print(info)
        msg = "Dist:" + str(int(length))
        cv2.putText(img, msg,(bbox2[0]+150,bbox2[1]-30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
cv2.imshow("Hands", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
