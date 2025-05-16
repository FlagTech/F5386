from cvzone3d.HandTrackingModule import HandDetector
import cv2

img = cv2.imread("images/hand3.jpg")
detector = HandDetector(detectionCon=0.5, maxHands=2)
hands, img = detector.findHands(img)
if hands:
    hand1 = hands[0]
    lmList1 = hand1["lmList"]
    bbox1 = hand1["bbox"]
    length, info, img = detector.findDistance(lmList1[4],lmList1[8], img,
                                              color=(255, 255, 0), scale=10)
    length2, info2 = detector.findDistance3D(lmList1[4],lmList1[8])
    print(int(length))
    print(info)
    print(int(length2))
    print(info2)    
    msg = "Dist:" + str(int(length)) + "/" + str(int(length2))
    cv2.putText(img, msg, (bbox1[0]+100,bbox1[1]-30),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)    
    
cv2.imshow("Hand", img)
cv2.waitKey(0)
cv2.destroyAllWindows()