from cvzone3d.HandTrackingModule import HandDetector
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(detectionCon=0.5, maxHands=2)
while cap.isOpened():
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        length, info, img = detector.findDistance3D(lmList1[8],lmList1[12], img,
                                                    color=(255, 255, 0), scale=10)
        msg = "Dist:" + str(int(length))
        cv2.putText(img, msg,(bbox1[0]+100,bbox1[1]-30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)        
        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            length, info, img = detector.findDistance(lmList1[8],lmList2[8], img,
                                                    color=(255, 255, 0), scale=10)
            msg = "Dist:" + str(int(length))
            cv2.putText(img, msg,(bbox2[0]+100,bbox2[1]-30),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Hands", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
cap.release()
cv2.destroyAllWindows()