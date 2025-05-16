from cvzone3d.HandTrackingModule import HandDetector
import cv2

img = cv2.imread("images/hand2.jpg")
detector = HandDetector(detectionCon=0.5, maxHands=2)
hands, img = detector.findHands(img)
if hands:
    hand1 = hands[0]
    lmList1 = hand1["lmList"]
    img_copy = img.copy()
    angle, img2D = detector.findAngle(lmList1[9], lmList1[10],
                                      lmList1[11], img_copy,
                                      color=(255, 255, 0),
                                      scale=10)
    print(angle)
    print(detector.angleCheck(angle, 150, offset=20))
    cv2.imshow("Hand", img2D)
    angle3D, img3D = detector.findAngle3D(lmList1[9], lmList1[10],
                                          lmList1[11], img,
                                          color=(255, 255, 0),
                                          scale=10)
    print(angle3D)
    print(detector.angleCheck(angle3D, 100, offset=20))
    cv2.imshow("Hand3D", img3D)

cv2.waitKey(0)
cv2.destroyAllWindows()