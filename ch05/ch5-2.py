from cvzone3d.HandTrackingModule import HandDetector
import cv2

detector = HandDetector(detectionCon=0.5, maxHands=1)
img = cv2.imread("images/Scissors.png")
msg = "None"
hands, img = detector.findHands(img)
if hands:
    hand = hands[0]
    fingers = detector.fingersUp(hand)
    print(fingers)
    totalFingers = fingers.count(1)
    if totalFingers == 5:
        msg = "Paper"
    if totalFingers == 0:
        msg = "Rock"
    if totalFingers == 2:
        if fingers[1] == 1 and fingers[2] == 1:
            msg = "Scissors"
    cv2.putText(img, msg, (hand["center"][0]-60, hand["center"][1]),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
cv2.imshow("Hand", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
