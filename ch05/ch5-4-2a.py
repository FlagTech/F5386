from cvzone3d.PoseModule import PoseDetector
import cv2

img = cv2.imread("images/fitness.jpg")
detector = PoseDetector()
img = detector.findPose(img)
lmList, bboxInfo = detector.findPosition(img)
if lmList:
    angle = detector.findAngle(lmList[24], lmList[26], lmList[28])
    print(angle)
    msg = str(int(angle))
    point1, point2, center = lmList[24][:2], lmList[26][:2], lmList[28][:2]
    cv2.line(img, point1, point2, (255, 255, 255), 3)
    cv2.line(img, center, point2, (255, 255, 255), 3)
    cv2.circle(img, point1, 10, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, point1, 15, (0, 0, 255), 2)
    cv2.circle(img, point2, 10, (0, 255, 255), cv2.FILLED)
    cv2.circle(img, point2, 15, (0, 255, 255), 2)
    cv2.circle(img, center, 10, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, center, 15, (0, 0, 255), 2)
    cv2.putText(img, msg, (10, 35),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
cv2.imshow("Pose", img)
cv2.waitKey(0)
cv2.destroyAllWindows()