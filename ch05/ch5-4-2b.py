from cvzone3d.PoseModule import PoseDetector
import cv2

img = cv2.imread("images/fitness.jpg")
detector = PoseDetector()
img = detector.findPose(img)
lmList, bboxInfo = detector.findPosition(img, draw=False)
if lmList:
    length, distInfo, img = detector.findDistance(
                               lmList[11], lmList[25], img,
                               color=(255, 255, 0), scale=10)
    print(length, distInfo)
    length2, distInfo2 = detector.findDistance3D(
                               lmList[11], lmList[25])
    print(length2, distInfo2)
    msg = "Dist:" + str(int(length)) + "/" + str(int(length2))
    cv2.putText(img, msg, (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
cv2.imshow("Pose", img)
cv2.waitKey(0)
cv2.destroyAllWindows()