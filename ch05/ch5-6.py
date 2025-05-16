from cvzone3d.PoseModule import PoseDetector
import cv2

img = cv2.imread("images/push_up.jpg")
detector = PoseDetector()
img = detector.findPose(img)
lmList, bboxInfo = detector.findPosition(img, draw=False)
if lmList:
    angle1, img = detector.findAngle3D(lmList[11], lmList[23], lmList[25], 
                                      img, color=(255, 255, 0), scale=10)
    print(angle1)
    angle2, img = detector.findAngle3D(lmList[11], lmList[13], lmList[15],
                                      img, color=(255, 255, 0), scale=10)
    print(angle2)
    
cv2.imshow("Pose", img)
cv2.waitKey(0)
cv2.destroyAllWindows()