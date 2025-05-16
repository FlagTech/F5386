from cvzone3d.PoseModule import PoseDetector
import cv2

img = cv2.imread("images/fitness.jpg")
detector = PoseDetector()
img = detector.findPose(img)
lmList, bboxInfo = detector.findPosition(img, draw=False)
if lmList:
    img_copy = img.copy()
    angle, img2D = detector.findAngle(lmList[24], lmList[26], lmList[28],
                               img_copy, color=(255, 255, 0), scale=10)
    print(angle)
    print(detector.angleCheck(angle, 140, offset=20))
    cv2.imshow("Pose", img2D)
    angle3D, img3D = detector.findAngle3D(lmList[24], lmList[26], lmList[28],
                               img, color=(255, 255, 0), scale=10)
    print(angle3D)
    print(detector.angleCheck(angle, 90, offset=20))
    cv2.imshow("Pose3D", img3D)

detector.plotPoseLandmarks3D(box_aspect=[1, 1, .5])
cv2.waitKey(0)
cv2.destroyAllWindows()