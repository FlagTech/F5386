from cvzone3d.PoseModule import PoseDetector
import cv2

img = cv2.imread("images/woman2.jpg")
detector = PoseDetector(detectionCon=0.5, trackCon=0.5)
img = detector.findPose(img, draw=False)
lmList, bboxInfo = detector.findPosition(img, draw=False)
if lmList:
    for point in lmList:
        cv2.circle(img, (point[0], point[1]), 3,
                   (0, 255, 255), cv2.FILLED)

cv2.imshow("Pose", img)
detector.plotPoseLandmarks3D(box_aspect=[1, 1, .5])
cv2.waitKey(0)
cv2.destroyAllWindows()
