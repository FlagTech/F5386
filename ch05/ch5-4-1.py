from cvzone3d.PoseModule import PoseDetector
import cv2

img = cv2.imread("images/woman.jpg")
detector = PoseDetector(detectionCon=0.5, trackCon=0.5)
img = detector.findPose(img)
lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)
if lmList:
    x1, y1, w, h = bboxInfo["bbox"] 
    cv2.rectangle(img, (x1, y1),
                       (x1 + w, y1 + h),
                       (255, 0, 255), 2)             
    center = bboxInfo["center"]
    cv2.circle(img, center, 15, (0, 255, 255), cv2.FILLED)

cv2.imshow("Pose", img)
detector.plotPoseLandmarks3D()
cv2.waitKey(0)
cv2.destroyAllWindows()
