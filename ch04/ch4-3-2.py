from cvzone3d.FaceDetectionModule import FaceDetector
import cv2

img = cv2.imread("images/face.jpg")
detector = FaceDetector()
img, faces = detector.findFaces(img, draw=False)
if faces:
    keypoints = detector.getFaceKeypoints(img, face_idx=0)
    print(keypoints)
    for keypoint in keypoints:
        cv2.circle(img, keypoint["keypoint"], 5, (255, 0, 255), cv2.FILLED)

cv2.imshow("Face", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
