import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
                    
img = cv2.imread("images/pose.jpg")
results = pose.process(img)
mp_drawing.draw_landmarks(
           img,
           results.pose_landmarks,
           mp_pose.POSE_CONNECTIONS)

cv2.imshow("MediaPipe Pose", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
