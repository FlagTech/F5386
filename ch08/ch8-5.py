from ultralytics import YOLO
import cv2
import math

def calculate_angle(img, p1, p2, p3, draw=True):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                         math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle = -angle
    if draw:
        cv2.line(img, p1, p2, (0, 255, 0), 2)
        cv2.line(img, p2, p3, (0, 255, 0), 2)
        cv2.circle(img, p1, 5, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, p2, 5, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, p3, 5, (0, 0, 255), cv2.FILLED)    
        cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                  cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return angle, img

model = YOLO("yolo11m-pose.pt") 

img = cv2.imread("images/site_up.jpg")
results = model(img)
img = results[0].plot()
result = results[0]
keypoints = result.keypoints[0].data.tolist()[0]
if keypoints is not None:
    p5 = (int(keypoints[5][0]), int(keypoints[5][1]))
    p11 = (int(keypoints[11][0]), int(keypoints[11][1]))
    p13 = (int(keypoints[13][0]), int(keypoints[13][1]))
    print(p5, p11, p13)    
    angle, img = calculate_angle(img, p5, p11, p13)
    print("Angle:", int(angle))

cv2.imshow("Pose", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

