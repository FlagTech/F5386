from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolo11m-pose.pt") 

image = cv2.imread("images/pose.jpg")
results = model(image)

def draw_head(img, list_p):
    list_p = list_p[0]
    for p in list_p[:5]:
        if p[0]!=0 and p[1]!=0:
            cv2.circle(img, (int(p[0]), int(p[1])), 8, (0,255,0),-1)            
    return img

def draw_body(img, list_p):
    list_p = list_p[0]
    points = []
    for p in [list_p[5], list_p[6], list_p[12],list_p[11]]:
        cv2.circle(img, (int(p[0]), int(p[1])), 12, (0,255,0),-1)
        point = (int(p[0]), int(p[1]))
        points.append(point)
    
    points = np.array(points)
    points = points.reshape((-1, 1, 2))
    isClosed = True
    color = (0, 0, 255)
    thickness = 3
    img = cv2.polylines(img, [points], isClosed, color, thickness)
    return img

def draw_upper(img, list_p):
    list_p = list_p[0]
    # upper left
    for i, p in enumerate([list_p[5], list_p[7], list_p[9]]):
        cv2.circle(img, ( int(p[0]), int(p[1])), 8, (0,255,0),-1)
    lines = [(5,7), (7,9)]
    for n in lines:
        start = (int(list_p[n[0]][0]), int(list_p[n[0]][1]))
        end = (int(list_p[n[1]][0]), int(list_p[n[1]][1]))
        cv2.line(img, start, end, (0,0,255), 3)
    # upper right
    for i, p in enumerate([list_p[6], list_p[8], list_p[10]]):
        cv2.circle(img, ( int(p[0]), int(p[1])), 8, (0,255,0),-1)
    lines = [(6,8), (8,10)]
    for n in lines:
        start = (int(list_p[n[0]][0]), int(list_p[n[0]][1]))
        end = (int(list_p[n[1]][0]), int(list_p[n[1]][1]))
        cv2.line(img, start, end, (0,0,255), 3)
    return img

def draw_lower(img, list_p):
    list_p = list_p[0]
    # lower left
    for i, p in enumerate([list_p[11], list_p[13], list_p[15]]):
        cv2.circle(img, ( int(p[0]), int(p[1])), 8, (0,255,0),-1)
    lines = [(11,13), (13,15)]
    for n in lines:
        start = (int(list_p[n[0]][0]), int(list_p[n[0]][1]))
        end = (int(list_p[n[1]][0]), int(list_p[n[1]][1]))
        cv2.line(img, start, end, (0,0,255), 3)
    # lower right
    for i, p in enumerate([list_p[12], list_p[14], list_p[16]]):
        cv2.circle(img, ( int(p[0]), int(p[1])), 8, (0,255,0),-1)
    lines = [(12,14), (14,16)]
    for n in lines:
        start = (int(list_p[n[0]][0]), int(list_p[n[0]][1]))
        end = (int(list_p[n[1]][0]), int(list_p[n[1]][1]))
        cv2.line(img, start, end, (0,0,255), 3)
    return img

result = results[0]
keypoints = result.keypoints
for keypoint in keypoints:
    list_p = keypoint.data.tolist()
    draw_head(image, list_p)
    draw_body(image, list_p)
    draw_upper(image, list_p)
    draw_lower(image, list_p)

cv2.imshow("Pose Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()