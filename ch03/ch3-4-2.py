import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

model_path = "models/frozen_east_text_detection.pb"
img = cv2.imread("images/car.jpg")
model = cv2.dnn.readNet(model_path)
outputLayers = []
outputLayers.append("feature_fusion/Conv_7/Sigmoid")
outputLayers.append("feature_fusion/concat_3")
height,width,colorch = img.shape
new_height = (height//32+1)*32
new_width = (width//32+1)*32
h_ratio = height/new_height
w_ratio = width/new_width
blob = cv2.dnn.blobFromImage(img ,1 ,(new_width, new_height),
                            (123.68,116.78,103.94), True)
model.setInput(blob)
(scores, geometry) = model.forward(outputLayers)
rectangles=[]
confidence_score=[]
rows = geometry.shape[2]
cols = geometry.shape[3]
for y in range(0, rows):
    for x in range(0, cols):
        if scores[0][0][y][x] < 0.5:
            continue
        offset_x = x*4
        offset_y = y*4
        bottom_x = int(offset_x + geometry[0][1][y][x])
        bottom_y = int(offset_y + geometry[0][2][y][x])
        top_x = int(offset_x - geometry[0][3][y][x])
        top_y = int(offset_y - geometry[0][0][y][x])
        rectangles.append((top_x, top_y, bottom_x, bottom_y))
        confidence_score.append(float(scores[0][0][y][x]))        
final_boxes = non_max_suppression(np.array(rectangles),
                                  probs=confidence_score,
                                  overlapThresh=0.5)
for (x1,y1,x2,y2) in final_boxes:
    area = abs(x2-x1) * abs(y2-y1)
    if area > 4000:
        x1 = int(x1*w_ratio)
        y1 = int(y1*h_ratio)
        x2 = int(x2*w_ratio)
        y2 = int(y2*h_ratio)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)    
cv2.imshow("EAST", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
