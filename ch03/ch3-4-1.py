import numpy as np
import cv2

model_path = "models/MobileNetSSD_deploy.caffemodel"
config_path = "models/MobileNetSSD_deploy.prototxt.txt"
class_path = "models/MobileNetSSD_labels.txt"
class_names = []
with open(class_path, "r") as f:
    class_names = f.read().split("\n")
net = cv2.dnn.readNet(config_path, model_path)    
img = cv2.imread("images/people.jpg")
h, w = img.shape[:2]
blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)
net.setInput(blob)
detections = net.forward()
print(detections.shape)
detections = np.squeeze(detections)
print(len(detections))  # 偵測到幾個
for i in range(0, detections.shape[0]):
   confidence = detections[i, 2]
   if confidence > 0.5:
       idx = int(detections[i, 1])
       box = detections[i, 3:7] * np.array([w, h, w, h])
       (startX, startY, endX, endY) = box.astype("int")
       cv2.rectangle(img, (startX, startY), (endX, endY),
                     (10, 255, 0), 2)       
       prob = np.round(confidence*100, 2)
       label = class_names[idx] + ": " + str(prob) + "%"
       y = startY - 15 if startY - 15 > 15 else startY + 15
       cv2.putText(img, label, (startX, y),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (10, 255, 0), 2)
       print(label) 

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
