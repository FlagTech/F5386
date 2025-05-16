import easyocr
import numpy as np
import cv2
 
img = cv2.imread("images/sample.jpg")
reader = easyocr.Reader(["ch_tra", "en"]) 
results = reader.readtext("images/sample.jpg")
for result in results:
    box = result[0]
    cv2.rectangle(img, box[0], box[2], (0, 0, 255), 3)
cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
