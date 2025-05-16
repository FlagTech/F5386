import easyocr
import numpy as np
import cv2
 
img = cv2.imread("images/sample.jpg")
reader = easyocr.Reader(["ch_tra", "en"]) 
horizontal_list, free_list = reader.detect(img)
for box in horizontal_list[0]:
    print(box)
    cv2.rectangle(img, (box[0], box[2]), (box[1], box[3]),
                  (0, 0, 255), 3)
cv2.imshow("Detection", img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
