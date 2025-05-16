import easyocr
import numpy as np
import cv2

boxes = [[32, 159, 8, 47],  # horizontal_list[0]
         [6, 191, 43, 81],
         [30, 178, 86, 118]]

img = cv2.imread("images/sample.jpg")
reader = easyocr.Reader(["ch_tra", "en"]) 
results = reader.recognize(img, horizontal_list=boxes,
                           free_list=[])
for result in results:
    print(result[0])
    print(result[1])
    print(result[2])
