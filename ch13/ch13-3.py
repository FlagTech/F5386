import easyocr
import numpy as np
import cv2
 
img = cv2.imread("images/car.jpg")
reader = easyocr.Reader(["en"])
results = reader.readtext(img)
y = 0
for box in results:
    points = box[0]
    points = np.array(points, np.int32)
    print(points)
    print(box[1])
    cv2.polylines(img, pts=[points], isClosed=True,
                  color=(0, 0, 255), thickness=3)
    y = y + 30
    cv2.putText(img, box[1], (10, y),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
cv2.imshow("License Plate Recognition", img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
