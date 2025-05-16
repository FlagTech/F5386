import cv2
import numpy as np

img = cv2.imread("koala.jpg")
cv2.line(img, (0,0), (200,200), (0,0,255), 5)
cv2.rectangle(img, (20,70), (120,160), (0,255,0), 2)
cv2.rectangle(img, (40,80), (100,140), (255,0,0), -1)
cv2.circle(img,(90,210), 30, (0,255,255), 3)
cv2.circle(img,(140,170), 15, (255,0,0), -1)
points = np.array([[220,220],[230,110],[240,120],
                   [240,140],[220,240]], np.int32)
cv2.polylines(img, [points], True, (255, 0, 255), 3)
cv2.imshow("Koala", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
