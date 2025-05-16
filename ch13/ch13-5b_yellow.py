import cv2
import numpy as np

image = cv2.imread("images/road_lane.jpg")
copy = np.copy(image)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
hue_value = 25
yellow_lower = np.array([hue_value-10, 100, 100], np.uint8)
yellow_upper = np.array([hue_value+10, 255, 255], np.uint8)
mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
cv2.imshow("Mask Yellow", mask_yellow)
edges = cv2.Canny(mask_yellow, 50, 150)
cv2.imshow("Road Lane edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
