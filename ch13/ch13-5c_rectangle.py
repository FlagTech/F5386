import cv2
import numpy as np

image = cv2.imread("images/road_lane1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (5, 5), 0)

edges = cv2.Canny(image, 50, 150)

height, width = edges.shape
print(height, width)
# 長方形區域
rectangle = np.array([[
             (0, int(height * 1 / 2)+20),
             (width, int(height * 1 / 2)+20),
             (width, height),
             (0, height),
            ]])
# 建立相同尺寸的黑色影像
mask = np.zeros_like(edges)
# 分割截取出車道位置的長方形
mask = cv2.fillPoly(mask, rectangle, 255)
isolated_area = cv2.bitwise_and(edges, mask)
cv2.imshow("Isolated Area", isolated_area)
isolated_area2 = cv2.addWeighted(mask, 0.8, isolated_area, 1, 1)
cv2.imshow("Isolated Area2", isolated_area2)

cv2.waitKey(0)
cv2.destroyAllWindows()
