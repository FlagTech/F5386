import cv2

image = cv2.imread("images/road_lane1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Road Lane", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
