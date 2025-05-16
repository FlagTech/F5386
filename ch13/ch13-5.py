import cv2

image = cv2.imread("images/road_lane1.jpg")
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Road Lane", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
