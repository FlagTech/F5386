import cv2

img = cv2.imread("penguins.jpg")
cv2.imshow("Penguins", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
