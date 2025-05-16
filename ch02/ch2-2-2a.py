import cv2

img = cv2.imread("penguins.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("Penguins:rgb", rgb_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
