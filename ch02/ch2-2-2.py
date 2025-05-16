import cv2

img = cv2.imread("koala.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Koala:gray", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
