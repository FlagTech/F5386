import cv2 

gray_img = cv2.imread("koala.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Koala:gray", gray_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
