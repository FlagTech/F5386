import cv2

img = cv2.imread("koala.jpg")
fliped_img = cv2.flip(img, -1)
cv2.imshow("Koala:fliped", fliped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
