import cv2, imutils

img = cv2.imread("koala.jpg")
print(img.shape)
resized_img = imutils.resize(img, width=200)
print(resized_img.shape)
cv2.imshow("Koala", img)
cv2.imshow("Koala:resized", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
