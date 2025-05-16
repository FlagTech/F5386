import cv2, imutils

img = cv2.imread("koala.jpg")
rotated_img = imutils.rotate(img, angle=90)
translated_img = imutils.translate(img, 25, -75)
cv2.imshow("Koala:rotated", rotated_img)
cv2.imshow("Koala:translated", translated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
