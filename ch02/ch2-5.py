import cv2

image = cv2.imread("Happy.jpg")
# 建立負片效果
negative_image = cv2.bitwise_not(image)

cv2.imshow("Original Image", image)
cv2.imshow("Negative Image", negative_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
