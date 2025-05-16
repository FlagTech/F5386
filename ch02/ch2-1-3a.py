import cv2

img = cv2.imread("penguins.jpg")
print(img.shape)
resized_img = cv2.resize(img, (400, 300))
print(resized_img.shape)
#cv2.imshow("Penguins", img)
cv2.imshow("Penguins:resized", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
