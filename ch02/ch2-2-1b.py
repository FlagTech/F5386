import cv2

img = cv2.imread("koala.jpg")
print(img.shape)
x = 10; y = 10
w = 200; h= 200
crop_img = img[y:y+h, x:x+w]
cv2.imshow("Koala", crop_img)
print(crop_img.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()
