import cv2

img = cv2.imread("koala.jpg")
width = int(img.shape[1] * 0.8)
height = int(img.shape[0] * 0.5)
resized_img = cv2.resize(img, (width, height))
cv2.imshow("Koala:resized", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
