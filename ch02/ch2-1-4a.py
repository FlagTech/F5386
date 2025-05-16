import cv2 

gray_img = cv2.imread("koala.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("result.png", gray_img)
