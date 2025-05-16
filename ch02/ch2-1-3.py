import cv2

img = cv2.imread("penguins.jpg")
print(img.shape)
h, w, c = img.shape
print("影像高:", h)
print("影像寬:", w)
