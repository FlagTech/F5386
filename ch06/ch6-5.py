from deepface import DeepFace
import cv2

img1 = cv2.imread("images/mary.jpg")
img2 = cv2.imread("images/mary2.jpg")
result = DeepFace.verify(img1, img2)
print(result)
