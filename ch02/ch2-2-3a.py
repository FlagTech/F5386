import cv2

img = cv2.imread("penguins.jpg")
cv2.putText(img, 'OpenCV', (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0,255,255), 5, cv2.LINE_AA)
cv2.putText(img, 'Hello!', (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255,0,255), 5, cv2.LINE_AA)
cv2.imshow("Penguins", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
