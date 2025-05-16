import cv2

eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

image = cv2.imread("images/Happy.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

eyes = eye_cascade.detectMultiScale(gray,
                                    scaleFactor=1.1,
                                    minNeighbors=5,
                                    minSize=(30, 30))

for (x, y, w, h) in eyes:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow("Eyes Detected", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
