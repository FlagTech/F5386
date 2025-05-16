import cv2

body_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_upperbody.xml")

image = cv2.imread("images/body.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

bodies = body_cascade.detectMultiScale(gray,
                                       scaleFactor=1.1,
                                       minNeighbors=5,
                                       minSize=(30, 30))

for (x, y, w, h) in bodies:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow("Upper Bodies Detected", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
