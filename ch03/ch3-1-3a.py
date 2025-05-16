import cv2

smile_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")

image = cv2.imread("images/Happy.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

smiles = smile_cascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=35,
                                        minSize=(25, 25))

for (x, y, w, h) in smiles:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow("Smiles Detected", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
