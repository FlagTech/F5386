import cv2

faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

image = cv2.imread("images/faces.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 從圖片偵測人臉
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print("人臉數:", len(faces))

# 在偵測出的人臉繪出長方形外框
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("preview", image)
cv2.waitKey(0)

cv2.destroyAllWindows()
