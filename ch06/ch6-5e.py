import cv2
from deepface import DeepFace

img = cv2.imread("images/faces3.jpg")  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
faceCascade = cv2.CascadeClassifier(
        "haarcascades/haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(gray) 

for (x, y, w, h) in faces:
    if w < 100 or h < 100:
        continue
    # 填充人臉方框
    padding = 20
    face = img[y-padding:y+h+padding,
               x-padding:x+w+padding]     
    try:
        analyze = DeepFace.analyze(face, actions=['emotion'])
        emotion = analyze[0]['dominant_emotion']
        print(emotion)        
        cv2.putText(img, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except:
        pass
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5) 

cv2.imshow("Emotions Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()