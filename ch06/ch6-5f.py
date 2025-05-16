import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(
        "haarcascades/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture("media/bean_input.mp4")

while True:
    ret, img = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
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
    if cv2.waitKey(5) == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()