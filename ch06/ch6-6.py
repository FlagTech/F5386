import cv2
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np

model = cv2.dnn.readNetFromONNX("models/emotion-ferplus-8.onnx")
img = cv2.imread("images/Happy.jpg")
emotions = ['Neutral', 'Happy', 'Surprise', 'Sad',
            'Anger', 'Disgust', 'Fear', 'Contempt']
detector = FaceDetector()
img, faces = detector.findFaces(img)
if faces:
    print("偵測到人臉數:", len(faces))
    x, y, w, h = faces[0]["bbox"]
    print(x, y, w, h)
    # 填充人臉方框
    padding = 20
    padded_face = img[y-padding:y+h+padding,
                      x-padding:x+w+padding]     
    gray = cv2.cvtColor(padded_face, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(gray, (64, 64))
    processed_face = resized_face.reshape(1, 1, 64, 64)
    model.setInput(processed_face)
    outputs = model.forward()
    print(outputs.shape)
    print(outputs)
    # 計算分數的 Softmax值  
    expanded = np.exp(outputs - np.max(outputs))
    probablities =  expanded / expanded.sum()
    # 取得最後的可能性 
    prob = np.squeeze(probablities)
    print(prob)
    # 取得最大可能性的表情
    predicted_emotion = emotions[prob.argmax()]
    print("可能的表情:", predicted_emotion)
    cv2.putText(img, predicted_emotion,(x,y+h+75),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 5)

cv2.imshow("Emotion Recognition", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
