import cv2
from deepface import DeepFace

img = cv2.imread("images/Surprise.jpg")
try:
    emotion = DeepFace.analyze(img, actions=["emotion"])  # 情緒
    age = DeepFace.analyze(img, actions=["age"])          # 年齡
    race = DeepFace.analyze(img, actions=["race"])        # 種族
    gender = DeepFace.analyze(img, actions=["gender"])    # 性別
    
    # print(emotion)
    print("情緒:", emotion[0]["dominant_emotion"])
    print("年齡", age[0]["age"])
    print("種族", race[0]["dominant_race"])
    print("性別", gender[0]["gender"])
except:
    print("Deepface人臉分析錯誤...")

cv2.imshow("Face Analysis", img)
cv2.waitKey(0)
cv2.destroyAllWindows()