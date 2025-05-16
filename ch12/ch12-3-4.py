from cvzone3d.PoseModule import PoseDetector
import cv2
import numpy as np

cap = cv2.VideoCapture("media/Pull_up.mp4")
detector = PoseDetector(detectionCon=0.5, trackCon=0.5)
dir = 0  # 0是在下方, 1是在上方
count = 0.5
while True:
    success, img = cap.read()
    if success:
        img = cv2.resize(img, (640, 480))
        h, w, c = img.shape
        img = detector.findPose(img, draw=True)
        lmList, bboxInfo = detector.findPosition(img, draw=False,
                                                 scaleZ=0.5)
        if lmList:
            # 右手肘的角度
            right_angle, img = detector.findAngle3D(lmList[12], lmList[14],
                                                    lmList[16], img,
                                                    color=(0, 0, 255),
                                                    scale=10)
            # 計算進度條的高度從50~180範圍轉換成200~400之間的長度
            right_bar = np.interp(right_angle, (50, 180),
                                  (h//2-100, h//2+100))
            # 繪出進度條的方框, 和進度條的長度
            cv2.rectangle(img, (500, h//2-100), (520, h//2+100),
                          (0, 255, 0), 3)
            cv2.rectangle(img, (500, int(right_bar)), (520, h//2+100),
                          (0, 255, 0), cv2.FILLED)
            # 左手肘的角度
            left_angle, img = detector.findAngle3D(lmList[11], lmList[13],
                                                   lmList[15], img,
                                                   color=(0, 255, 255),
                                                   scale=10)
            left_bar = np.interp(left_angle, (50, 180),
                                 (h//2-100, h//2+100))
            cv2.rectangle(img, (100, h//2-100), (120, h//2+100),
                          (0, 255, 0), 3)
            cv2.rectangle(img, (100, int(left_bar)), (120, h//2+100),
                          (0, 255, 0), cv2.FILLED)
            print(int(left_angle), int(right_angle))
            # 目前狀態:下方
            if left_angle >= 150 and right_angle >= 150:
                if dir == 1: # 之前狀態:上方
                    count = count + 0.5
                    dir = 0
            # 目前狀態:上方        
            if left_angle <= 95 and right_angle <= 95:
                if dir == 0: # 之前狀態:下方
                    count = count + 0.5
                    dir = 1
            msg = str(int(count))        
            cv2.putText(img, msg, (50, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        5, (255, 255, 255), 20)
            cv2.imshow("Pull up", img)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
