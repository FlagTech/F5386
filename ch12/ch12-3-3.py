from cvzone3d.PoseModule import PoseDetector
import cv2
import numpy as np

cap = cv2.VideoCapture("media/Squat.mp4")
detector = PoseDetector(detectionCon=0.5, trackCon=0.5)
dir = 0  # 0: 站起  1: 蹲下
count = 0.5
while True:
    success, img = cap.read()
    if success:
        h, w, c = img.shape
        img = detector.findPose(img, draw=True)
        lmList, bboxInfo = detector.findPosition(img, draw=False,
                                                 scaleZ=0.5)
        if lmList:
            angle, img = detector.findAngle3D(lmList[24], lmList[26],
                                              lmList[28], img,
                                              color=(0, 255, 255),
                                              scale=10)
            print(int(angle))
            # 顯示進度條
            bar = np.interp(angle, (50, 150), (w//2-100, w//2+100))
            cv2.rectangle(img, (w//2-100, 50), (int(bar), 100),
                               (0, 255, 0), cv2.FILLED)
            if angle <= 80:   # 目前狀態:蹲下
                if dir == 0:  # 之前狀態:站起
                    count = count + 0.5
                    dir = 1   # 更新狀態:蹲下
            if angle >= 140:  # 目前狀態:站起
                if dir == 1:  # 之前狀態:蹲下
                    count = count + 0.5
                    dir = 0   # 更新狀態:站起
            msg = str(int(count))        
            cv2.putText(img, msg, (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 5,
                        (255, 255, 255), 20)
            cv2.imshow("Squat", img)        
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

