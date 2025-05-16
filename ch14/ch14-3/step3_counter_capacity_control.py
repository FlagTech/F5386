import hotzone
import cv2, numpy as np
from ultralytics import YOLO

video_path = "../media/Counter.mp4"
# 熱區座標串列
Line1Zone = [(416, 3), (503, 3), (591, 510), (454, 521)]        # 走道1

model = YOLO("yolo12m.pt")
cap = cv2.VideoCapture(video_path)

while True:    
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.resize(frame,(800,600))
    results = model(frame, verbose=False)
    # 畫出熱區域
    hotzone.drawArea(frame, Line1Zone, (0,255,0), 5)
    # 儲存偵測到的物體, 只儲存0, 是'person'
    pList=[]
    # 一一取出偵測到的物體
    for box in results[0].boxes.data:
        if int(box[5]) == 0:                # 判斷是人"person"
            obj = [int(box[0]),int(box[1]), # 邊界框座標
                   int(box[2]),int(box[3]),
                   round(float(box[4]),2)]  # 信心指數        
            pList.append(obj) 
     
    Line1Count = 0
    # 統計人數和繪出邊界框
    for p in pList:
        area = [[p[0],p[1]],[p[2],p[1]],
                [p[2],p[3]],[p[0],p[3]]]
        # 判斷是否在走道
        if hotzone.inHotZonePercent(p, Line1Zone) > 25:
            Line1Count += 1                
            hotzone.drawArea(frame, area, (0,0,255), 3)

    cv2.putText(frame, "Line1=" + str(Line1Count), (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                2, cv2.LINE_AA)
    cv2.imshow("Cashier counter", frame)
    # 按下 "q" 鍵跳出迴圈
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
