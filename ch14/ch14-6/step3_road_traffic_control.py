from ultralytics import YOLO
import cv2, hotzone
import numpy as np

video_path = "../media/highway2.mp4"

SouthZone = [(270, 459), (587, 463), (538, 642), (37, 602)]     # 南下熱區域
NorthZone = [(695, 464), (1006, 480), (1186, 606), (733, 628)]  # 北上熱區域

model = YOLO("yolo12m.pt")
cap = cv2.VideoCapture(video_path)

SouthTrackList = [] # 南下追蹤串列
NorthTrackList = [] # 北上追蹤串列

while True:
    success, frame = cap.read()
    if not success:
        break
    results = model.track(frame, persist=True, verbose=False,
                          classes=[2,5,7])  # 'car','truck','bus'
    # 檢查是否有追蹤到物體
    if results[0].boxes.id is None:
        continue
    print("track:", len(results[0].boxes.data))
    # 畫出2個熱區域
    hotzone.drawArea(frame, NorthZone, (0,255,0), 5)
    hotzone.drawArea(frame, SouthZone, (255,0,0), 5)
    # 取得追蹤物體的資訊
    for data in results[0].boxes.data:
        x1, y1 = int(data[0]),int(data[1])
        x2, y2 = int(data[2]),int(data[3])
        print(len(data))
        tid = int(data[4])
        r = round(float(data[5]),2)
        name = model.names[int(data[6])]        
        # 取得偵測到物體與熱區域的重疊比例
        p_s = hotzone.inHotZonePercent((x1,y1,x2,y2), SouthZone)
        # 判斷是否是在熱區域
        if p_s >= 30: # 南下 綠
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
            # 顯示物體名與追蹤編號
            cv2.putText(frame, name + ":" + str(tid), (x1,y1-10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            if tid not in SouthTrackList:
                SouthTrackList.append(tid)
        p_n = hotzone.inHotZonePercent((x1,y1,x2,y2), NorthZone)
        if p_n >= 30: # 北上 藍
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 3)
            # 顯示物體名與追蹤編號
            cv2.putText(frame,name + ":#" + str(tid), (x1,y1-10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            if tid not in NorthTrackList:
                NorthTrackList.append(tid)
  
    SouthCount, NorthCount = len(SouthTrackList), len(NorthTrackList)
    cv2.putText(frame, "South:" + str(SouthCount), (300, 450),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, "North:" + str(NorthCount), (800, 450),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.imshow("Road traffic", frame)
    # 按下 "q" 鍵跳出迴圈
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

