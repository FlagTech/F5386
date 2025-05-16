import cv2
import numpy as np

video_path = "../media/Counter.mp4"
scaled_size = (800, 600)    # 調整尺寸, 值None是不調整, 元組值是新尺寸(800, 600)

# 回撥函數，用於處理滑鼠事件
def draw_polygon(event, x, y, flags, param):
    global points, drawing, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        if drawing:
            points.append((x, y))
            if len(points) == 4:
                cv2.polylines(img, [np.array(points)],
                              isClosed=True, color=(0, 255, 0),
                              thickness=2)
                cv2.imshow("Frame", img)
                print(f"Points: {points}")
                points = []                
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = img.copy()
        if points:
            for i in range(len(points)):
                cv2.line(img_copy, points[i],
                         points[(i + 1) % len(points)],
                         (0, 255, 0), 2)
            cv2.line(img_copy, points[-1], (x, y), (0, 255, 0), 2)
        cv2.imshow("Frame", img_copy)
    elif event == cv2.EVENT_RBUTTONUP:
        drawing = False
# 初始化變數
drawing = True
points = []
# 開啟影片檔
cap = cv2.VideoCapture(video_path)
# 讀取第一個影格
ret, img = cap.read()
if scaled_size:   # 是否需調整影格尺寸
    img = cv2.resize(img, scaled_size)
if not ret:
    print("錯誤: 無法開啟影片檔...")
    cap.release()
    cv2.destroyAllWindows()
else:
    # 建立一個視窗
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", draw_polygon)
    cv2.imshow("Frame", img)
    cv2.waitKey(0)

# 釋放資源
cap.release()
cv2.destroyAllWindows()
