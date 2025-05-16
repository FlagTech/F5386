import cv2
import os

# 來源影片檔的路徑
video_file = "starwars.mp4"
# 切割影格成圖檔的輸出目錄
output_dir = "frames"
# -------------------------------------------
# 確保輸出目錄存在
os.makedirs(output_dir, exist_ok=True)
# 讀取影片
cap = cv2.VideoCapture(video_file)
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 儲存一個影格成為JPG圖檔
    frame_name = os.path.join(output_dir, f"frame{frame_count:d}.jpg")
    cv2.imwrite(frame_name, frame)
    frame_count += 1

cap.release()
print(f"總共 {frame_count} 個影格被抽出儲存至 '{output_dir}' 子目錄.")

