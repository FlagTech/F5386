from ultralytics import YOLO
import cv2
import torch

image_file = "images/car.jpg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 載入 YOLO 車牌辨識模型
model = YOLO("license-plate.pt")
model.to(device)
result = model.predict(source=image_file, device=device)
# 取得 bbox 座標
for ele in result:
    for bbox in ele.boxes:
        x1, y1, x2, y2 = bbox.xyxy[0]  # 取得左上角和右下角座標
        print(f"Bounding box: ({x1}, {y1}), ({x2}, {y2})")
res_plotted = result[0].plot()
cv2.imshow("License Plate Recognition", res_plotted)
cv2.imwrite("Result.jpg", img=res_plotted) 
cv2.waitKey(0)
cv2.destroyAllWindows() 
