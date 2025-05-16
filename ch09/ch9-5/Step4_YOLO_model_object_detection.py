from ultralytics import YOLO
import cv2
import torch

# 圖檔路徑
input = "Hamster02.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 載入 YOLO 模型
model1 = YOLO("best.pt")
model1.to(device)
# 執行物體偵測
results = model1.predict(source=input, device=device)
print("--------------------------")
for result in results:
    boxes = result.boxes
    if boxes:
        print(boxes[0].xyxy.tolist()[0])
        conf = float(boxes[0].conf[0]*100)/100
        print("信心指數: ", conf)

res_plotted = results[0].plot()
cv2.imshow("YOLO Object Detection", res_plotted)
cv2.imwrite("Result.jpg", img = res_plotted) 
cv2.waitKey(0)
cv2.destroyAllWindows()
