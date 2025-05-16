from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-cls.pt")

image = cv2.imread("images/cat.jpg")
results = model(image, verbose=False)
for result in results:
    probs = result.probs   # 取得機率資訊
    top1_idx = probs.top1  # 取得最高機率的分類索引
    top5_idx = probs.top5  # 取得前五高機率的分類索引
    # 使用 .data 取得信心指數
    top1_conf = float(probs.data[top1_idx])  # top-1 分類的信心指數
    top5_conf = [float(probs.data[idx])      # top-5 分類的信心指數
                 for idx in top5_idx]  
    # 取得分類名稱
    top1_class = str(result.names[top1_idx])
    top5_classes = [str(result.names[idx]) for idx in top5_idx]

    print(f"最有可能的分類: {top1_class}, 信心指數: {top1_conf:.2f}")
    print("前5個最有可能的分類和信心指數:")
    print("============================")
    for i, (cls, conf) in enumerate(zip(top5_classes, top5_conf)):
        label = f"{i+1}. {cls}: {conf:.2f}"
        print(label)
        cv2.putText(image, label, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
cv2.imshow("Image Classification", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
