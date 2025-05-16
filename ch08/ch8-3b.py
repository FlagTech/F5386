from ultralytics import YOLO
import cv2

model = YOLO("yolo11m-pose.pt") 

image_path = "images/pose.jpg"
results = model(image_path, verbose=False)
annotated_image = results[0].plot()

result = results[0]
boxes = result.boxes
keypoints = result.keypoints
for box, keypoint in zip(boxes, keypoints):
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    class_id = int(box.cls[0].item())
    conf = box.conf[0].item()
    conf = round(conf*100, 2)
    print("分類索引:", class_id)
    print("分類名稱:", model.names[class_id])
    print("邊界框座標:", cords)
    print("可能性:", conf, "%")
    print("=============================")
    list_p = keypoint.data.tolist()
    for i, point in enumerate(list_p[0]):
        x = int(point[0])
        y = int(point[1])
        print(f"{i}: ({x}, {y})")

cv2.imshow("Pose Estimation", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()