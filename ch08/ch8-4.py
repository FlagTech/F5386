from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolo11n-seg.pt")

image_path = "images/cat.jpg"
background_path = "images/background.jpg"
image = cv2.imread(image_path)
background = cv2.imread(background_path)
# 確保背景影像和原始影像大小一致
background = cv2.resize(background, (image.shape[1],
                                     image.shape[0]))
results = model(image)
# 取得分割遮罩
polygon = results[0].masks.xy[0]
polygon = polygon.astype(np.int32)
# 建立無符號8位元整數且符合原始影像大小的遮罩
mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask,[polygon],color=(255))
# 建立遮罩與反遮罩
mask = (mask > 0.5).astype(np.uint8)
inverse_mask = 1 - mask
# 使用遮罩更新影像背景
foreground = cv2.bitwise_and(image, image, mask=mask)
background = cv2.bitwise_and(background, background,
                             mask=inverse_mask)
result = cv2.add(foreground, background)

cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
