import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math

image_path = "images/business-person.png" 
# 欲顯示的高度與寬度
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
# 建立用於建立 FaceStylizer 物件的選項
base_options = python.BaseOptions(model_asset_path=
                     "models/face_stylizer_color_sketch.task")
options = vision.FaceStylizerOptions(base_options=base_options)
# 建立臉部風格化物件
with vision.FaceStylizer.create_from_options(options) as stylizer:
    # 讀取 MediaPipe 所需的圖檔
    image = mp.Image.create_from_file(image_path)
    # 取得風格化後的影像
    stylized_image = stylizer.stylize(image)
    # 顯示風格化後的影像
    rgb_stylized_image = cv2.cvtColor(stylized_image.numpy_view(),
                                      cv2.COLOR_BGR2RGB)
    # 調整成所需的影像尺寸
    h, w = rgb_stylized_image.shape[:2]
    if h < w:
       img = cv2.resize(rgb_stylized_image,
                        (DESIRED_WIDTH,
                         math.floor(h/(w/DESIRED_WIDTH))))
    else:
       img = cv2.resize(rgb_stylized_image,
                        (math.floor(w/(h/DESIRED_HEIGHT)),
                         DESIRED_HEIGHT))
    cv2.imshow("Color Sketch", img)
    
cv2.waitKey(0)   
cv2.destroyAllWindows()  