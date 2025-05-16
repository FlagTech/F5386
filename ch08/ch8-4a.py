from cvzone.SelfiSegmentationModule import SelfiSegmentation
import cv2

segmentor = SelfiSegmentation()

image_path = "images/dog.jpg"
background_path = "images/background.jpg"
image = cv2.imread(image_path)
background = cv2.imread(background_path)
# 確保背景影像和原始影像大小一致
background = cv2.resize(background, (image.shape[1],
                                     image.shape[0]))
# 執行影像分割, 參數cutThreshold是分割敏感度
segmented_img = segmentor.removeBG(image, background,
                                   cutThreshold=0.1)

cv2.imshow("Result", segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
