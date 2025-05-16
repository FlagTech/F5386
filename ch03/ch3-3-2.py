import cv2

model_path = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_path = "models/deploy.prototxt"
model = cv2.dnn.readNet(model=model_path, config=config_path,
                        framework="Caffe")
image = cv2.imread("images/faces2.jpg")
image_height, image_width = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0,
                             size=(300, 300), mean=(104.0, 117.0, 123.0),
                             swapRB=False, crop=False)
model.setInput(blob)
results = model.forward()
for face in results[0][0]:
    face_confidence = face[2]  # 取得信心指數
    if face_confidence > 0.5:  # 只處理信心指令大於0.5
        x1 = int(face[3] * image_width)
        y1 = int(face[4] * image_height)
        x2 = int(face[5] * image_width)
        y2 = int(face[6] * image_height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
