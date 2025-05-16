import cv2
import numpy as np

model_path = "models/DenseNet_121.caffemodel"
config_path = "models/DenseNet_121.prototxt.txt"
class_path = "models/classification_classes_ILSVRC2012.txt"
class_names = []
with open(class_path, "r") as f:
    for line in f.readlines():
        class_names.append(line.split(",")[0].strip())
model = cv2.dnn.readNet(model=model_path, config=config_path,
                        framework="Caffe")
img = cv2.imread("images/cat3.jpg")
blob = cv2.dnn.blobFromImage(image=img, scalefactor=0.01,
                             size=(224, 224), mean=(104, 117, 123))
model.setInput(blob)
outputs = model.forward()
final_outputs = outputs[0].reshape(1000, 1)
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))
probs = softmax(final_outputs)
final_prob = np.round(np.max(probs)*100, 2)
label_id = np.argmax(probs)
out_name = class_names[label_id]
cv2.putText(img, out_name, (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
out_msg = str(final_prob) + "%"
cv2.putText(img, out_msg, (25, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()