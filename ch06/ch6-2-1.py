try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
import cv2
import numpy as np

model_path = "models/mobilenet_v1_1.0_224_quantized_1_metadata_1.tflite"
label_path = "models/labels.txt"
label_names = []
with open(label_path, "r") as f:
    for line in f.readlines():
        label_names.append(line.strip())
interpreter = Interpreter(model_path)
print("成功載入模型...")
interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]["shape"]
print("影像尺寸: (", width, ",", height, ")")
image = cv2.imread("images/dog.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (width, height))
input_data = np.expand_dims(image_resized, axis=0)
interpreter.set_tensor(
    interpreter.get_input_details()[0]["index"],input_data)
interpreter.invoke()
output_details = interpreter.get_output_details()[0]
output = np.squeeze(interpreter.get_tensor(output_details["index"]))
label_id = np.argmax(output)
scale, zero_point = output_details["quantization"]
prob = scale * (output[label_id] - zero_point)
classification_label = label_names[label_id]
print("分類名稱 =", classification_label)
print("影像可能性 =", np.round(prob*100, 2), "%")