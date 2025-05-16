try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter 
import cv2
import numpy as np

model_path = "models/model.tflite"
label_path = "models/labels.txt"
label_names = []
with open(label_path, "r") as f:
    for line in f.readlines():
        class_name = line.split(" ")
        label_names.append(class_name[1].strip())
interpreter = Interpreter(model_path)
print("成功載入模型...")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
_, height, width, _ = input_details[0]["shape"]
print("影像尺寸: (", width, ",", height, ")")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while cap.isOpened():
    success, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
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
    final_prob = np.round(prob*100, 2)
    print("影像可能性 =", final_prob, "%")
    cv2.putText(frame, classification_label, (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out_msg = str(final_prob) + "%"
    cv2.putText(frame, out_msg, (25, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

