try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
import cv2
import numpy as np

model_path = "models/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite"
label_path = "models/labelmap.txt"
label_names = []
with open(label_path, "r") as f:
    for line in f.readlines():
        label_names.append(line.strip())
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = input_details[0]["shape"]
cap = cv2.VideoCapture("media/motor_bike.mp4")
imWidth  = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
imHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
while cap.isOpened():
    success, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    interpreter.set_tensor(input_details[0]["index"],input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]["index"])[0]
    classes = interpreter.get_tensor(output_details[1]["index"])[0]
    scores = interpreter.get_tensor(output_details[2]["index"])[0] 
    for i in range(len(scores)):
        if ((scores[i] > 0.5) and (scores[i] <= 1.0)):
         startY = int(max(1,(boxes[i][0] * imHeight)))
         startX = int(max(1,(boxes[i][1] * imWidth)))
         endY = int(min(imHeight,(boxes[i][2] * imHeight)))
         endX = int(min(imWidth,(boxes[i][3] * imWidth)))              
         cv2.rectangle(frame, (startX, startY), (endX, endY),
                       (10, 255, 0), 2)  
         object_name = label_names[int(classes[i])]
         prob = np.round(scores[i]*100, 2)
         label = object_name + ": " + str(prob) + "%"
         y = startY - 15 if startY - 15 > 15 else startY + 15
         cv2.putText(frame, label, (startX, y),
                     cv2.FONT_HERSHEY_SIMPLEX,
                     0.5, (10, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

