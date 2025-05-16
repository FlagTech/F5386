import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

base_options = python.BaseOptions(
       model_asset_path='models/blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)
img = mp.Image.create_from_file("images/face.jpg")
detection_result = detector.detect(img)
if detection_result:
    img = img.numpy_view()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x1 = bbox.origin_x
        y1 = bbox.origin_y
        x2 = bbox.origin_x + bbox.width
        y2 = bbox.origin_y + bbox.height
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for keypoint in detection.keypoints:
            cx = int(keypoint.x * width)  
            cy = int(keypoint.y * height)
            cv2.circle(img, (cx, cy), 2, (255, 0, 255), 2)

        category = detection.categories[0]
        probability = round(category.score, 2)
        score = str(probability * 100) + "%"
        cv2.putText(img, score, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("MediaPipe Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


