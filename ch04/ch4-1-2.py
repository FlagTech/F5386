import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
                      min_detection_confidence=0.5)

img = cv2.imread("images/face.jpg")
results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

if results.detections:
    for detection in results.detections:
        mp_drawing.draw_detection(img, detection)
        bbox = detection.location_data.relative_bounding_box
        ih, iw, _ = img.shape
        x, y = int(bbox.xmin * iw), int(bbox.ymin * ih)
        score = str(int(detection.score[0]*100)) + "%"
        cv2.putText(img, score, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("MediaPipe Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

