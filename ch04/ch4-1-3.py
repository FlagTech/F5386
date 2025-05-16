import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

img = cv2.imread("images/face2.jpg")
results = face_mesh.process(img)
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(image=img,
                   landmark_list=face_landmarks,
                   connections=mp_face_mesh.FACEMESH_CONTOURS,
                   landmark_drawing_spec=drawing_spec,
                   connection_drawing_spec=drawing_spec)

cv2.imshow("MediaPipe FaceMesh", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
