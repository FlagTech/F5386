import cv2

cap = cv2.VideoCapture("YouTube.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS =", fps)

cap.release()
