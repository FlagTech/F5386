import cv2

cap = cv2.VideoCapture("YouTube.mp4")

frame_count = 0
while True:
  ret, frame = cap.read()
  if not ret:
      break
  frame_count = frame_count + 1

print("總影格數 = ", frame_count)
cap.release()
