import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
cap.set(cv2.CAP_PROP_FPS, 25)

while(cap.isOpened()):
  ret, frame = cap.read()
  cv2.imshow("Frame", frame)      
  if cv2.waitKey(1) & 0xFF == ord("q"):
      break
  
cap.release()
cv2.destroyAllWindows()
