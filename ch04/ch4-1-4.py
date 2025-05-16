import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

img = cv2.imread("images/hands.jpg")
results = hands.process(img)
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(img, hand_landmarks, 
                                  mp_hands.HAND_CONNECTIONS)

cv2.imshow("MediaPipe Hands", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
