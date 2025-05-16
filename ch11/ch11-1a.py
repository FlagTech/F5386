import face_recognition
import cv2
import numpy as np
import pickle

with open("faces_encoding.dat", "rb") as f:
    known_face_list = pickle.load(f)    
known_face_encodings = []
for data in known_face_list:
    known_face_encodings.append(data["face_encoding"])

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    success, img = cap.read()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_img)
    encodings = face_recognition.face_encodings(rgb_img,
                                                locations)
    for idx, encoding in enumerate(encodings):
        top, right, bottom, left = locations[idx]        
        distances = face_recognition.face_distance(
                           known_face_encodings, encoding)
        best_match_index = np.argmin(distances)
        if distances[best_match_index] < 0.4:
            name = known_face_list[best_match_index]["name"]
        else:
            name = "Unknown"
        cv2.rectangle(img, (left, top), (right, bottom),
                                  (0, 0, 255), 2)
        cv2.rectangle(img, (left, bottom-35), (right, bottom),
                                  (0, 0, 255), cv2.FILLED) 
        cv2.putText(img, name, (left+6, bottom-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 1)
    cv2.imshow("Face", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
