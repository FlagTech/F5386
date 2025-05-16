import face_recognition
import cv2
import pickle

known_face_list = [
    {
        "name": "Mary",
        "filename": "mary.jpg",
        "face_encoding": None
    },
    {
        "name": "Jane",
        "filename": "jane.jpg",
        "face_encoding": None      
    },
    {
        "name": "Grace",
        "filename": "grace.jpg",
        "face_encoding": None      
    }
]

for data in known_face_list:
    fname = "images/"+data["filename"]
    print("人臉:[", fname, "]編碼中...")
    img = cv2.imread(fname)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_img)
    data["face_encoding"] = encodings[0]
    print("人臉:[", fname, "]編碼完成...")
     
with open("faces_encoding.dat", "wb") as f:
    pickle.dump(known_face_list, f)
print("人臉編碼已經成功寫入faces_encoding.dat...")
