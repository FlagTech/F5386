from deepface import DeepFace

results = DeepFace.find(img_path="images/obama.png",
                       db_path="face_database/")
print(results[0])