from deepface import DeepFace

result = DeepFace.verify("images/mary.jpg",
                         "images/mary2.jpg",
                         model_name="Facenet")
print(result)
