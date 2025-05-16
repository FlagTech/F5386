from deepface import DeepFace

results = DeepFace.represent(img_path="images/obama.png")
print(len(results[0]["embedding"]), "維特徵向量:")
print(results[0]["embedding"])