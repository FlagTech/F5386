import easyocr
import cv2

reader = easyocr.Reader(["en"]) 
result = reader.readtext("images/number.jpg")
print(result)
reader = easyocr.Reader(["ch_sim", "en"])
img = cv2.imread("images/simple.jpg")
result = reader.readtext(img)
print(result)
reader = easyocr.Reader(["ch_tra", "en"])
with open("images/traditional.jpg", "rb") as f:
    img = f.read()
result = reader.readtext(img)
print(result)
