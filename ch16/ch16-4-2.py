import ollama
from PIL import Image
import base64
import io

image_url = "images/taipei.jpg"
prompt = "請針對此圖片，使用繁體中文說明你看到了什麼？如果你是一位觀光客，請說明你看到的特點。"

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

image = Image.open(image_url)
base64_image = encode_image(image)
# 使用 Llama-Vision 分析影像
response = ollama.chat(
    model="llama3.2-vision:11b",
    messages=[{
        "role": "user",
        "content": prompt,
        "images": [base64_image]
    }]
)
content = response["message"]["content"].strip()
print(content)