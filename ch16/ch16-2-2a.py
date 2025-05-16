from groq import Groq
import base64
import io
from PIL import Image

#image_url = "https://media-hosting.imagekit.io//06cae7630...gw__"
image_url = "images/taipei.jpg"
is_url = False
prompt = "請針對此圖片，說明你看到了什麼？如果你是一位觀光客，請說明你看到的特點。"

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

if is_url:
    image_content = {"type": "image_url", "image_url": {"url": image_url}}
else:
    image = Image.open(image_url)
    base64_image = encode_image(image)
    image_content = {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}

client = Groq(
    api_key="<API-KEY>" 
)

completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            },
            image_content
        ]
    }],
    temperature=1,
    max_completion_tokens=512,
    top_p=1,
    stream=False,
    stop=None,
)

# 取出 content 的回應內容
content = completion.choices[0].message.content
print(content)

