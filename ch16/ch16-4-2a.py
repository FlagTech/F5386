import ollama

image_url = "images/taipei.jpg"
prompt = "請針對此圖片，使用繁體中文說明你看到了什麼？如果你是一位觀光客，請說明你看到的特點。"

# 使用 Llama-Vision 分析影像
response = ollama.chat(
    model="llama3.2-vision:11b",
    messages=[{
        "role": "user",
        "content": prompt,
        "images": [image_url]
    }]
)
content = response["message"]["content"].strip()
print(content)
