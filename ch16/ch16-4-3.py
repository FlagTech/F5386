import streamlit as st
import ollama
from groq import Groq
from PIL import Image
import base64
import io

# 請在下方填入您的 Groq API Key
GROQ_API_KEY="<API-KEY>"

def encode_image(image):
    """將圖片轉換為 Base64 編碼"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def ollama_analyze_image(image, prompt):
    """使用 Ollama API 分析圖片"""
    try:
        base64_image = encode_image(image)
        response = ollama.chat(
            model="llama3.2-vision:11b",
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [base64_image]
            }]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Ollama API 錯誤: {str(e)}"

def groq_analyze_image(image, prompt):
    """使用 Groq API 分析圖片"""
    try:
        base64_image = encode_image(image)
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }
         
        client = Groq(api_key=GROQ_API_KEY)
        
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content
                ]
            }],
            temperature=1,
            max_completion_tokens=512,
            top_p=1,
            stream=False,
            stop=None
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq API 錯誤: {str(e)}"

st.title("Llama-Vision視覺分析助手")    
# API 選擇
api_choice = st.radio(
    "選擇 API:", 
    ["Ollama", "Groq"], 
    horizontal=True
)
# 圖檔上傳
uploaded_file = st.file_uploader(
    "上傳圖檔", 
    type=["jpg", "jpeg", "png"], 
    help="請上傳您想分析的圖檔"
)
# 預設提示詞
default_prompt = "請針對此圖片，使用繁體中文說明你看到了什麼？如果你是一位觀光客，請說明你看到的特點。"
prompt = st.text_area(
    "請輸入提示詞", 
    value=default_prompt, 
    height=100
)
# 分析按鈕
if st.button("分析影像") and uploaded_file is not None:
    column1, column2 = st.columns(2)
    # 讀取上傳的圖檔
    image = Image.open(uploaded_file)
    column1.image(image, caption="已上傳的圖檔",
                  use_container_width=True)
    # 顯示分析中的載入動畫
    with st.spinner(f"使用 {api_choice} API 分析中, 請稍等一下..."):
        # 依據選擇的 API 進行分析
        if api_choice == "Ollama":
            result = ollama_analyze_image(image, prompt)
        else:
            result = groq_analyze_image(image, prompt)
    # 顯示分析結果
    st.success("影像分析完成!")
    column2.write(result)
