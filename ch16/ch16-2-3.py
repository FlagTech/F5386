import streamlit as st
from groq import Groq

# 初始化 Groq 客戶端
client = Groq(
    api_key="<API-KEY>" 
)
# 使用 Groq API 使用 Llama 的 LLM 
def call_llama_model(messages):
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq API錯誤: {str(e)}"

# Streamlit App
st.title("整合Streamlit與Groq API")
st.write("這是一個基於Groq API和Llama模型的聊天介面。")
# 按鈕：建立新聊天
if st.button("新聊天"):
    st.session_state.messages = []  # 清空聊天記錄
# 初始化聊天記錄
if "messages" not in st.session_state:
    st.session_state.messages = []
# 顯示聊天記錄
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# 聊天輸入框
user_input = st.chat_input("請輸入您的訊息")
if user_input:
    user_message = {"role": "user", "content": user_input}
    st.session_state.messages.append(user_message)    
    try:
        messages_to_send = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages
        ]
        # 呼叫 Groq API，傳入完整的對話記錄
        ai_response = call_llama_model(messages_to_send)
        # 新增 AI 回應到聊天記錄
        ai_message = {"role": "assistant", "content": ai_response}
        st.session_state.messages.append(ai_message)
        # 顯示使用者的輸入
        with st.chat_message("user"):
            st.markdown(user_input)
        # 顯示 AI 回應
        with st.chat_message("assistant"):
            st.markdown(ai_response)
    except Exception as e:
        st.error(f"發生錯誤：{str(e)}")