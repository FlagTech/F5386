import streamlit as st
import numpy as np
import pandas as pd
import time

# 10-7-2 
st.set_page_config(
   page_title="我的Streamlit應用程式",
   page_icon="random",
   layout="centered",
   initial_sidebar_state="expanded",
   menu_items={
      "Get Help": 'https://fchart.github.io/',
      "About": "**[fChart](https://fchart.github.io/)** 是fChart工具的網頁"
   }
)
# 10-2
st.title("我的Streamlit應用程式")
st.subheader("YOLO物體偵測介面")
# 10-3
name = "YOLO"
st.write(name)
st.write("建立**DataFrame資料表格**：")

df = pd.DataFrame({
    "索引": [15, 16, 17, 18],
    "值": [100, 200, 300, 400]
})
df
name
# 10-4
chart_data = pd.DataFrame(
    np.random.randn(30, 3),
    columns=["A", "B", "C"])
st.line_chart(chart_data)

map_data = pd.DataFrame(
    np.random.randn(30, 2) / [50, 50] + [22.6, 120.4],
    columns=["lat", "lon"])
st.map(map_data)
# 10-5
if st.button("請按我"):
    st.text("你已經按下此按鈕！")

if st.checkbox("顯示圖表"):
    chart_data = pd.DataFrame(
        np.random.randn(30, 3),
        columns=["A", "B", "C"])
    st.line_chart(chart_data)

option = st.selectbox(
    "你最喜歡的電腦視覺套件？",
    ["OpenCV", "MediaPipe/CVZone", "Dlib", "YOLO"])
st.text("你的選擇: " + option)

confidence = int(st.slider("選擇信心指數", 30, 100, 50))
st.write(confidence/100.0)

with st.form(key="register_form"):
    name = st.text_input(label="姓名", placeholder="請輸入姓名")
    gender = st.selectbox("性別", ["男", "女"])
    birthday = st.date_input("生日")
    submit_btn = st.form_submit_button(label="送出")
if submit_btn:
    st.write(f"註冊資料: {name}, 性別:{gender}, 生日:{birthday}")
# 10-6-1 
confidence = int(st.sidebar.slider("選擇信心指數", 30, 100, 50,
                                   key="conf"))
st.sidebar.write(confidence/100.0)

column1, column2 = st.columns(2)
column1.write("第一欄")
column2.write("第二欄")

expander = st.expander("請點擊展開...")
expander.write("展開顯示更多內容。")

tab1, tab2 = st.tabs(["一隻貓", "一隻狗"])
with tab1:
   st.header("一隻貓")
   st.image("images/cat.jpg", width=200)
with tab2:
   st.header("一隻狗")
   st.image("images/dog.jpg", width=200)
# 10-6-2
if st.button("開始計數"):
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1, f"目前進度: {i+1} %")
        time.sleep(0.05)
    bar.progress(100, "載入完成...")

if st.button("儲存資料", type="primary"):
    st.toast("編輯內容已經成功儲存...")

if st.button("顯示訊息框", type="secondary"):
    st.success("成功...")
    st.info("資訊...")
    st.warning("警告...")
    st.error("錯誤...")
    
if st.button("顯示網頁特效"):
    st.balloons()
    st.snow()
# 10-6-3
with st.chat_message("user"): 
    st.write("Hi! 請問你是誰？")
message = st.chat_message("assistant")
message.write("你好！ LLM 可以回答你各種問題...")
message.write("請問我有什麼可以幫助你的嗎？")

st.chat_input("請輸入聊天訊息...")
# 10-7-1
@st.cache_data(ttl=3600, show_spinner="正在載入資料...")
def load_csv_data(url):
    return pd.read_csv(url)

df = load_csv_data("https://raw.githubusercontent.com/fchart/PythonCV/refs/heads/main/26k-consumer-complaints.csv")
st.dataframe(df)




