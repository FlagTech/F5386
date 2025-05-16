import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO("yolo12n.pt")
# Streamlit 網頁介面
st.title("上傳圖檔執行YOLO物體偵測")
source_img = st.file_uploader(
    label="選擇圖檔...",
    type=("jpg", "jpeg", "png", 'bmp', 'webp')
)
col1, col2 = st.columns(2)
with col1:
    if source_img:
        uploaded_image = Image.open(source_img)
        st.image(image=source_img,
                 caption="上傳圖檔",
                 use_container_width=True)
if source_img:
    if st.button("執行"):
        with st.spinner("執行中..."):
            res = model.predict(uploaded_image,
                                conf=0.6)                    
            res_plotted = res[0].plot()[:, :, ::-1]
            with col2:
                st.image(res_plotted,
                         caption="偵測結果影像",
                         use_container_width=True)
                try:
                    with st.expander("偵測結果"):
                        for box in res[0].boxes:
                            cords = box.xyxy[0].tolist()
                            cords = [round(x) for x in cords]
                            class_id = int(box.cls[0].item())
                            conf = box.conf[0].item()
                            conf = round(conf*100, 2)
                            st.write("分類編號:", class_id)
                            st.write("分類名稱:", model.names[class_id])
                            st.write("邊界框座標:", cords)
                            st.write("信心指數:", conf, "%")
                            st.write("------------------------")    
                except Exception as ex:
                    st.write("尚未有圖檔上傳!")
                    st.write(ex)

