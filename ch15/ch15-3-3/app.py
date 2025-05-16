#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     app.py
   @Author:        Luyao.zhang
   @Date:          2023/5/15
   @Update Date:   2024/11/22 修改和中文化 By 陳會安
   @Description:
-------------------------------------------------
"""
from pathlib import Path
from PIL import Image
import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video
from utils import infer_uploaded_webcam, infer_uploaded_IPcam

# 設定頁面佈局
st.set_page_config(
    page_title="Ultralytics YOLO 模型任務的互動介面",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# 主頁面標題
st.title("Ultralytics YOLO 模型任務的互動介面")

# 側邊欄
st.sidebar.header("YOLO模型設置")

# 作業選項
task_type = st.sidebar.selectbox(
    "選擇任務",
    ["物體偵測",
     "姿態評估"]
)
# 模型選項
model_type = None
if task_type == "物體偵測":
    model_type = st.sidebar.selectbox(
        "選擇模型",
        config.DETECTION_MODEL_LIST
    )
elif task_type == "姿態評估":
    model_type = st.sidebar.selectbox(
        "選擇模型",
        config.POSE_MODEL_LIST
    )
else:
    st.error("目前已經實作 '物體偵測' 和 '姿態評估' 功能")
# 信心指數選項
confidence = float(st.sidebar.slider(
    "選擇信心指數", 30, 100, 50)) / 100
# 取得模型儲存的路徑
model_path = ""
if model_type:
    if task_type == "物體偵測":
        model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
    elif task_type == "姿態評估":
        model_path = Path(config.POSE_MODEL_DIR, str(model_type))
else:
    st.error("請在側邊欄選擇模型種類")

# 載入YOLO預訓練模型
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"無法載入模型. 請確認模型路徑: {model_path}")

# 圖片/影片選項
st.sidebar.header("圖片檔/影片檔設置")
source_selectbox = st.sidebar.selectbox(
    "選擇來源",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]:   # Image
    infer_uploaded_image(confidence, model, task_type)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model, task_type)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    infer_uploaded_webcam(confidence, model, task_type)
elif source_selectbox == config.SOURCES_LIST[3]: # IPcam
    infer_uploaded_IPcam(confidence, model, task_type)    
else:
    st.error("目前資料來源支援'圖片檔' ,'影片檔' ,'網路攝影機' 和 '網路監控攝影機'")