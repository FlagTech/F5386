#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     utils.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Update Date:   2024/11/22 修改和中文化 By 陳會安
   @Description:
-------------------------------------------------
"""
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile

def _display_detected_frames(conf, model, st_frame, image, task_type):
    """
    使用 YOLO11 模型在影片幀上標記出偵測到的物體。
    :param conf (float): 物體偵測的信心指數閾值。
    :param model (YOLO11): YOLO11 模型的實例。
    :param st_frame (Streamlit 物件): 用於顯示偵測影片結果的 Streamlit 物件。
    :param image (numpy 陣列): 影片幀的 numpy 陣列。
    :param task_type (string): YOLO 任務類型。
    :return: 無返回值。
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
       
    # Predict the objects in the image using YOLO11 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='偵測結果影片',
                   channels="BGR",
                   use_container_width=True
                   )


@st.cache_resource
def load_model(model_path):
    """
    在指定 model_path 目錄下載 YOLO 模型。

    參數：

    model_path (str): YOLO 模型檔案的路徑。

    返回： YOLO 模型。
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model, task_type):
    """
    執行上傳圖片的推論
    :param conf (float): 物體偵測的信心指數閾值。
    :param model (YOLO11): YOLO11 模型的實例。
    :param task_type (string): YOLO 任務類型。
    :return: 無返回值。
    """
    source_img = st.sidebar.file_uploader(
        label="選擇圖檔...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="上傳圖檔",
                use_container_width=True
            )

    if source_img:
        if st.button("執行"):
            with st.spinner("執行中..."):
                if task_type == "物體偵測": 
                    res = model.predict(uploaded_image,
                                        conf=conf)                    
                    res_plotted = res[0].plot()[:, :, ::-1]
                    with col2:
                        st.image(res_plotted,
                                 caption="偵測結果影像",
                                 use_container_width=True)
                        try:
                            with st.expander("偵測結果"):
                                #boxes = res[0].boxes
                                #for box in boxes:
                                #    st.write(box.xywh)
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
                elif task_type == "姿態評估":
                    res = model.predict(uploaded_image,
                                        conf=conf)
                    res_plotted = res[0].plot()[:, :, ::-1]
                    with col2:
                        st.image(res_plotted,
                                 caption="偵測結果影像",
                                 use_container_width=True)
                        try:
                            with st.expander("偵測結果"):
                                for r in res:
                                    boxes = r.boxes
                                    kps = r.keypoints
                                    for idx, p in enumerate(kps):
                                        list_p = p.data.tolist()
                                        st.write("姿態評估偵測的17個關鍵點:", idx)
                                        for i, point in enumerate(list_p[0]):
                                            st.write(f"關鍵點: {i}: ({int(point[0])}, {int(point[1])})")
                                        st.write("------------------------")    
                                            
                        except Exception as ex:
                            st.write("尚未上傳圖檔!")
                            st.write(ex)                    

def infer_uploaded_video(conf, model, task_type):
    """
    執行上傳影片檔的推論
    :param conf (float): 物體偵測的信心指數閾值。
    :param model (YOLO11): YOLO11 模型的實例。
    :param task_type (string): YOLO 任務類型。
    :return: 無返回值。
    """
    source_video = st.sidebar.file_uploader(
        label="選擇上傳影片檔..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("執行"):
            with st.spinner("執行中..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image,
                                                     task_type
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"載入影片檔錯誤: {e}")


def infer_uploaded_webcam(conf, model, task_type):
    """
    執行網路攝影機Webcam的推論
    :param conf (float): 物體偵測的信心指數閾值。
    :param model (YOLO11): YOLO11 模型的實例。
    :param task_type (string): YOLO 任務類型。
    :return: 無返回值。
    """
    try:
        flag = st.button(
            label="停止執行"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image,
                    task_type
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"載入網路攝影機錯誤: {str(e)}")
        
def infer_uploaded_IPcam(conf, model, task_type):
    """
    執行網路監控攝影機IP Camera的推論
    :param conf (float): 物體偵測的信心指數閾值。
    :param model (YOLO11): YOLO11 模型的實例。
    :param task_type (string): YOLO 任務類型。
    :return: 無返回值。
    """
    # tw live可搜尋台灣即時影像：https://trafficvideo2.tainan.gov.tw/b596d902
    # 在側邊欄中輸入 URL
    url = st.sidebar.text_input("輸入URL")
    if url:
        st.write(f"您輸入的URL是: {url}")
        try:
            flag = st.button(
                label="停止執行"
            )
            vid_cap = cv2.VideoCapture(url)  # IP camera
            st_frame = st.empty()
            while not flag:
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(
                        conf,
                        model,
                        st_frame,
                        image,
                        task_type
                    )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.error(f"載入網路攝影機錯誤: {str(e)}")        
    else:
        st.error(f"請在側邊欄[輸入URL]欄位輸入網路監控攝影機IP Camera的URL網址後, 按Enter鍵")

