import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

model = YOLO("yolo12n.pt")

def _display_detected_frames(conf, model, st_frame, image):
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    res = model.predict(image, conf=conf)
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='偵測結果影片',
                   channels="BGR",
                   use_container_width=True)

# Streamlit 網頁介面
st.title("上傳影片檔執行YOLO物體偵測")
# 上傳影片檔
source_video = st.file_uploader(
    label="選擇上傳影片檔..."
)
if source_video:
    st.video(source_video)
    if st.button("執行"):
        with st.spinner("執行中..."):
            try:
                tfile = tempfile.NamedTemporaryFile()
                tfile.write(source_video.read())
                vid_cap = cv2.VideoCapture(tfile.name)
                st_frame = st.empty()
                while (vid_cap.isOpened()):
                    success, image = vid_cap.read()
                    if success:
                        _display_detected_frames(0.6, model,
                                                 st_frame, image)
                    else:
                        vid_cap.release()
                        break
            except Exception as e:
                st.error(f"載入影片檔錯誤: {e}")

