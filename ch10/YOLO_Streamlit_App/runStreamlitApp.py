import os
# ch10/YOLO_Streamlit_App專案目錄
py_file = "app.py"

def run_streamlit_app():
    # 建立 Streamlit 應用的命令
    command = "streamlit run " + py_file
    # 執行命令
    os.system(command)

if __name__ == "__main__":
    run_streamlit_app()
