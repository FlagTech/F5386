import os

py_file = "ch16-5.py"

def run_streamlit_app():
    # 建立 Streamlit 應用的命令
    command = "streamlit run " + py_file
    # 執行 streamlit run 命令
    os.system(command)

if __name__ == "__main__":
    run_streamlit_app()
