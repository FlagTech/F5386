import os
import subprocess

def kill_streamlit():
    try:
        # Windows 系統使用 taskkill 命令結束 streamlit 進程
        result = subprocess.run(
            ["taskkill", "/f", "/im", "streamlit.exe"], 
            capture_output=True, 
            text=True
        )
        
        # 輸出執行結果
        if result.returncode == 0:
            print("成功結束 Streamlit 進程!")
            print(result.stdout)
        else:
            print("嘗試結束 Streamlit 進程時出現錯誤:")
            print(result.stderr)
            
    except Exception as e:
        print(f"執行命令時發生錯誤: {e}")

if __name__ == "__main__":
    print("正在嘗試結束所有 Streamlit 進程...")
    kill_streamlit()