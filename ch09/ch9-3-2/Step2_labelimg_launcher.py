import os
import subprocess

def run_labelImg_script():
    # 建立執行 Python 腳本的路徑
    script_path = os.path.join("YOLO_labelImg", "labelImg.py")    
    # 確認 Python 腳本存在
    if not os.path.exists(script_path):
        print(f"{script_path} 不存在...")
        return    
    # 執行 Python 腳本
    try:
        subprocess.run(["python", script_path], check=True)
        print(f"成功執行 {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"錯誤: 執行 {script_path}: {e} 發生錯誤...")

# 執行LabelImg
run_labelImg_script()
