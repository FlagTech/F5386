#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     config.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Update Date:   2024/11/22 修改和中文化 By 陳會安
   @Description: configuration file
-------------------------------------------------
"""
from pathlib import Path
import sys

# 取得目前檔案的絕對路徑
file_path = Path(__file__).resolve()

# 取得目前檔案的父路徑
root_path = file_path.parent

# 如果根路徑不存在 sys.path 串列, 在 sys.path路徑加上根路徑
if root_path not in sys.path:
    sys.path.append(str(root_path))

# 取得根目錄和目前工作目錄的相對目錄
ROOT = root_path.relative_to(Path.cwd())

# 資料來源
#SOURCES_LIST = ["Image", "Video", "Webcam"]
SOURCES_LIST = ["圖片檔", "影片檔", "網路攝影機Webcam", "網路監控攝影機IPcam"]

# YOLO 物體偵測模型設置
DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
YOLO11n = DETECTION_MODEL_DIR / "yolo12n.pt"
YOLO11s = DETECTION_MODEL_DIR / "yolo12s.pt"
YOLO11m = DETECTION_MODEL_DIR / "yolo12m.pt"
YOLO11l = DETECTION_MODEL_DIR / "yolo12l.pt"
YOLO11x = DETECTION_MODEL_DIR / "yolo12x.pt"
# YOLO 姿態評估模型設置
POSE_MODEL_DIR = ROOT / 'weights' / 'pos'
YOLO11n_pose = DETECTION_MODEL_DIR / "yolo11n-pose.pt"
YOLO11s_pose = DETECTION_MODEL_DIR / "yolo11s-pose.pt"
YOLO11m_pose = DETECTION_MODEL_DIR / "yolo11m-pose.pt"
YOLO11l_pose = DETECTION_MODEL_DIR / "yolo11l-pose.pt"
YOLO11x_pose = DETECTION_MODEL_DIR / "yolo11x-pose.pt"

DETECTION_MODEL_LIST = [
    "yolo12n.pt",
    "yolo12s.pt",
    "yolo12m.pt",
    "yolo12l.pt",
    "yolo12x.pt",
    "best.pt"]

POSE_MODEL_LIST = [
    "yolo11n-pose.pt",
    "yolo11s-pose.pt",
    "yolo11m-pose.pt",
    "yolo11l-pose.pt",
    "yolo11x-pose.pt"]
