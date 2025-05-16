from multiprocessing import freeze_support
from ultralytics import YOLO
import os
import torch

# 定義模型訓練的參數
model_size = "s" # YOLO 模型尺寸是"n", "s", "m", "l", "x"
version = "12"   # YOLO 版本是"12", "11", "v8"
epochs = 50      # 訓練周期
batch = 16       # 批次大小，每次迭代中使用的數據樣本數
imgsz = 640      # 圖片尺寸，指圖像在訓練時會被調整到的尺寸
plots = True     # 是否在訓練過程中繪製圖表，用於可視化訓練過程

def main():
    global model_size, version, epochs, batch, imgsz, plots
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_directory = os.getcwd()  # 使用 getcwd() 函數取得目前的工作目錄
    print("目前的工作目錄是：", current_directory)  # 輸出目前的工作目錄
    # 第一種方式: 使用模型架構從頭開始訓練
    # model = YOLO('yolo'+version+model_size+'.yaml') 
    # 第二種方式: 使用預訓練模型進行訓練, 建議方式
    model = YOLO("yolo"+version+model_size+".pt") 
    # 第三種方式: 使用 YAML 文件建構模型架構，然後從 .pt 載入預訓練權重。
    # 此方式允許建構模型時使用自定義配置.yaml 檔案（例如層數、節點數等）， 然後再載入預訓練模型的權重進行微調。
    # model = YOLO('yolo'+version+model_size+'.yaml').load('yolo'+version+model_size+'.pt')
    model.to(device)
    YAML_FILE_PATH = current_directory + "\data.yaml"
    results = model.train(data=YAML_FILE_PATH, epochs=epochs, batch=batch,
                          imgsz=imgsz, plots=plots, device=device)
    print("模型訓練結果============")
    print("map50-95:", results.box.map)    # map50-95
    print("map50:", results.box.map50)     # map50
    print("map75:", results.box.map75)     # map75
    print("每一分類的map50-95:", results.box.maps) 
    metrics = model.val()
    print("模型驗證結果============")
    print("map50-95:", metrics.box.map)    # map50-95
    print("map50:", metrics.box.map50)     # map50
    print("map75:", metrics.box.map75)     # map75
    print("每一分類的map50-95:", metrics.box.maps) 

if __name__ == "__main__":
    # 確保只有在直接執行主模組時，才會執行freeze_support()方法，以便可以正確的啟動子行程。
    freeze_support() 
    main()
    
# 在 Windows 系統上，multiprocessing 模組使用 spawn 方法來啟動子行程，
# 這意味著每個子行程都是從頭開始啟動的，而不是從父行程複製的。
# 因此，必須在主模組中使用freeze_support()來啟動子行程。