from shapely.geometry import Polygon

# 定義多邊形的頂點座標
coordinates = [(0, 0), (2, 0),
               (1.5, 1.5), (0.5, 2),
               (0, 0)]  # 最後的 (0, 0) 將多邊形封閉
polygon = Polygon(coordinates)  # 使用 Shapely 的 Polygon 類別建立多邊形
print("多邊形的面積:", polygon.area)  # 計算多邊形的面積
print("多邊形的周長:", polygon.length)  # 計算多邊形的周長
