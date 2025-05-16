from shapely.geometry import Polygon
import matplotlib.pyplot as plt

# 定義多邊形的頂點座標
coordinates = [(0, 0), (2, 0),
               (1.5, 1.5), (0.5, 2),
               (0, 0)]  # 最後的 (0, 0) 將多邊形封閉
polygon = Polygon(coordinates)  # 使用 Shapely 的 Polygon 類別建立多邊形
# 提取多邊形的 x 和 y 座標來進行繪製
x, y = polygon.exterior.xy  # 獲取多邊形邊界的 x 和 y 座標
# 使用 matplotlib 繪製多邊形
plt.figure()
plt.fill(x, y, alpha=0.5, fc='blue', ec='black')  # 繪製多邊形，填充藍色
plt.title("Shapely Polygon")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
