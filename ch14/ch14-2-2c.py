from shapely.geometry import Polygon
from shapely.geometry import mapping
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

# 第一個多邊形的頂點座標
polygon1_coords = [(1, 1), (4, 1), (4, 4), (1, 4), (1, 1)]
polygon1 = Polygon(polygon1_coords)
print("多邊形1的面積:", polygon1.area)
# 第二個多邊形的頂點座標
polygon2_coords = [(3, 3), (6, 3), (6, 6), (3, 6), (3, 3)]
polygon2 = Polygon(polygon2_coords)
print("多邊形2的面積:", polygon2.area)
# 檢查是否有重疊
if polygon1.intersects(polygon2):
    print("多邊形1和2有重疊區域！")
    # 計算重疊的面積
    overlap_area = polygon1.intersection(polygon2)
    print("重疊區域的面積:", overlap_area.area)
else:
    print("兩個多邊形沒有重疊區域。")
# 使用 Matplotlib 視覺化多邊形
fig, ax = plt.subplots()
# 繪製第一個多邊形
patch1 = MplPolygon(polygon1.exterior.coords, closed=True,
                    edgecolor="blue", facecolor="cyan",
                    alpha=0.5, label="Polygon 1")
ax.add_patch(patch1)
# 繪製第二個多邊形
patch2 = MplPolygon(polygon2.exterior.coords, closed=True,
                    edgecolor="red", facecolor="orange",
                    alpha=0.5, label="Polygon 2")
ax.add_patch(patch2)
# 如果有重疊，繪製重疊區域
if polygon1.intersects(polygon2):
    overlap_patch = MplPolygon(mapping(overlap_area)["coordinates"][0],
                               closed=True, edgecolor="purple",
                               facecolor="magenta", alpha=0.6,
                               label="Overlapping Areas")
    ax.add_patch(overlap_patch)
# 調整圖表
ax.set_xlim(0, 7)
ax.set_ylim(0, 7)
ax.set_aspect("equal", adjustable="box")
ax.legend()
plt.grid(True)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polygons and Their Overlapping Areas")
plt.show()
