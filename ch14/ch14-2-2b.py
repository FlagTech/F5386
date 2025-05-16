from shapely.geometry import Polygon

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
