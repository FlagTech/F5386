from shapely.geometry import Polygon
import cv2, numpy as np

# 繪製熱區域邊界框
def drawArea(frame, area, color, t):
    v =  np.array(area, np.int32)
    cv2.polylines(frame, [v], isClosed=True, color=color, thickness=t)
        
# 回傳物體target進入hotzone區域的重疊百分比
def inHotZonePercent(target, hotzone):
    p1= [[target[0], target[1]], [target[2], target[1]],
         [target[2], target[3]], [target[0], target[3]]]
    poly1 = Polygon(p1)
    poly2 = Polygon(hotzone)
    overlap_percent = overlap_percentage(poly1, poly2)
    
    return overlap_percent

# poly1面積作為分母來計算重疊面積, 物體poly1進入poly2區域的比例
def overlap_percentage(poly1, poly2):
    intersection_area = poly1.intersection(poly2).area
    poly1_area = poly1.area
    overlap_percent = (intersection_area / poly1_area) * 100
    
    return overlap_percent
