import cv2

# 建立滑鼠控制的回撥函數
def select_region(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end
    global selecting, image_temp

    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        x_start, y_start = x, y
        image_temp = image.copy()
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            x_end, y_end = x, y
            image_temp = image.copy()
            cv2.rectangle(image_temp, (x_start, y_start),
                          (x_end, y_end), (0, 255, 0), 2)
            cv2.imshow('Image', image_temp)
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        x_end, y_end = x, y
        apply_mosaic_to_selected_region(image_temp, x_start,
                                        y_start, x_end, y_end)

# 在指定區域套用馬賽克效果
def apply_mosaic_to_selected_region(image, x_start, y_start,
                                    x_end, y_end):
    x1, y1 = min(x_start, x_end), min(y_start, y_end)
    x2, y2 = max(x_start, x_end), max(y_start, y_end)
    roi = image[y1:y2, x1:x2]
    small = cv2.resize(roi, (max(1, int((x2-x1) * 0.1)),
                             max(1, int((y2-y1) * 0.1))),
                       interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (x2-x1, y2-y1),
                        interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = mosaic
    cv2.imshow('Image', image)

image = cv2.imread("Happy.jpg")
x_start, y_start, x_end, y_end = -1, -1, -1, -1
selecting = False
image_temp = image.copy()

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", select_region)

while True:
    cv2.imshow("Image", image_temp)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
