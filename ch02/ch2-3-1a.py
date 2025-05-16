import cv2

cap = cv2.VideoCapture("YouTube.mp4")

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("影格尺寸:", width, "x", height)
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
codec = (chr(fourcc&0xFF)+chr((fourcc>>8)&0xFF)+
        chr((fourcc>>16)&0xFF)+chr((fourcc>>24)&0xFF))
print("Codec編碼:", codec)
