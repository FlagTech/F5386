import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
img = cv2.imread("images/number.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_w = img.shape[1]
img_h = img.shape[0]
boxes = pytesseract.image_to_boxes(img)
print(boxes)
for box in boxes.splitlines():
    box = box.split(" ")
    character = box[0]
    x = int(box[1])
    y = int(box[2])
    x2 = int(box[3])
    y2 = int(box[4])
    cv2.rectangle(img, (x, img_h - y),
                  (x2, img_h - y2), (0, 255, 0), 1)
    cv2.putText(img, character, (x, img_h - y2 - 10),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
