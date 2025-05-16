from cvzone.HandTrackingModule import HandDetector
import cv2, os
from win32com.client import Dispatch

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(detectionCon=0.5, maxHands=1)
app = Dispatch("PowerPoint.Application")
app.Visible = 1
pptx = app.Presentations.Open(os.getcwd()+"/Turtle.pptx")
msg = "Stop"
isRun = False
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        bbox = hand["bbox"]        
        fingers = detector.fingersUp(hand)
        totalFingers = fingers.count(1)
        if totalFingers == 0: 
            msg = "Stop"
        if totalFingers == 1:
            if fingers[0] == 1:
                if msg != "Next" and isRun:
                    msg = "Next"
                    pptx.SlideShowWindow.View.Next()
            if fingers[1] == 1:
                if msg != "Previous" and isRun:
                    msg = "Previous"
                    pptx.SlideShowWindow.View.Previous()    
        if totalFingers == 2:
            if fingers[1] == 1 and fingers[2] == 1:
                msg = "Run"
                pptx.SlideShowSettings.Run()
                isRun = True
        if totalFingers == 3:
            if isRun:
                pptx.SlideShowWindow.View.Exit()
            pptx.Close()    
            break
        cv2.putText(img, msg, (bbox[0]+100,bbox[1]-30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    else:
        msg = "Stop"
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
os.system('taskkill /F /IM POWERPNT.EXE')  #app.Quit() not work
cap.release()
cv2.destroyAllWindows()
