import win32com
from win32com.client import Dispatch
import time, os

app = Dispatch("PowerPoint.Application")
app.Visible = 1
app.DisplayAlerts = 0
pptx = app.Presentations.Open(os.getcwd()+"\\test.pptx")
pptx.SlideShowSettings.Run()
time.sleep(1)
pptx.SlideShowWindow.View.Next()
time.sleep(1)
pptx.SlideShowWindow.View.Next()
time.sleep(1)
pptx.SlideShowWindow.View.Previous()
time.sleep(1)
pptx.SlideShowWindow.View.Exit()
os.system('taskkill /F /IM POWERPNT.EXE')  #app.Quit() not work
