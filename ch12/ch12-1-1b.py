import win32com
from win32com.client import Dispatch
import os

app = Dispatch("Word.Application")
app.Visible = 1
app.DisplayAlerts = 0
docx = app.Documents.Add()
pos = docx.Range(0, 0)
pos.InsertBefore("Python程式設計")
docx.SaveAs(os.getcwd()+"\\test2.docx")
docx.Close()
app.Quit()
