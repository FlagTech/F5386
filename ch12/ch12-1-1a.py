import win32com
from win32com.client import Dispatch
import os

app = Dispatch("Word.Application")
app.Visible = 1
app.DisplayAlerts = 0
docx = app.Documents.Open(os.getcwd()+"\\test.docx")
print ("段落數: ", docx.Paragraphs.count)
for i in range(len(docx.Paragraphs)):
    para = docx.Paragraphs[i]
    print(para.Range.text)
docx.Close()
app.Quit()
