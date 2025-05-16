import win32com
from win32com.client import Dispatch
import os

app = Dispatch("Excel.Application")
app.Visible = 1
app.DisplayAlerts = 0
xlsx = app.Workbooks.Add()
sheet = xlsx.Worksheets(1)
sheet.Cells(1, 1).Value = 1
sheet.Cells(1, 2).Value = 2
sheet.Cells(2, 1).Value = 3
sheet.Cells(2, 2).Value = 4
sheet.Cells(3, 1).Value = 5
sheet.Cells(3, 2).Value = 6
xlsx.SaveAs(os.getcwd()+"\\test2.xlsx")
xlsx.Close(False)
app.Quit()
