import win32com
from win32com.client import Dispatch
import os

app = Dispatch("Excel.Application")
app.Visible = 1
app.DisplayAlerts = 0
xlsx = app.Workbooks.Open(os.getcwd()+"\\test.xlsx")
sheet = xlsx.Worksheets(1)
row = sheet.UsedRange.Rows.Count
col = sheet.UsedRange.Columns.Count
print(row, col)
print("Cells(2, 1)=", sheet.Cells(2, 1).Value)
print("Cells(2, 2)=", sheet.Cells(2, 2).Value)
value = sheet.Range("A1:B3").Value
print(value)
xlsx.Close(False)
app.Quit()
