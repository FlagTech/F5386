hour = int(input("請輸入24小時制==> ")) 

hour = hour-12 if hour >= 12 else hour 
print("12小時制 =", hour)
