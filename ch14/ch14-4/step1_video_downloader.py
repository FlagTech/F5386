import requests

# 影片檔的URL網址
url = "https://github.com/fchart/PythonCV/raw/refs/heads/main/media_files/street.mp4"
# 下載的檔名
output_file = "../media/street.mp4"
# ------------------------------------------
# 發送HTTP GET請求，使用stream模式
response = requests.get(url, stream=True)
# 獲取檔案總大小（位元組）
total_size = int(response.headers.get('content-length', 0))
downloaded_size = 0
# 開始下載並顯示進度
with open(output_file, "wb") as file:
    for data in response.iter_content(chunk_size=1024):
        file.write(data)
        downloaded_size += len(data)
        # 計算與顯示進度百分比
        percent_done = (downloaded_size / total_size) * 100
        print(f"\rDownloading: {percent_done:.2f}% [{downloaded_size}/{total_size} bytes]", end="")

print(f"\n影片已下載並儲存為 {output_file}")
