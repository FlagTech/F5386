import requests

# 影片檔的URL網址
url = "https://github.com/fchart/PythonCV/raw/refs/heads/main/media_files/starwars.mp4"
# 下載的檔名
output_file = "starwars.mp4"
# ------------------------------------------
response = requests.get(url)
with open(output_file, "wb") as file:
    file.write(response.content)

print(f"影片已下載並儲存為 {output_file}")
