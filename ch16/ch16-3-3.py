import ollama

prompt1 = "請分析句子 '"
prompt2 = "' 的語意含義，是「開」還是「關」。答案僅用一個字表達。"
question = prompt1 + "Turn on the lights in the room." + prompt2

response = ollama.chat(
    model = "llama3.1:8b",
    messages = [
         {"role": "system", "content": "你是語意分析機器人"},
         {"role": "user", "content": question}
    ])
print("Q:", question)
print(response['message']['content'])
