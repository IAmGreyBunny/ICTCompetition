import requests

json_data = {"data": "Hi this is a test"}

response = requests.post("http://127.0.0.1:5000/bertriage_api", json=json_data)
print(response.text)
