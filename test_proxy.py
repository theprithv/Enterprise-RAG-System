import requests
import json
import time

url = "http://127.0.0.1:8001/v1/chat/completions"
headers = {
    "X-OpenWebUI-User-Email": "finance@nexacloud.com",
    "Content-Type": "application/json"
}
payload = {
    "messages": [
        {"role": "user", "content": "What is Q1 revenue?"}
    ]
}

print(f"Sending simulated Open WebUI request to {url} as finance@nexacloud.com...")
try:
    response = requests.post(url, headers=headers, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
