import requests

url = "https://llm-memory-proxy.onrender.com/chat"
payload = {
    "user_id": "test_user",
    "message": "안녕하세요! 기억을 테스트하고 있습니다."
}

try:
    print(f"서버에 전송하는 요청: {payload}")
    response = requests.post(url, json=payload)
    response.raise_for_status()
    print("\n--- 서버 응답 성공 ---")
    print(response.json())
except requests.exceptions.RequestException as e:
    print("\n--- 서버 응답 실패 ---")
    print(f"Error: {e}")
    if response is not None:
        print(response.text)
