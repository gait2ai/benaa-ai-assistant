import requests
while True:
    requests.get("https://your-app-name.up.railway.app/health")
    time.sleep(300)  # كل 5 دقائق