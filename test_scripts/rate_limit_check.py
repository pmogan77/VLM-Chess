import requests
import json

api_key = "..."
response = requests.get(
  url="https://openrouter.ai/api/v1/key",
  headers={
    "Authorization": f"Bearer {api_key}"
  }
)
print(json.dumps(response.json(), indent=2))