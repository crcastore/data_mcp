import os

import httpx
from openai import OpenAI

TOKEN = os.environ["GITHUB_TOKEN"]
BASE = "https://models.inference.ai.azure.com"

if __name__ == "__main__":
    client = OpenAI(api_key=TOKEN, base_url=BASE)

    resp = httpx.get(f"{BASE}/models", headers={"Authorization": f"Bearer {TOKEN}"})
    resp.raise_for_status()
    for model in resp.json():
        print(model["id"])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say hello like a pirate"}
        ],
    )
    
    print(response.choices[0].message.content)



