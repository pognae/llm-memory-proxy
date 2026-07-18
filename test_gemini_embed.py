import os
import requests

gemini_key = os.getenv("GEMINI_API_KEY")
url = "https://generativelanguage.googleapis.com/v1beta/openai/embeddings" # wait, openai python client hits /v1/embeddings
# Let's just use openai python client
from openai import OpenAI

client = OpenAI(
    api_key=gemini_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

for model_name in ["text-embedding-004", "embedding-001", "models/text-embedding-004", "models/embedding-001"]:
    try:
        response = client.embeddings.create(
            input="Your text string goes here",
            model=model_name
        )
        print(f"SUCCESS with {model_name}")
        break
    except Exception as e:
        print(f"Failed with {model_name}: {e}")
