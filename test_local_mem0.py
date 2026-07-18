import os
from mem0 import Memory

os.environ["QDRANT_URL"] = "http://localhost:6333" 
os.environ["QDRANT_API_KEY"] = "fake"

# Get real Gemini API key from environment if possible, or use fake
gemini_key = os.getenv("GEMINI_API_KEY", "fake")

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "url": os.getenv("QDRANT_URL"),
            "api_key": os.getenv("QDRANT_API_KEY"),
            "collection_name": "developer_memory",
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gemini-1.5-flash",
            "api_key": gemini_key,
            "openai_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-004",
            "api_key": gemini_key,
            "openai_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
        }
    }
}

try:
    m = Memory.from_config(config)
    print("Success loading memory instance with OpenAI-compatible Gemini provider!")
except Exception as e:
    print(f"Error: {str(e)}")
