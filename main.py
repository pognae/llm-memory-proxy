from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import openai

# 런타임 패치: NVIDIA NIM 임베딩 모델(nv-embedqa 등)은 'input_type' 파라미터를 필수로 요구합니다.
# mem0가 내부적으로 사용하는 openai 클라이언트를 가로채서 강제로 input_type="query"를 주입합니다.
original_create = openai.resources.Embeddings.create

def patched_create(self, *args, **kwargs):
    if "extra_body" not in kwargs or kwargs["extra_body"] is None:
        kwargs["extra_body"] = {}
    kwargs["extra_body"]["input_type"] = "query"
    return original_create(self, *args, **kwargs)

openai.resources.Embeddings.create = patched_create

load_dotenv()

app = FastAPI()

m = None

def get_memory():
    global m
    if m is None:
        try:
            from mem0 import Memory
            qdrant_url = os.getenv("QDRANT_URL")
            if not qdrant_url:
                raise ValueError("QDRANT_URL이 설정되지 않았습니다. .env 파일을 확인하세요.")

            # NVIDIA NIM을 OpenAI 호환 API 규격으로 사용합니다.
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "url": qdrant_url,
                        "api_key": os.getenv("QDRANT_API_KEY"),
                        "collection_name": "developer_memory_1024",
                    }
                },
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": "meta/llama3-70b-instruct",
                        "api_key": os.getenv("NVIDIA_API_KEY"),
                        "openai_base_url": "https://integrate.api.nvidia.com/v1"
                    }
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": "nvidia/nv-embedqa-e5-v5",
                        "api_key": os.getenv("NVIDIA_API_KEY"),
                        "openai_base_url": "https://integrate.api.nvidia.com/v1"
                    }
                }
            }
            m = Memory.from_config(config)
        except Exception as e:
            raise RuntimeError(f"메모리 DB 초기화 실패: {str(e)}")
    return m

@app.get("/")
def read_root():
    return {"status": "Memory Proxy 서버가 정상적으로 실행 중입니다!"}

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
async def chat_with_memory(req: ChatRequest):
    print(f"--- [입력] 사용자: {req.user_id} ---")
    print(f"메시지: {req.message}")
    try:
        mem0_instance = get_memory()
        relevant_memories = mem0_instance.search(req.message, filters={"user_id": req.user_id})
        memory_context = "\n".join([mem['memory'] for mem in relevant_memories])
        
        system_prompt = f"""
        당신은 나만을 위한 전담 코딩 어시스턴트입니다.
        아래의 [과거 기억]을 참고하여 답변하세요.
        
        [과거 기억]
        {memory_context}
        """
        
        mem0_instance.add(req.message, user_id=req.user_id)
        
        response_data = {
            "status": "success",
            "injected_memory": memory_context,
            "final_prompt_to_llm": f"{system_prompt}\n\n[사용자 질문]: {req.message}"
        }
        print(f"--- [출력] 프롬프트에 주입된 메모리 ---")
        print(memory_context if memory_context else "(없음)")
        print(f"--------------------------------------")
        return response_data
    except Exception as e:
        print(f"--- [에러 발생] ---")
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))