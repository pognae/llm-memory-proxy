from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# .env 명시적 로드 (파일이 없으면 에러가 나도록 절대 경로 사용 권장)
load_dotenv()

app = FastAPI()

# 1. 서버 시작 시 바로 DB에 붙지 않고 변수만 만들어 둡니다. (지연 로딩)
m = None

def get_memory():
    global m
    if m is None:
        try:
            from mem0 import Memory
            # 환경변수가 제대로 들어왔는지 확인
            qdrant_url = os.getenv("QDRANT_URL")
            if not qdrant_url:
                raise ValueError("QDRANT_URL이 설정되지 않았습니다. .env 파일을 확인하세요.")

            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "url": qdrant_url,
                        "api_key": os.getenv("QDRANT_API_KEY"),
                        "collection_name": "developer_memory",
                    }
                },
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": "gemini-1.5-flash",
                        "api_key": os.getenv("GEMINI_API_KEY"),
                        "openai_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
                    }
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": "embedding-001",
                        "api_key": os.getenv("GEMINI_API_KEY"),
                        "openai_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
                    }
                }
            }
            m = Memory.from_config(config)
        except Exception as e:
            raise RuntimeError(f"메모리 DB 초기화 실패: {str(e)}")
    return m

# 2. 서버가 정상적으로 켜졌는지 확인하기 위한 테스트 엔드포인트
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
        # mem0 최신 버전에 맞게 user_id를 filters를 통해 전달합니다.
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
        # 어디서 에러가 났는지 브라우저에 표시해줍니다.
        raise HTTPException(status_code=500, detail=str(e))