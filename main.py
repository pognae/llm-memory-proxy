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
    try:
        mem0_instance = get_memory()
        relevant_memories = mem0_instance.search(req.message, user_id=req.user_id)
        memory_context = "\n".join([mem['memory'] for mem in relevant_memories])
        
        system_prompt = f"""
        당신은 나만을 위한 전담 코딩 어시스턴트입니다.
        아래의 [과거 기억]을 참고하여 답변하세요.
        
        [과거 기억]
        {memory_context}
        """
        
        mem0_instance.add(req.message, user_id=req.user_id)
        
        return {
            "status": "success",
            "injected_memory": memory_context,
            "final_prompt_to_llm": f"{system_prompt}\n\n[사용자 질문]: {req.message}"
        }
    except Exception as e:
        # 어디서 에러가 났는지 브라우저에 표시해줍니다.
        raise HTTPException(status_code=500, detail=str(e))