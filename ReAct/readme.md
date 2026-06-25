현업에서 발생하는 에이전트의 오작동, 네트워크 병목, JSON 파싱 에러를 완벽하게 제어하려면 프레임워크(LangChain)의 추상화 뒤에 숨겨진 Raw API의 날것 그대로의 통신 구조를 반드시 먼저 뜯어보고 코딩해야 합니다.
랭체인이 내부적으로 숨겨놓은 while 루프와 Tool Calling 메커니즘을 순수 파이썬 코드(Raw API)로만 구현한 백엔드 인터널 샘플을 짜왔습니다.
이 코드를 한 라인씩 뜯어보시면, LLM 에이전트가 어떤 데이터(JSON)를 주고받으며 핑퐁을 치는지 완전히 감이 오실 겁니다.


### 랭체인 없이 순수 API로 구현한 ReAct 에이전트 인터널 ###
파이썬 기본 openai 라이브러리만 사용하여 에이전트의 '판단-행동-관찰' 루프를 직접 코딩한 예제입니다.
```
import json
import os
from openai import OpenAI

# 1. 원시 클라이언트 선언
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"))

# ==========================================
# [STEP 1] 원시 파이썬 함수 (도구) 정의
# ==========================================
def get_stock_price(ticker: str) -> float:
    mock_db = {"AAPL": 180.0, "TSLA": 175.0, "NVDA": 900.0}
    return mock_db.get(ticker.upper(), 0.0)

def calculate_profit(buy_price: float, current_price: float) -> float:
    return round(((current_price - buy_price) / buy_price) * 100, 2)

# 파이썬 함수와 이름을 매핑해둔 라우팅 테이블
AVAILABLE_TOOLS = {
    "get_stock_price": get_stock_price,
    "calculate_profit": calculate_profit
}

# ==========================================
# [STEP 2] LLM에게 매번 실어 보낼 "도구 명세서(JSON Schema)"
# ==========================================
tools_spec = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "실시간 주식 가격을 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_profit",
            "description": "매수 가격과 현재 가격을 비교하여 수익률(%)을 계산합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "buy_price": {"type": "number"},
                    "current_price": {"type": "number"}
                },
                "required": ["buy_price", "current_price"]
            }
        }
    }
]

# ==========================================
# [STEP 3] 대화 히스토리 및 루프 제어 (핵심 디버깅 영역)
# ==========================================
messages = [
    {"role": "system", "content": "너는 도구를 사용하여 단계별로 계산을 수행하는 에이전트다."},
    {"role": "user", "content": "내가 NVDA 주식을 500달러에 샀는데 현재 주가 확인해서 내 수익률을 계산해줘."}
]

MAX_LOOPS = 5  # 무한 루프 방어벽
loop_count = 0

print("🚀 Raw API 에이전트 제어 루프 가동\n")

while loop_count < MAX_LOOPS:
    loop_count += 1
    print(f"--- [ROUND {loop_count}] LLM 인퍼런스 요청 중... ---")
    
    # 🔍 디버깅 포인트: LLM에 '대화 기록'과 '도구 명세서'를 함께 실어 보냄 (히스토리가 누적됨)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools_spec,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # LLM의 출력을 히스토리에 누적 (다음 라운드에 기억하게 하기 위함)
    messages.append(response_message)

    # 판단 케이스 1: LLM이 "더 이상 도구 호출 안 해도 된다"고 판단하면 최종 답변 출력 후 종료
    if not tool_calls:
        print(f"\n🏁 [최종 답변 반환]:\n{response_message.content}")
        break

    # 판단 케이스 2: LLM이 "도구 실행이 필요하다"고 판단하여 JSON 신호(tool_calls)를 뱉은 경우
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"🤖 LLM의 판단 -> 도구 호출 요구: {function_name}({function_args})")
        
        # 실제 파이썬 함수 동적 실행 (라우팅)
        run_function = AVAILABLE_TOOLS.get(function_name)
        if run_function:
            # 주가 조회나 계산기 함수를 실제로 실행하여 리턴값을 얻음
            observation = run_function(**function_args)
            print(f"💻 파이썬 코드 실행 결과 (Observation): {observation}")
            
            # 🔍 핵심 인터널: 함수 실행 결과를 'tool' 역할로 메시지 스택에 다시 집어넣음
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(observation),
            })
        else:
            print(f"❌ 에러: {function_name} 이라는 도구는 존재하지 않습니다.")
            
    print("\n" + "="*50 + "\n")

else:
    print("⚠️ 무한 루프 방어벽 작동: 최대 루프 횟수를 초과했습니다.")
```
