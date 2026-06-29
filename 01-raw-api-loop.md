현업에서 발생하는 에이전트의 오작동, 네트워크 병목, JSON 파싱 에러를 완벽하게 제어하려면 프레임워크(LangChain)의 추상화 뒤에 숨겨진 Raw API의 날것 그대로의 통신 구조를 반드시 먼저 뜯어보고 코딩할 필요가 있다. 
아래 코드는 랭체인이 내부적으로 숨겨놓은 while 루프와 Tool Calling 메커니즘을 순수 파이썬 코드(Raw API)로만 구현한 백엔드 인터널 샘플이다.

### 랭체인 없이 순수 API로 구현한 ReAct 에이전트 인터널 ###
파이썬 기본 openai 라이브러리만 사용하여 에이전트의 '판단-행동-관찰' 루프를 직접 코딩한 예제.
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


* 메시지 스택(messages)의 누적 과정 추적:  
핑퐁이 돌 때마다 messages 리스트 안에 assistant가 뱉은 JSON과 tool이 뱉은 결과값이 차곡차곡 쌓이는 게 눈으로 보입니다. LLM이 아까 얻은 주가 900.0을 어떻게 기억하고 다음 단계로 넘기는지 그 비밀이 이 리스트의 누적에 있다.
	
* 에러 핸들링 지점의 명확성:  
만약 LLM이 오작동해서 function_args에 이상한 텍스트를 주면 json.loads()나 run_function(**function_args)에서 파이썬 에러가 발생한다. 랭체인을 쓰면 내부에서 에러가 묻혀서 알기 어렵지만, Raw API에서는 정확히 어느 라인에서 LLM 뱉은 값이 망가졌는지 바로 디버깅(try-except 래핑)이 가능하다.

* 결정론적 제어의 시작:
while 루프 중간에 개발자가 if loop_count == 2: 같은 하드코딩 조건문을 걸어서 강제로 루프를 꺾거나 멈추는 식의 **완벽한 통제(Deterministic Control)**를 직접 구현해볼 수 있다.


### 🤖 LLM 에이전트 및 도구 호출(Tool Calling) 원리 ###
* tool_calls 내부 동작: LLM이 실제로 도구를 실행하는 것이 아니라, 시스템 프롬프트 뒤에 붙은 도구 명세서를 보고 "이 도구를 실행해달라"고 청구서(JSON)를 적어 보내면 파이썬 SDK가 객체로 파싱해 주는 구조입니다.
* 조건반사적 판단: LLM은 자아성찰을 통해 모른다고 판단하는 것이 아니라, '현재 주가' 같은 실시간 정보 요청 맥락이 들어오면 도구 명세서와 수학적으로 매칭하도록 혹독하게 학습(Fine-tuning)된 결과에 따라 움직입니다.
* 무한한 경우의 수 커버: 인간이 If-Else 문으로 모든 경우의 수를 짤 필요 없이, LLM이 문맥의 '의도'를 뇌 속 지도(벡터 공간)의 좌표로 파악하기 때문에 수만 가지 질문 패턴을 알아서 도구와 연결합니다.
* 도구가 없을 때의 예외 처리: 연관된 도구가 없다면 LLM은 tool_calls를 비워두고, 일반 텍스트 답변(content) 창을 채워 "도구가 없어 답변이 어렵다"는 식으로 대응하며, 코드 루프는 이를 감지해 안전하게 탈출합니다.

### ⚠️ 에이전트 시스템의 한계와 리스크 ###
* 확률 기계의 엣지 케이스: LLM은 엉뚱한 도구를 고르거나(Wrong Pick), 가짜 도구를 지어내거나(Hallucination), 인자 형식을 망가뜨리는 치명적인 실수를 반드시 저지릅니다.
* 시스템적 보완 (제어 루프): 이를 막기 위해 파이썬 레벨에서 화이트리스트 검증, 스키마 검사, 그리고 에러를 다시 LLM에게 먹여 스스로 고치게 만드는 while 리트라이 루프(Deterministic Control)가 필수적입니다.
* 지능(IQ)이 떨어지는 멍청한 LLM을 에이전트의 뇌로 쓰는 순간, 무한 루프에 갇혀 API 비용만 날리고 프로젝트가 나락 가기 때문에 현업에서는 무조건 최고 스펙의 LLM을 레일(프레임워크) 위에 묶어서 제어합니다.

